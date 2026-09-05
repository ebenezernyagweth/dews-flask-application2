from pathlib import Path

from flask import Flask, jsonify

from flask import request

from Main_dash import run_full_pipeline
from main_muac import run_muac_pipelines
from main_muac import unzip_intermediary_datasets
from extract_latest_muac import extract_latest_MUAC
import os
import traceback

import requests
import pandas as pd
from threading import Lock, Thread
from datetime import datetime
from main_muac import zip_intermediary_datasets
import ee

app = Flask(__name__)

muac_pipeline_lock = Lock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
UPLOAD_FOLDER = os.path.join(BASE_DIR, "Kenya_MUAC_NDMA_implementation")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
EXPECTED_COLUMNS = [
    "MUACIndicatorID","QID","County","SubCounty","Ward","LivelihoodZone","Month","Year","HouseholdCode","ChildName","Gender","MUAC","MUAC_Color","AgeInMonths","LiveInHousehold","SufferedIllnesses",
    "InterviewDate","DivisionID","CountyID","SiteID","LivelihoodZoneID"
]

def safe_run_muac_pipelines():
    if not muac_pipeline_lock.acquire(blocking=False):
        print("MUAC pipeline already running — skipping")
        return

    try:
        print("Starting MUAC pipeline")
        run_muac_pipelines()
        print("MUAC pipeline finished")

    except Exception as e:
        print(f"MUAC pipeline failed: {e}")

    finally:
        muac_pipeline_lock.release()


@app.route("/")
def home():
 return "Welcome to the Milk Production API!"

@app.route("/service-api/v1/muac/process/modeling_validation", methods=["GET", "POST"])
def api_trigger_muac_pipelines():
    """
    Wrapper endpoint that delegates to the main MUAC modeling pipeline endpoint.
    Useful for calling from other services or scheduled jobs.
    """
    today = datetime.now()
    if today.day < 20:
        return jsonify({
            "message": f"MUAC modeling pipeline can only be run from the 20th of each month onwards. "
                       f"Today is the {today.day}{'th' if 4 <= today.day <= 20 else ['st','nd','rd'][today.day % 10 - 1] if today.day % 10 in [1,2,3] else 'th'}. "
                       f"Please try again from the 20th — not all datasets are ready for download before this date."
        }), 425  # 425 Too Early

    # Fast rejection if already running
    if muac_pipeline_lock.locked():
        return jsonify({
            "message": "MUAC pipeline is already running"
        }), 409
        
    try:
        return jsonify({
            "message": "MUAC data processing and modeling started"
        }), 202
    except Exception as e:
        return jsonify({"error": f"Failed to trigger MUAC pipeline: {str(e)}"}), 500


@app.route("/service-api/v1/muac/process/modeling", methods=["GET", "POST"])
def api_run_muac_pipelines():
    # Check if today is on or after the 20th of the month
    """
    Wrapper endpoint that delegates to the main MUAC modeling pipeline endpoint.
    Useful for calling from other services or scheduled jobs.
    """
    today = datetime.now()
    if today.day < 20:
        return jsonify({
            "message": f"MUAC modeling pipeline can only be run from the 20th of each month onwards. "
                       f"Today is the {today.day}{'th' if 4 <= today.day <= 20 else ['st','nd','rd'][today.day % 10 - 1] if today.day % 10 in [1,2,3] else 'th'}. "
                       f"Please try again from the 20th — not all datasets are ready for download before this date."
        }), 425  # 425 Too Early

    # Fast rejection if already running
    if muac_pipeline_lock.locked():
        return jsonify({
            "message": "MUAC pipeline is already running"
        }), 409

    # Check Earth Engine tasks
    ee_busy, count = is_ee_busy()
    if ee_busy:
        return jsonify({
            "message": f"Earth Engine has {count} active task(s). Aborting pipeline run."
        }), 409      

    try:
        Thread(
            target=safe_run_muac_pipelines,
            daemon=True
        ).start()
        return jsonify({
            "message": "MUAC data processing and modeling started"
        }), 202
    except Exception as e:
        return jsonify({"error": str(e)}), 500
#
# @app.route("/service-api/v1/muac/process/modeling", methods=["GET", "POST"])
# def api_run_muac_pipelines():

#     # Fast rejection if already running
#     if muac_pipeline_lock.locked():
#         return jsonify({
#             "message": "MUAC pipeline is already running"
#         }), 409

#     try:
#         Thread(
#             target=safe_run_muac_pipelines,
#             daemon=True
#         ).start()

#         return jsonify({
#             "message": "MUAC data processing and modeling started"
#         }), 202

#     except Exception as e:
#         return jsonify({"error": str(e)}), 500


@app.route("/service-api/v1/muac/process/dashboard", methods=["GET", "POST"])
def api_start_muac_dashboard():
    try:
        # Call function
        run_full_pipeline()
        
        return jsonify({"message": "MUAC dashboard initiated. access dashboard at http://localhost:8080"}), 200

    except SystemExit as e:
        return jsonify({
            "error": "MUAC pipeline failed",
            "exit_code": e.code
        }), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/service-api/v1/muac/extract/database", methods=["GET", "POST"])
def extract_muac_data():
    """
    Route that extracts MUAC data from database instead of accepting file uploads.
    GET request now triggers the database extraction.
    """
    try:
        # Call the extraction function
        result_df, file_path = extract_latest_MUAC()
        
        # Check if extraction was successful
        if result_df.empty or file_path is None:
            return jsonify({
                "error": "No data could be extracted from the database. Please check if there are sufficient HHA records."
            }), 404
        
        # Build success message with data summary
        return jsonify({
            "message": f"MUAC data extracted successfully from database and saved as 'MUAC_Data.xlsx'.",
            "path": file_path,
            "records_count": len(result_df),
            "data_summary": {
                "total_records": len(result_df),
                "unique_children": result_df['MUACIndicatorID'].nunique(),
                "counties": result_df['County'].nunique(),
                "date_range": {
                    "earliest": result_df['InterviewDate'].min().strftime('%Y-%m-%d') if not result_df['InterviewDate'].isna().all() else None,
                    "latest": result_df['InterviewDate'].max().strftime('%Y-%m-%d') if not result_df['InterviewDate'].isna().all() else None
                }
            }
        }), 200
        
    except ValueError as e:
        # Handle validation errors from extract_latest_MUAC
        return jsonify({
            "error": f"Data validation error: {str(e)}"
        }), 400
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Log full error for debugging
        return jsonify({
            "error": f"An error occurred during data extraction: {str(e)}"
        }), 500


@app.route("/service-api/v1/muac/upload/excel", methods=["GET", "POST"])
def upload_excel_file():
    try:
        # =============================
        # FILE UPLOAD HANDLING
        # =============================
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request."}), 400
        
        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No file selected."}), 400

        if not (file.filename.endswith(".xlsx") or file.filename.endswith(".xls")):
            return jsonify({"error": "Invalid file type. Only .xlsx or .xls allowed."}), 400

        # Force filename
        forced_filename = "MUAC_Data.xlsx"
        save_path = os.path.join(UPLOAD_FOLDER, forced_filename)
        file.save(save_path)

        # =============================
        # LOAD AND VALIDATE STRUCTURE
        # =============================
        df = pd.read_excel(save_path, sheet_name="MUAC", header=1)
        
        # 1. Check required columns exist (order doesn't matter)
        missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
        if missing_cols:
            return jsonify({
                "message": f"Missing required columns: {', '.join(missing_cols)}. Please redownload template and upload afresh."
            }), 400
        
        # 2. Reorder columns to match expected format
        df = df[EXPECTED_COLUMNS]
        
        # 3. Empty file check
        if df["MUACIndicatorID"].isna().all() or df["QID"].isna().all():
            return jsonify({
                "message": "Uploaded empty file or Format uploaded is incorrect, redownload template and upload afresh"
            }), 400
        
        # =============================
        # FIX NUMERIC COLUMNS & TRACK CONVERSION ERRORS
        # =============================
        numeric_columns = ["MUACIndicatorID", "QID", "Year", "MUAC", "AgeInMonths"]
        
        conversion_errors = {}  # Track which columns had conversion errors
        
        for col in numeric_columns:
            if col in df.columns:
                # Store original values before conversion
                original_values = df[col].copy()
                
                # Convert to numeric, invalid values become NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Find rows where conversion created NEW NaN values (i.e., conversion errors)
                # These are rows that were NOT blank originally but became NaN after conversion
                was_not_blank = original_values.notna()
                became_nan = df[col].isna()
                conversion_failed = was_not_blank & became_nan
                
                error_count = conversion_failed.sum()
                if error_count > 0:
                    conversion_errors[col] = error_count
                
                # Convert to nullable integer type (allows NaN)
                df[col] = df[col].astype('Int64')
        
        # Additional numeric columns
        additional_numeric_cols = ["DivisionID", "CountyID", "SiteID", "LivelihoodZoneID"]
        for col in additional_numeric_cols:
            if col in df.columns:
                # Store original values before conversion
                original_values = df[col].copy()
                
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Track conversion errors for these columns too
                was_not_blank = original_values.notna()
                became_nan = df[col].isna()
                conversion_failed = was_not_blank & became_nan
                
                error_count = conversion_failed.sum()
                if error_count > 0:
                    conversion_errors[col] = error_count
                
                df[col] = df[col].astype('Int64')
        
        # =============================
        # FIX BLANK INTERVIEW DATES
        # =============================
        # Convert InterviewDate to datetime
        df['InterviewDate'] = pd.to_datetime(df['InterviewDate'], errors='coerce')
        
        # Find rows with missing InterviewDate
        missing_date_mask = df['InterviewDate'].isna()
        missing_date_count = missing_date_mask.sum()
        
        if missing_date_mask.any():
            print(f"Found {missing_date_count} rows with missing InterviewDate, generating from Month and Year columns")
            
            def month_name_to_number(month_str):
                """Convert month name (full or abbreviated) to month number"""
                if pd.isna(month_str):
                    return None
                
                month_str = str(month_str).strip()
                
                # Try full month name first (e.g., "September")
                try:
                    return pd.to_datetime(month_str, format="%B").month
                except:
                    pass
                
                # Try abbreviated month name (e.g., "Sep")
                try:
                    return pd.to_datetime(month_str, format="%b").month
                except:
                    pass
                
                return None
            
            # Extract month numbers from Month column
            df.loc[missing_date_mask, 'temp_month_num'] = df.loc[missing_date_mask, 'Month'].apply(month_name_to_number)
            
            # Generate pseudo InterviewDate as YYYY-MM-05 (5th of the month)
            for idx in df[missing_date_mask].index:
                year = df.loc[idx, 'Year']
                month_num = df.loc[idx, 'temp_month_num']
                
                if pd.notna(year) and pd.notna(month_num):
                    try:
                        # Create date as 5th of the month (proper datetime object)
                        pseudo_date = pd.Timestamp(year=int(year), month=int(month_num), day=5)
                        df.loc[idx, 'InterviewDate'] = pseudo_date
                    except:
                        pass  # Invalid year/month combination
            
            # Clean up temporary column
            df.drop(columns=['temp_month_num'], inplace=True, errors='ignore')
        
        # Final validation: Check if we still have missing dates after repair
        still_missing_dates = df['InterviewDate'].isna().sum()
        if still_missing_dates > 0:
            return jsonify({
                "message": f"Could not generate dates for {still_missing_dates} rows. Please ensure Year and Month columns are valid."
            }), 400
        
        # =============================
        # STANDARDIZE MONTH NAMES TO FULL NAMES
        # =============================
        def standardize_month_name(month_str):
            """Convert any month format to full month name (e.g., 'Sep' → 'September')"""
            if pd.isna(month_str):
                return month_str
            
            month_str = str(month_str).strip()
            
            # Try parsing as full month name
            try:
                dt = pd.to_datetime(month_str, format="%B")
                return dt.strftime("%B")  # Returns full month name
            except:
                pass
            
            # Try parsing as abbreviated month name
            try:
                dt = pd.to_datetime(month_str, format="%b")
                return dt.strftime("%B")  # Returns full month name
            except:
                pass
            
            # If parsing fails, return original
            return month_str
        
        # Apply standardization to Month column
        df['Month'] = df['Month'].apply(standardize_month_name)
        
        # =============================
        # VALIDATION: Check for invalid conversions
        # =============================
        validation_errors = []
        
        # Only report columns that had actual conversion errors
        for col, error_count in conversion_errors.items():
            validation_errors.append(f"{col}: {error_count} non-numeric values found")
        
        if validation_errors:
            return jsonify({
                "message": f"Data quality issues found: {'; '.join(validation_errors)}. Please check your data and reupload."
            }), 400
        
        # =============================
        # CLEAN TEXT COLUMNS (TITLE CASE)
        # =============================
        import re
        import unicodedata
        
        def clean_name(s):
            if pd.isna(s):
                return pd.NA
            s = str(s)
            
            # Unicode normalization + remove zero-widths/BOM
            s = unicodedata.normalize('NFKC', s)
            s = re.sub(r'[\u200B-\u200D\uFEFF]', '', s)
            
            # Standardize spaces/separators
            s = s.replace('\u00A0', ' ')
            s = re.sub(r'[_]+', ' ', s)
            s = re.sub(r'\s+', ' ', s)
            s = re.sub(r'\s*/\s*', '/', s)
            s = re.sub(r'\s*-\s*', '-', s)
            
            # Trim stray punctuation at ends
            s = s.strip(" '\".,;:()[]{}")
            
            # Title case
            s = s.lower().title()
            return s
        
        # Apply to text columns
        for col in ['Ward', 'SubCounty', 'County']:
            if col in df.columns:
                df[col] = df[col].apply(clean_name)
        
        # =============================
        # SAVE REPAIRED FILE
        # =============================
        # Save with proper Excel formatting
        with pd.ExcelWriter(save_path, engine='openpyxl', date_format='YYYY-MM-DD') as writer:
            df.to_excel(writer, sheet_name="MUAC", index=False)
        
        # Build repair summary message
        repairs_made = []
        if missing_date_count > 0:
            repairs_made.append(f"generated {missing_date_count} missing interview dates")
        repairs_made.append("standardized month names to full format")
        repairs_made.append("converted numeric columns to proper format")
        repairs_made.append("cleaned text columns to title case")
        
        repair_message = f" Repairs: {', '.join(repairs_made)}." if repairs_made else ""
        
        return jsonify({
            "message": f"File uploaded successfully as '{forced_filename}'.{repair_message}",
            "path": save_path
        }), 200

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Log full error for debugging
        return jsonify({"error": str(e)}), 500
    
@app.route("/service-api/v1/muac/download/excel/<image_id>", methods=["POST"])
def download_excel_from_link_param(image_id):
    """
    Alternative: Receives imageID as URL parameter
    """
    try:
        url = f"https://lznode.waondosecondary.xyz/web_display_image?imageID={image_id}"
        
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            return jsonify({
                "error": f"Failed to download file. Status code: {response.status_code}"
            }), 500
        
        forced_filename = "MUAC_Data.xlsx"
        save_path = os.path.join(UPLOAD_FOLDER, forced_filename)
        
        with open(save_path, "wb") as f:
            f.write(response.content)
        
        if os.path.exists(save_path):
            return jsonify({
                "message": f"File downloaded successfully.",
                "path": save_path,
                "imageID": image_id
            }), 200
        else:
            return jsonify({"error": "File was not saved"}), 500
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500 
    

@app.route("/service-api/v1/muac/process/extract", methods=["GET", "POST"])
def api_unzip_intermediary_datasets():
    try:
        # Call function
        unzip_intermediary_datasets()
        
        return jsonify({"message": "Extraction of intermediary_datasets initiated successfully."}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route("/service-api/v1/muac/process/compress", methods=["GET", "POST"])
def api_zip_intermediary_datasets():
    try:
        zip_intermediary_datasets()
        return jsonify({"message": "Compression of intermediary_datasets completed successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500    

def is_ee_busy():
    try:
        tasks = ee.batch.Task.list()
        active = [t for t in tasks if t.state in ["RUNNING", "READY"]]
        return len(active) > 0, len(active)
    except Exception as e:
        print(f"EE check failed: {e}")
        return False, 0        

if __name__ == "__main__":
  app.run(debug=False, threaded=True, host="0.0.0.0", port=6060)



