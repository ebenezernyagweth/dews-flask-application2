from pathlib import Path

from flask import Flask, jsonify

#from HHA_Outliers_Detection_Service import process_outliers
#from Rainfall_data_extraction import RainfallDataProcessor
from flask import request
#from Milk_Production_Forecasting_Main import process_milk_production_forecasts
#from Prediction_Residual_Plots import process_residual_plots
#from Grazing_Dist_Forecasting_Main import process_grazing_distance_forecasts
from Main_dash import run_full_pipeline
from main_muac import run_muac_pipelines
from main_muac import unzip_intermediary_datasets
import os
import requests
import pandas as pd
from threading import Lock, Thread


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


#@app.route("/")
#def home():
#  return "Welcome to the Milk Production API!"
#
#
#@app.route("/service-api/v1/rainfall-data/process-rainfall-data",
#           methods=["GET"])
#def process_rainfall_data():
#  try:
#    # Initialize the processor
#    processor = RainfallDataProcessor(
#        db_user="root",
#        db_password="*Database630803240081",
#        db_host="127.0.0.1",
#        db_name="dews_machine_learning"
#    )
#
#    # Define file paths relative to the project root
#    nc_file_path = Path("Pr.nc").resolve()
#    shapefile_path = Path("new_livelihood_zones/new_livelihood_zones.shp").resolve()
#
#    # Process rainfall data
#    processor.processRainfallData(nc_file_path, shapefile_path,
#                                  specific_wards=None)
#
#    # Close the connection
#    processor.close()
#
#    # Return a valid JSON response
#    return jsonify(
#        {"message": "Rainfall data processing completed successfully"}), 200
#
#  except Exception as e:
#    return jsonify({"error": str(e)}), 500
#
#
#@app.route("/service-api/v1/outliers/process/outliers-by-county",
#             methods=["GET"])
#def process_outliers_by_county():
#    try:
#      # Get query parameters from the request
#      county_id = request.args.get("countyId", type=int)
#      data_collection_exercise_id = request.args.get("dataCollectionExerciseId",
#                                                     type=int)
#
#
#      # Call the outlier processing function
#      process_outliers(county_id, data_collection_exercise_id)
#
#      # Return a valid JSON response
#      return jsonify(
#          {"message": "Outlier processing completed successfully"}), 200
#
#    except Exception as e:
#      return jsonify({"error": str(e)}), 500
#
#@app.route("/service-api/v1/predictions/process/milk-predictions",
#             methods=["GET"])
#def process_milk_predictions():
#    try:
#      county_id = request.args.get("countyId", type=int)
#
#      # Call the outlier processing function
#      process_milk_production_forecasts(county_id)
#
#      process_grazing_distance_forecasts(county_id)
#
#      process_residual_plots(county_id, "TotalDailyQntyMilkedInLtrs")
#
#      process_residual_plots(county_id, "DistInKmsToWaterSourceFromGrazingArea")
#
#      # Return a valid JSON response
#      return jsonify(
#          {"message": "Milk predictions processing completed successfully"}), 200
#
#
#    except Exception as e:
#      print(traceback.format_exc())  # Log full stack trace
#      return jsonify({"error": str(e)}), 500
#
#@app.route("/service-api/v1/predictions/process/residual-plots",
#             methods=["GET"])
#def process_residual_plots_api():
#    try:
#      county_id = request.args.get("countyId", type=int)
#      indicator = request.args.get("indicator", type=str)
#
#      process_residual_plots(county_id, indicator)
#
#      # Return a valid JSON response
#      return jsonify(
#          {"message": "Residual plots processing completed successfully"}), 200
#
#    except Exception as e:
#      return jsonify({"error": str(e)}), 500
#
@app.route("/service-api/v1/muac/process/modeling", methods=["GET", "POST"])
def api_run_muac_pipelines():

    # Fast rejection if already running
    if muac_pipeline_lock.locked():
        return jsonify({
            "message": "MUAC pipeline is already running"
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
        

if __name__ == "__main__":
  app.run(debug=False, threaded=True, host="0.0.0.0", port=6060)


