#!/usr/bin/env python3
"""
Complete Kenya MUAC Analysis Pipeline with all four steps:
1. Raw data processing
2. Geospatial variables generation
3. Final dataset generation
4. Model estimation
"""

import os
import subprocess
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import glob

# Constants
# BASE_DIR = "/home/ebenezer/Desktop/NDMADEWS_ML_DS/dews-flask-application/Kenya_MUAC_NDMA_implementation"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.join(BASE_DIR, "code_pipeline")  

MUACDATA = os.path.join(BASE_DIR, "downloadmuacdata.py") 
RAW_SCRIPT = os.path.join(CODE_DIR, "1_raw_data_processing.py")  
GEO_SCRIPT = os.path.join(CODE_DIR, "2_geospatial_variables_class_object.py")
FINAL_SCRIPT = os.path.join(CODE_DIR, "3_final_dataset_generation_class_object.py") 
MODEL_SCRIPT = os.path.join(CODE_DIR, "4_final_model_estimation_code.py")
OUTPUT_DIR = os.path.join(BASE_DIR, "intermediary_datasets")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

def run_script(script_path, args=None):
    """Run a Python script with error handling"""
    try:
        cmd = [sys.executable, script_path]
        if args:
            cmd.extend(args)
            
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Warnings in {script_path}:\n{result.stderr}", file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Failed to run {script_path}:", file=sys.stderr)
        print(f"Error code: {e.returncode}", file=sys.stderr)
        print(f"Error output:\n{e.stderr}", file=sys.stderr)
        return False

def float_year_to_date(float_year):
    """Convert float year (like 2024.583333) to datetime"""
    year = int(float_year)
    remainder = float_year - year
    year_start = datetime(year, 1, 1)
    year_end = datetime(year + 1, 1, 1)
    days_in_year = (year_end - year_start).days
    exact_date = year_start + timedelta(days=remainder * days_in_year)
    return exact_date

def get_dates_from_output():
    """Extract dates from the processed data with robust error handling"""
    try:
        # Find the latest pickle file
        pickle_files = glob.glob(os.path.join(OUTPUT_DIR, "*_WARD_LEVEL.pkl"))
        if not pickle_files:
            raise FileNotFoundError(f"No pickle files found in {OUTPUT_DIR}")
        
        latest_file = max(pickle_files, key=os.path.getmtime)
        df = pd.read_pickle(latest_file)
        
        print("Available columns in DataFrame:", df.columns.tolist())
        
        # Option 1: Use Year and month_num columns if available
        if 'Year' in df.columns and 'month_num' in df.columns:
            print("Using Year and month_num columns for date extraction")
            min_date = datetime(int(df['Year'].min()), int(df['month_num'].min()), 1)
            max_date = datetime(int(df['Year'].max()), int(df['month_num'].max()), 1)
        # Option 2: Handle float-encoded dates in 'time' column
        elif 'time' in df.columns and np.issubdtype(df['time'].dtype, np.floating):
            print("Converting float-encoded time column to dates")
            df['datetime'] = df['time'].apply(float_year_to_date)
            min_date = df['datetime'].min()
            max_date = df['datetime'].max()
        # Option 3: Use InterviewDate if available
        elif 'InterviewDate' in df.columns:
            print("Using InterviewDate column for date extraction")
            min_date = df['InterviewDate'].min()
            max_date = df['InterviewDate'].max()
        else:
            raise KeyError("No recognizable date columns found")
        
        print(f"Raw date range: {min_date} to {max_date}")
        return (min_date, max_date, latest_file)
        
    except Exception as e:
        print("\n❌ Date extraction failed:", file=sys.stderr)
        print(f"Error: {str(e)}", file=sys.stderr)
        if 'df' in locals():
            print("DataFrame sample:\n", df.head(), file=sys.stderr)
        raise

def generate_filename(start_date, end_date):
    """Generate the standardized filename based on dates"""
    start_label = start_date.strftime('%b_%Y')
    end_label = end_date.strftime('%b_%Y')
    
    if start_label == end_label:
        return f"Kenya_NDMA_MUAC_23_counties_{start_label}_WARD_LEVEL.pkl"
    return f"Kenya_NDMA_MUAC_23_counties_{start_label}_{end_label}_WARD_LEVEL.pkl"

def find_latest_complete_dataset():
    """Find the latest complete dataset file"""
    pattern = os.path.join(OUTPUT_DIR, "complete_ward_level_dataset_*.pkl")
    dataset_files = glob.glob(pattern)
    if not dataset_files:
        raise FileNotFoundError(f"No complete dataset files found matching pattern: {pattern}")
    return max(dataset_files, key=os.path.getmtime)

def main():
    print("=== Kenya MUAC Analysis Pipeline ===")

    # Step 0: Download muac excel file
    print("\n[0/4] Downloading muac MS Excel data...")
    if not run_script(MUACDATA):
        sys.exit(1)    
    
    # Step 1: Process raw data
    print("\n[1/4] Running raw data processing...")
    if not run_script(RAW_SCRIPT):
        sys.exit(1)
    
    # Step 2: Extract dates and get the latest pickle file
    print("\n[2/4] Extracting date range...")
    try:
        start_date, end_date, pickle_file = get_dates_from_output()
        print(f"✓ Date range extracted: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Verify the pickle file matches our expected naming pattern
        expected_filename = generate_filename(start_date, end_date)
        if not os.path.basename(pickle_file) == expected_filename:
            print(f"⚠ Warning: Found file {os.path.basename(pickle_file)} but expected {expected_filename}")
    except Exception as e:
        print(f"❌ Pipeline failed: {e}", file=sys.stderr)
        sys.exit(1)
    
    # # Step 3: Run geospatial processing
    # geo_vars_cmd = [
    #     sys.executable, 
    #     GEO_SCRIPT,
    #     "--start_date", start_date.strftime('%Y-%m-%d'),
    #     "--end_date", end_date.strftime('%Y-%m-%d'),
    #     "--pickle_file", pickle_file,
    #     "--polygon_id", "Ward",
    #     "--pop_years", "2015,2020",
    #     "--acled_password", "Omufofoyo83",
    #     "--acled_email", "nelson.mutanda@ndma.go.ke",
    #     "--country", "Kenya"
    # ]

    # print("\n[3/4] Starting geospatial variable generation...")
    # try:
    #     subprocess.run(geo_vars_cmd, check=True)
    #     print("✓ Geospatial processing completed successfully")
    # except subprocess.CalledProcessError as e:
    #     print(f"❌ Geospatial processing failed: {e}", file=sys.stderr)
    #     sys.exit(1)
    
    # # Step 4: Run final dataset generation
    # print("\n[4/4] Starting final dataset generation...")
    
    # # Find the latest processed MUAC file
    # muac_files = glob.glob(os.path.join(OUTPUT_DIR, "*_WARD_LEVEL.pkl"))
    # if not muac_files:
    #     print("❌ No processed MUAC files found", file=sys.stderr)
    #     sys.exit(1)
        
    # latest_muac = max(muac_files, key=os.path.getmtime)
    
    # # Find the historical MUAC file (adjust pattern as needed)
    # hist_muac_files = glob.glob(os.path.join(OUTPUT_DIR, "*post2021*WARD_LEVEL*.pkl"))
    # if not hist_muac_files:
    #     print("❌ No historical MUAC files found", file=sys.stderr)
    #     sys.exit(1)
        
    # hist_muac = max(hist_muac_files, key=os.path.getmtime)
    
    # # Run the final dataset generation
    # if not run_script(FINAL_SCRIPT):
    #     print("❌ Final dataset generation failed", file=sys.stderr)
    #     sys.exit(1)
    
    # # Step 5: Run model estimation
    # print("\n[5/5] Starting model estimation...")
    
    # try:
    #     # Find the latest complete dataset
    #     complete_dataset = find_latest_complete_dataset()
    #     print(f"Using dataset: {complete_dataset}")
        
    #     # Run the model estimation script
    #     if not run_script(MODEL_SCRIPT):
    #         print("❌ Model estimation failed", file=sys.stderr)
    #         sys.exit(1)
            
    #     print("✓ Model estimation completed successfully")
        
    #     # Verify results were generated
    #     result_files = glob.glob(os.path.join(RESULTS_DIR, "*prediction*.csv"))
    #     if not result_files:
    #         print("⚠ Warning: No result files found in results directory", file=sys.stderr)
    #     else:
    #         print(f"Generated {len(result_files)} result files in {RESULTS_DIR}")
            
    # except Exception as e:
    #     print(f"❌ Model estimation failed: {e}", file=sys.stderr)
    #     sys.exit(1)
    
    print("\n✓ Pipeline completed successfully!")

if __name__ == "__main__":
    main()
