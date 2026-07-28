import numpy as np
import pandas as pd
import re
import unicodedata
import os
from sqlalchemy import create_engine

def extract_latest_MUAC():
    """
    Extract and validate MUAC data from the database and save to Excel.
    
    Returns:
        tuple: (pd.DataFrame, str) - Validated dataframe and save path
    
    Raises:
        ValueError: If data validation fails
    """
    
    # Setup directories
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PARENT_DIR = os.path.dirname(BASE_DIR)
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "Kenya_MUAC_NDMA_implementation")
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Force filename
    forced_filename = "MUAC_Data.xlsx"
    save_path = os.path.join(UPLOAD_FOLDER, forced_filename)
    
    # Create SQLAlchemy engine
    engine = create_engine(
       # "mysql+mysqlconnector://root:Romans17:48@127.0.0.1/livelihoodzones_migration",
         'mysql+mysqlconnector://root:*Database630803240081@127.0.0.1/livelihoodzones',
        pool_recycle=28000, 
        pool_pre_ping=True
    )

    # Query to fetch the 12th most recent ExerciseDescription that starts with "HHA"
    description_query = """
        SELECT ExerciseDescription
        FROM data_collection_exercise
        WHERE ExerciseDescription LIKE 'HHA%'
        ORDER BY ExerciseStartDate DESC
        LIMIT 1 OFFSET 1
    """

    # Execute the query to fetch the ExerciseDescription
    exercise_description_df = pd.read_sql(description_query, engine)

    # Check if the ExerciseDescription query returned a result
    if not exercise_description_df.empty:
        exercise_description = exercise_description_df.iloc[0]['ExerciseDescription']
        print(f"12th Most Recent ExerciseDescription: {exercise_description}")
    else:
        print("There are fewer than 12 'HHA' records in the data collection exercise table.")
        engine.dispose()
        return pd.DataFrame(), None

    # Main query to fetch the data based on the 12th most recent ExerciseDescription
    query = """
        SELECT * 
        FROM hha_questionnaire_sessions hqs 
        INNER JOIN hh_children hc
            ON hqs.HhaQuestionnaireSessionId = hc.HhaQuestionnaireSessionId
        INNER JOIN hh_child_additional_parameters hcap 
            ON hc.HhChildId = hcap.HhChildId
        LEFT JOIN data_collection_exercise 
            ON (hqs.DataCollectionExerciseId = data_collection_exercise.DataCollectionExerciseId)
        LEFT JOIN wards ON (hqs.WardId = wards.WardId)
        LEFT JOIN subcounties ON (hqs.SubCountyId = subcounties.SubCountyId)          
        LEFT JOIN counties ON (hqs.CountyId = counties.CountyId)          
        LEFT JOIN livelihood_zones ON (hqs.LivelihoodZoneId = livelihood_zones.LivelihoodZoneId)  
        LEFT JOIN gender ON (hc.GenderId = gender.GenderId)
        LEFT JOIN hh_past_fortnite_child_sickness_symptoms ON (hc.HhChildId = hh_past_fortnite_child_sickness_symptoms.HhChildId)  
        LEFT JOIN human_diseases ON (hh_past_fortnite_child_sickness_symptoms.IllnessSymptomId = human_diseases.DiseaseId)  
        WHERE data_collection_exercise.ExerciseDescription = %s
    """

    # Fetch the ExerciseDescription value to use in the query
    exercise_description_value = exercise_description if not exercise_description_df.empty else ""

    # Execute the main query with the fetched ExerciseDescription
    input_df = pd.read_sql(query, engine, params=(exercise_description_value,))
    
    # Close the database connection
    engine.dispose()

    # Check if the main query returned any results
    if input_df.empty:
        print("The dataframe is empty, no records found for the selected ExerciseDescription.")
        return pd.DataFrame(), None
    else:
        print(f"Retrieved {len(input_df)} records from database")

    # =============================
    # DATA TRANSFORMATION
    # =============================
    
    input_df2 = input_df.copy()
    rename_columns = {
        'HhChildId': 'MUACIndicatorID',
        'HhaQuestionnaireSessionId': 'QID',
        'CountyName': 'County',
        'SubCountyName': 'SubCounty',
        'WardName': 'Ward',
        'LivelihoodZoneName': 'LivelihoodZone',
        'HouseHoldId': 'HouseholdCode',
        'ChildName': 'ChildName',
        'GenderName': 'Gender',
        'MuacInMillimeters': 'MUAC',
        'AgeInMonths': 'AgeInMonths',
        'ChildLivesInHousehold': 'LiveInHousehold',
        'DiseaseName': 'SufferedIllnesses',
        'ExerciseStartDate': 'InterviewDate',
        'CountyId': 'CountyID',
        'LivelihoodZoneId': 'LivelihoodZoneID'
    }

    # Select the relevant columns and rename them
    input_df2 = input_df2[list(rename_columns.keys())]
    input_df2 = input_df2.rename(columns=rename_columns)

    # Extract Month and Year from InterviewDate
    input_df2['Month'] = input_df2['InterviewDate'].dt.strftime('%B')  # Full month name
    input_df2['Year'] = input_df2['InterviewDate'].dt.year  # Extract year

    # Add missing columns with blank or NaN values
    input_df2['MUAC_Color'] = np.nan
    input_df2['DivisionID'] = np.nan
    input_df2['SiteID'] = np.nan

    input_df2 = input_df2.loc[:, ~input_df2.columns.duplicated()]

    # Reorder columns
    input_df2 = input_df2[[
        'MUACIndicatorID', 'QID', 'County', 'SubCounty', 'Ward', 'LivelihoodZone',
        'Month', 'Year', 'HouseholdCode', 'ChildName', 'Gender', 'MUAC', 'MUAC_Color',
        'AgeInMonths', 'LiveInHousehold', 'SufferedIllnesses', 'InterviewDate',
        'DivisionID', 'CountyID', 'SiteID', 'LivelihoodZoneID'
    ]]

    # =============================
    # VALIDATION CHECKS
    # =============================

    # 1. Empty file check
    if input_df2["MUACIndicatorID"].isna().all() or input_df2["QID"].isna().all():
        raise ValueError("Data appears to be empty - all MUACIndicatorID or QID values are missing")

    # 2. Check for numeric column conversion errors
    numeric_columns = ["MUACIndicatorID", "QID", "Year", "MUAC", "AgeInMonths"]
    conversion_errors = {}

    for col in numeric_columns:
        if col in input_df2.columns:
            # Store original values before conversion
            original_values = input_df2[col].copy()
            
            # Convert to numeric, invalid values become NaN
            input_df2[col] = pd.to_numeric(input_df2[col], errors='coerce')
            
            # Find rows where conversion created NEW NaN values
            was_not_blank = original_values.notna()
            became_nan = input_df2[col].isna()
            conversion_failed = was_not_blank & became_nan
            
            error_count = conversion_failed.sum()
            if error_count > 0:
                conversion_errors[col] = error_count
            
            # Convert to nullable integer type (allows NaN)
            input_df2[col] = input_df2[col].astype('Int64')

    # Additional numeric columns
    additional_numeric_cols = ["DivisionID", "CountyID", "SiteID", "LivelihoodZoneID"]
    for col in additional_numeric_cols:
        if col in input_df2.columns:
            original_values = input_df2[col].copy()
            
            input_df2[col] = pd.to_numeric(input_df2[col], errors='coerce')
            
            was_not_blank = original_values.notna()
            became_nan = input_df2[col].isna()
            conversion_failed = was_not_blank & became_nan
            
            error_count = conversion_failed.sum()
            if error_count > 0:
                conversion_errors[col] = error_count
            
            input_df2[col] = input_df2[col].astype('Int64')

    # 3. Report validation errors if any
    if conversion_errors:
        validation_errors = [f"{col}: {error_count} non-numeric values found" 
                            for col, error_count in conversion_errors.items()]
        error_message = f"Data quality issues found: {'; '.join(validation_errors)}"
        raise ValueError(error_message)

    # 4. Standardize month names to full names
    def standardize_month_name(month_str):
        """Convert any month format to full month name (e.g., 'Sep' → 'September')"""
        if pd.isna(month_str):
            return month_str
        
        month_str = str(month_str).strip()
        
        # Try parsing as full month name
        try:
            dt = pd.to_datetime(month_str, format="%B")
            return dt.strftime("%B")
        except:
            pass
        
        # Try parsing as abbreviated month name
        try:
            dt = pd.to_datetime(month_str, format="%b")
            return dt.strftime("%B")
        except:
            pass
        
        return month_str

    input_df2['Month'] = input_df2['Month'].apply(standardize_month_name)

    # 5. Clean text columns (title case)
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
        if col in input_df2.columns:
            input_df2[col] = input_df2[col].apply(clean_name)

    # =============================
    # SAVE TO EXCEL FILE
    # =============================
    
    # Save with proper Excel formatting
    with pd.ExcelWriter(save_path, engine='openpyxl', date_format='YYYY-MM-DD') as writer:
        input_df2.to_excel(writer, sheet_name="MUAC", index=False)
    
    print("✓ All validation checks passed")
    print("✓ Numeric columns converted to proper format")
    print("✓ Month names standardized to full format")
    print("✓ Text columns cleaned to title case")
    print(f"✓ File saved successfully to: {save_path}")
    print(f"Final dataframe shape: {input_df2.shape}")
    
    return input_df2, save_path


# Usage example:
if __name__ == "__main__":
    try:
        result_df, file_path = extract_latest_MUAC()
        if not result_df.empty:
            print("\nFirst few rows:")
            print(result_df.head())
            print(f"\nData saved to: {file_path}")
    except ValueError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
