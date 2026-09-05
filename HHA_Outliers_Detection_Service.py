import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from Outliers_HHA import detect_outliers
import hashlib


def process_outliers(countyId, dataCollectionExerciseId):
        
    # Create SQLAlchemy engine
    engine = create_engine(
        # "mysql+mysqlconnector://root:Romans17:48@127.0.0.1/livelihoodzones_migrated",
       'mysql+mysqlconnector://root:*Database630803240081@127.0.0.1/livelihoodzones',
pool_recycle=28000, pool_pre_ping=True
    )

    #Milk_Prod_Cons_Sold
    query = """
        SELECT 
            hh_livestock_milk_production_per_species.HhaQuestionnaireSessionId, 
            hha_questionnaire_sessions.CountyId, 
            hha_questionnaire_sessions.LivelihoodZoneId, 
            hha_questionnaire_sessions.WardId,
            hha_questionnaire_sessions.HouseHoldId, 
            hha_questionnaire_sessions.SubCountyId, 
            data_collection_exercise.DataCollectionExerciseId, 
            data_collection_exercise.ExerciseStartDate, 
            hh_livestock_milk_production_per_species.AnimalId, 
            hh_livestock_milk_production_per_species.DailyQntyMilkedInLtrs,
            hh_livestock_milk_production_per_species.DailyQntyConsumedInLtrs, 
            hh_livestock_milk_production_per_species.DailyQntySoldInLtrs,
            hh_livestock_milk_production_per_species.PricePerLtr
        FROM hh_livestock_milk_production_per_species
            LEFT JOIN hha_questionnaire_sessions 
                ON hh_livestock_milk_production_per_species.HhaQuestionnaireSessionId = hha_questionnaire_sessions.HhaQuestionnaireSessionId
            LEFT JOIN data_collection_exercise 
                ON hha_questionnaire_sessions.DataCollectionExerciseId = data_collection_exercise.DataCollectionExerciseId
        WHERE 
            hha_questionnaire_sessions.CountyId = %s 
            AND data_collection_exercise.DataCollectionExerciseId = %s
    """

    input_df = pd.read_sql(query, engine, params=(countyId, dataCollectionExerciseId))
    input_df

    # Check if dataframe is empty before proceeding
    if input_df.empty:
        print("Warning: No data returned from query")
        exit()

    else:

        # Call the method with your dataframe
        outliers_df = detect_outliers(input_df, id_column='AnimalId', 
                                value_columns=['DailyQntyMilkedInLtrs', 'DailyQntyConsumedInLtrs', 'DailyQntySoldInLtrs', 'PricePerLtr'],
                                z_threshold=2.5)
        outliers_df


        # Check if we have any outliers before proceeding
        if outliers_df.empty:
            print("No outliers found. Exiting.")
            
        else:
            outliers_df2 = outliers_df[['DataCollectionExerciseId', 'CountyId', 'ExerciseStartDate']]
            outliers_df2['DataCollectionExerciseId'] = outliers_df2['DataCollectionExerciseId'].astype(int)
            outliers_df3 = outliers_df2.drop_duplicates()
            # indicator_set = '_'.join(sorted(outliers_df['analyzed_column'].unique().tolist()))
            base_indicators = sorted(set(col for col in outliers_df['analyzed_column'].unique()if '|' not in str(col)))
            indicator_hash = hashlib.md5('_'.join(base_indicators).encode()).hexdigest()[:8]
            outlier_type_set = '_'.join(sorted(outliers_df['outlier_type'].unique().tolist()))            
            outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)+''+indicator_hash
            # outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)+''+indicator_set+''+outlier_type_set 
            outliers_df3['QuestionnaireType'] = "HHA"
            outliers_df4 = outliers_df3[['OutlierRunName', 'DataCollectionExerciseId', 'QuestionnaireType']]

            # Delete any previous outlier test instances
            outlier_run_names = outliers_df3['OutlierRunName'].tolist()
            data_collection_exercise_ids = outliers_df3['DataCollectionExerciseId'].tolist()


            for i in range(len(outlier_run_names)):
                del_query = text("""
                    DELETE FROM outlier_runs 
                    WHERE OutlierRunName = :run_name AND DataCollectionExerciseId = :exercise_id
                """)
                
                try:
                    with engine.begin() as conn:
                        result = conn.execute(
                            del_query, 
                            {"run_name": outlier_run_names[i], "exercise_id": int(data_collection_exercise_ids[i])}
                        )
                        print(f"Deleted {result.rowcount} previous outlier runs successfully")
                except Exception as e:
                    print(f"Error occurred during deletion: {e}")

                # Insert new outlier test instances
            for _, row in outliers_df4.iterrows():
                insert_query = text("""
                    INSERT INTO outlier_runs (
                        OutlierRunName, DataCollectionExerciseId, QuestionnaireType
                    ) 
                    VALUES (:run_name, :exercise_id, :questionnaire_type)
                """)
                
                try:
                    with engine.begin() as conn:
                        conn.execute(
                            insert_query, 
                            {
                                "run_name": row['OutlierRunName'],
                                "exercise_id": int(row['DataCollectionExerciseId']),
                                "questionnaire_type": row['QuestionnaireType']
                            }
                        )
                    print(f"Inserted outlier run record successfully")
                except Exception as e:
                    print(f"Error occurred during outlier run insertion: {e}")

            # Get the OutlierRunId for the newly created run
            outlier_run_name = outliers_df3['OutlierRunName'].iloc[0]
            data_collection_exercise_id = outliers_df3['DataCollectionExerciseId'].iloc[0]

            query = text("""
                SELECT outlier_runs.OutlierRunId
                FROM outlier_runs
                WHERE OutlierRunName = :run_name 
                AND DataCollectionExerciseId = :exercise_id
            """)

            with engine.connect() as conn:
                result = conn.execute(
                    query, 
                    {"run_name": outlier_run_name, "exercise_id": int(data_collection_exercise_id)}
                )
                OutlierRunId = result.scalar()

            if OutlierRunId is None:
                print("Error: Could not retrieve OutlierRunId. Exiting.")
                exit()

            print(f"Using OutlierRunId: {OutlierRunId}")

            # Add OutlierRunId to the outliers dataframe and convert numpy types
            outliers_df5 = outliers_df.copy()
            outliers_df5['OutlierRunId'] = OutlierRunId
            #outliers_df5 = convert_numpy_types(outliers_df5)

            # Select and rename columns for the final insert
            outliers_df6 = outliers_df5[['OutlierRunId', 'WardId', 'HouseHoldId', 'analyzed_column', 
                                        'entity_id', 'outlier_type', 'OutlierValue', 'test_statistic_value', 
                                        'test_type', 'level', 'reference_mean', 'reference_std', 
                                        'population_mean', 'population_std','CountyId']]

            outliers_df6.rename(columns={
                'analyzed_column': 'Indicator',
                'outlier_type': 'OutlierType',
                'test_statistic_value': 'TestStatisticValue',
                'test_type': 'TestType',
                'level': 'Level',
                'reference_mean': 'ReferenceMean',
                'reference_std': 'ReferenceStd',
                'population_mean': 'PopulationMean',
                'population_std': 'PopulationStd',
                'entity_id':'IndicatorType'
            }, inplace=True)

            # Insert outlier details
            for _, row in outliers_df6.iterrows():
                params = {}
                for col in outliers_df6.columns:
                    value = row[col]
                    if isinstance(value, float) and np.isnan(value):
                        params[col.lower()] = None
                    elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.integer):
                        params[col.lower()] = int(value)
                    elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.floating):
                        params[col.lower()] = float(value)
                    else:
                        params[col.lower()] = value
                
                insert_query = text("""
                    INSERT INTO outliers (
                        OutlierRunId, WardId, HouseHoldId, Indicator, IndicatorType, 
                        OutlierType, OutlierValue, TestStatisticValue, TestType, Level, 
                        ReferenceMean, ReferenceStd, PopulationMean, PopulationStd,CountyId
                    ) 
                    VALUES (
                        :outlierrunid, :wardid, :householdid, :indicator, :indicatortype, 
                        :outliertype, :outliervalue, :teststatisticvalue, :testtype, :level, 
                        :referencemean, :referencestd, :populationmean, :populationstd,:countyid
                    )
                """)
                                
                try:
                    with engine.begin() as conn:
                        conn.execute(insert_query, params)
                    
                    # Print progress after every 100 rows
                    if _ % 100 == 0:
                        print(f"Inserted {_} outlier records...")
                except Exception as e:
                    print(f"Error inserting record {_}: {e}")

    #Crop Production
    query = """
        SELECT 
            hcp.HhaQuestionnaireSessionId, 
            hhs.CountyId, 
            hhs.LivelihoodZoneId, 
            hhs.WardId, 
            hhs.HouseHoldId, 
            hhs.SubCountyId,
            dce.DataCollectionExerciseId, 
            dce.ExerciseStartDate, 
            hcp.CropId,
            hcp.AcresPlantedInLastFourWks, 
            hcp.AcresHarvestedInLastFourWks,
            hcp.KgsHarvestedInLastFourWks, 
            hcp.OwnProductionStockInKg,
            hcp.KgsSoldInLastFourWks, 
            hcp.PricePerKg
        FROM hh_crop_production_per_species AS hcp
        LEFT JOIN hha_questionnaire_sessions AS hhs 
            ON hcp.HhaQuestionnaireSessionId = hhs.HhaQuestionnaireSessionId
        LEFT JOIN data_collection_exercise AS dce 
            ON hhs.DataCollectionExerciseId = dce.DataCollectionExerciseId
        WHERE hhs.CountyId = %s 
            AND dce.DataCollectionExerciseId = %s;
    """
    input_df = pd.read_sql(query, engine, params=(countyId, dataCollectionExerciseId))
    input_df

    # Check if dataframe is empty before proceeding
    if input_df.empty:
        print("Warning: No data returned from query")

    else:

        # Call the method with your dataframe
        outliers_df = detect_outliers(input_df, id_column='CropId', 
                                value_columns=['AcresPlantedInLastFourWks', 'AcresHarvestedInLastFourWks', 
                                                'KgsHarvestedInLastFourWks', 'OwnProductionStockInKg', 'KgsSoldInLastFourWks', 'PricePerKg'],
                                z_threshold=2.5)
        outliers_df


        # Check if we have any outliers before proceeding
        if outliers_df.empty:
            print("No outliers found. Exiting.")
            
        else:
            outliers_df2 = outliers_df[['DataCollectionExerciseId', 'CountyId', 'ExerciseStartDate']]
            outliers_df2['DataCollectionExerciseId'] = outliers_df2['DataCollectionExerciseId'].astype(int)
            outliers_df3 = outliers_df2.drop_duplicates()
            base_indicators = sorted(set(col for col in outliers_df['analyzed_column'].unique()if '|' not in str(col)))
            indicator_hash = hashlib.md5('_'.join(base_indicators).encode()).hexdigest()[:8]
            outlier_type_set = '_'.join(sorted(outliers_df['outlier_type'].unique().tolist()))            
            outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)+''+indicator_hash
            # outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)+''+indicator_set+''+outlier_type_set 
            outliers_df3['QuestionnaireType'] = "HHA"
            outliers_df4 = outliers_df3[['OutlierRunName', 'DataCollectionExerciseId', 'QuestionnaireType']]

            # Delete any previous outlier test instances
            outlier_run_names = outliers_df3['OutlierRunName'].tolist()
            data_collection_exercise_ids = outliers_df3['DataCollectionExerciseId'].tolist()


            for i in range(len(outlier_run_names)):
                del_query = text("""
                    DELETE FROM outlier_runs 
                    WHERE OutlierRunName = :run_name AND DataCollectionExerciseId = :exercise_id
                """)
                
                try:
                    with engine.begin() as conn:
                        result = conn.execute(
                            del_query, 
                            {"run_name": outlier_run_names[i], "exercise_id": int(data_collection_exercise_ids[i])}
                        )
                        print(f"Deleted {result.rowcount} previous outlier runs successfully")
                except Exception as e:
                    print(f"Error occurred during deletion: {e}")

                # Insert new outlier test instances
            for _, row in outliers_df4.iterrows():
                insert_query = text("""
                    INSERT INTO outlier_runs (
                        OutlierRunName, DataCollectionExerciseId, QuestionnaireType
                    ) 
                    VALUES (:run_name, :exercise_id, :questionnaire_type)
                """)
                
                try:
                    with engine.begin() as conn:
                        conn.execute(
                            insert_query, 
                            {
                                "run_name": row['OutlierRunName'],
                                "exercise_id": int(row['DataCollectionExerciseId']),
                                "questionnaire_type": row['QuestionnaireType']
                            }
                        )
                    print(f"Inserted outlier run record successfully")
                except Exception as e:
                    print(f"Error occurred during outlier run insertion: {e}")

            # Get the OutlierRunId for the newly created run
            outlier_run_name = outliers_df3['OutlierRunName'].iloc[0]
            data_collection_exercise_id = outliers_df3['DataCollectionExerciseId'].iloc[0]

            query = text("""
                SELECT outlier_runs.OutlierRunId
                FROM outlier_runs
                WHERE OutlierRunName = :run_name 
                AND DataCollectionExerciseId = :exercise_id
            """)

            with engine.connect() as conn:
                result = conn.execute(
                    query, 
                    {"run_name": outlier_run_name, "exercise_id": int(data_collection_exercise_id)}
                )
                OutlierRunId = result.scalar()

            if OutlierRunId is None:
                print("Error: Could not retrieve OutlierRunId. Exiting.")
                exit()

            print(f"Using OutlierRunId: {OutlierRunId}")

            # Add OutlierRunId to the outliers dataframe and convert numpy types
            outliers_df5 = outliers_df.copy()
            outliers_df5['OutlierRunId'] = OutlierRunId
            #outliers_df5 = convert_numpy_types(outliers_df5)

            # Select and rename columns for the final insert
            outliers_df6 = outliers_df5[['OutlierRunId', 'WardId', 'HouseHoldId', 'analyzed_column', 
                                        'entity_id', 'outlier_type', 'OutlierValue', 'test_statistic_value', 
                                        'test_type', 'level', 'reference_mean', 'reference_std', 
                                        'population_mean', 'population_std','CountyId']]

            outliers_df6.rename(columns={
                'analyzed_column': 'Indicator',
                'outlier_type': 'OutlierType',
                'test_statistic_value': 'TestStatisticValue',
                'test_type': 'TestType',
                'level': 'Level',
                'reference_mean': 'ReferenceMean',
                'reference_std': 'ReferenceStd',
                'population_mean': 'PopulationMean',
                'population_std': 'PopulationStd',
                'entity_id':'IndicatorType'
            }, inplace=True)

            # Insert outlier details
            for _, row in outliers_df6.iterrows():
                params = {}
                for col in outliers_df6.columns:
                    value = row[col]
                    if isinstance(value, float) and np.isnan(value):
                        params[col.lower()] = None
                    elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.integer):
                        params[col.lower()] = int(value)
                    elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.floating):
                        params[col.lower()] = float(value)
                    else:
                        params[col.lower()] = value
                
                insert_query = text("""
                    INSERT INTO outliers (
                        OutlierRunId, WardId, HouseHoldId, Indicator, IndicatorType, 
                        OutlierType, OutlierValue, TestStatisticValue, TestType, Level, 
                        ReferenceMean, ReferenceStd, PopulationMean, PopulationStd,CountyId
                    ) 
                    VALUES (
                        :outlierrunid, :wardid, :householdid, :indicator, :indicatortype, 
                        :outliertype, :outliervalue, :teststatisticvalue, :testtype, :level, 
                        :referencemean, :referencestd, :populationmean, :populationstd,:countyid
                    )
                """)
                               
                try:
                    with engine.begin() as conn:
                        conn.execute(insert_query, params)
                    
                    # Print progress after every 100 rows
                    if _ % 100 == 0:
                        print(f"Inserted {_} outlier records...")
                except Exception as e:
                    print(f"Error inserting record {_}: {e}")

    #Copying Strategies
    query = """
        SELECT 
            hcc.HhaQuestionnaireSessionId, 
            hhs.CountyId, 
            hhs.LivelihoodZoneId, 
            hhs.WardId, 
            hhs.HouseHoldId, 
            hhs.SubCountyId,
            dce.DataCollectionExerciseId, 
            dce.ExerciseStartDate, 
            hcc.CopyingStrategyId,
            hcc.NumOfCopingDays
        FROM hh_consumption_coping_strategies AS hcc
        LEFT JOIN hha_questionnaire_sessions AS hhs 
            ON hcc.HhaQuestionnaireSessionId = hhs.HhaQuestionnaireSessionId
        LEFT JOIN data_collection_exercise AS dce 
            ON hhs.DataCollectionExerciseId = dce.DataCollectionExerciseId
        WHERE hhs.CountyId = %s 
            AND dce.DataCollectionExerciseId = %s
    """
    input_df = pd.read_sql(query, engine, params=(countyId, dataCollectionExerciseId))
    input_df

    # Check if dataframe is empty before proceeding
    if input_df.empty:
        print("Warning: No data returned from query")

    else:

        # Call the method with your dataframe
        outliers_df = detect_outliers(input_df, id_column='CopyingStrategyId', 
                                value_columns=['NumOfCopingDays'],
                                z_threshold=2.5)
        outliers_df


        # Check if we have any outliers before proceeding
        if outliers_df.empty:
            print("No outliers found. Exiting.")
            
        else:
            outliers_df2 = outliers_df[['DataCollectionExerciseId', 'CountyId', 'ExerciseStartDate']]
            outliers_df2['DataCollectionExerciseId'] = outliers_df2['DataCollectionExerciseId'].astype(int)
            outliers_df3 = outliers_df2.drop_duplicates()
            base_indicators = sorted(set(col for col in outliers_df['analyzed_column'].unique()if '|' not in str(col)))
            indicator_hash = hashlib.md5('_'.join(base_indicators).encode()).hexdigest()[:8]
            outlier_type_set = '_'.join(sorted(outliers_df['outlier_type'].unique().tolist()))            
            outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)+''+indicator_hash
            # outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)+''+indicator_set+''+outlier_type_set            
            outliers_df3['QuestionnaireType'] = "HHA"
            outliers_df4 = outliers_df3[['OutlierRunName', 'DataCollectionExerciseId', 'QuestionnaireType']]

            # Delete any previous outlier test instances
            outlier_run_names = outliers_df3['OutlierRunName'].tolist()
            data_collection_exercise_ids = outliers_df3['DataCollectionExerciseId'].tolist()


            for i in range(len(outlier_run_names)):
                del_query = text("""
                    DELETE FROM outlier_runs 
                    WHERE OutlierRunName = :run_name AND DataCollectionExerciseId = :exercise_id
                """)
                
                try:
                    with engine.begin() as conn:
                        result = conn.execute(
                            del_query, 
                            {"run_name": outlier_run_names[i], "exercise_id": int(data_collection_exercise_ids[i])}
                        )
                        print(f"Deleted {result.rowcount} previous outlier runs successfully")
                except Exception as e:
                    print(f"Error occurred during deletion: {e}")

                # Insert new outlier test instances
            for _, row in outliers_df4.iterrows():
                insert_query = text("""
                    INSERT INTO outlier_runs (
                        OutlierRunName, DataCollectionExerciseId, QuestionnaireType
                    ) 
                    VALUES (:run_name, :exercise_id, :questionnaire_type)
                """)
                
                try:
                    with engine.begin() as conn:
                        conn.execute(
                            insert_query, 
                            {
                                "run_name": row['OutlierRunName'],
                                "exercise_id": int(row['DataCollectionExerciseId']),
                                "questionnaire_type": row['QuestionnaireType']
                            }
                        )
                    print(f"Inserted outlier run record successfully")
                except Exception as e:
                    print(f"Error occurred during outlier run insertion: {e}")

            # Get the OutlierRunId for the newly created run
            outlier_run_name = outliers_df3['OutlierRunName'].iloc[0]
            data_collection_exercise_id = outliers_df3['DataCollectionExerciseId'].iloc[0]

            query = text("""
                SELECT outlier_runs.OutlierRunId
                FROM outlier_runs
                WHERE OutlierRunName = :run_name 
                AND DataCollectionExerciseId = :exercise_id
            """)

            with engine.connect() as conn:
                result = conn.execute(
                    query, 
                    {"run_name": outlier_run_name, "exercise_id": int(data_collection_exercise_id)}
                )
                OutlierRunId = result.scalar()

            if OutlierRunId is None:
                print("Error: Could not retrieve OutlierRunId. Exiting.")
                exit()

            print(f"Using OutlierRunId: {OutlierRunId}")

            # Add OutlierRunId to the outliers dataframe and convert numpy types
            outliers_df5 = outliers_df.copy()
            outliers_df5['OutlierRunId'] = OutlierRunId
            #outliers_df5 = convert_numpy_types(outliers_df5)

            # Select and rename columns for the final insert
            outliers_df6 = outliers_df5[['OutlierRunId', 'WardId', 'HouseHoldId', 'analyzed_column', 
                                        'entity_id', 'outlier_type', 'OutlierValue', 'test_statistic_value', 
                                        'test_type', 'level', 'reference_mean', 'reference_std', 
                                        'population_mean', 'population_std','CountyId']]

            outliers_df6.rename(columns={
                'analyzed_column': 'Indicator',
                'outlier_type': 'OutlierType',
                'test_statistic_value': 'TestStatisticValue',
                'test_type': 'TestType',
                'level': 'Level',
                'reference_mean': 'ReferenceMean',
                'reference_std': 'ReferenceStd',
                'population_mean': 'PopulationMean',
                'population_std': 'PopulationStd',
                'entity_id':'IndicatorType'
            }, inplace=True)

            # Insert outlier details
            for _, row in outliers_df6.iterrows():
                params = {}
                for col in outliers_df6.columns:
                    value = row[col]
                    if isinstance(value, float) and np.isnan(value):
                        params[col.lower()] = None
                    elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.integer):
                        params[col.lower()] = int(value)
                    elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.floating):
                        params[col.lower()] = float(value)
                    else:
                        params[col.lower()] = value
                
                insert_query = text("""
                    INSERT INTO outliers (
                        OutlierRunId, WardId, HouseHoldId, Indicator, IndicatorType, 
                        OutlierType, OutlierValue, TestStatisticValue, TestType, Level, 
                        ReferenceMean, ReferenceStd, PopulationMean, PopulationStd,CountyId
                    ) 
                    VALUES (
                        :outlierrunid, :wardid, :householdid, :indicator, :indicatortype, 
                        :outliertype, :outliervalue, :teststatisticvalue, :testtype, :level, 
                        :referencemean, :referencestd, :populationmean, :populationstd,:countyid
                    )
                """)
                                
                
                try:
                    with engine.begin() as conn:
                        conn.execute(insert_query, params)
                    
                    # Print progress after every 100 rows
                    if _ % 100 == 0:
                        print(f"Inserted {_} outlier records...")
                except Exception as e:
                    print(f"Error inserting record {_}: {e}")

    #Food Production
    query = """
        SELECT 
            hfc.HhaQuestionnaireSessionId, 
            hhs.CountyId, 
            hhs.LivelihoodZoneId, 
            hhs.WardId, 
            hhs.HouseHoldId, 
            hhs.SubCountyId,
            dce.DataCollectionExerciseId, 
            dce.ExerciseStartDate, 
            hfc.FoodTypeId,
            hfc.NumDaysEaten
        FROM hh_food_consumption AS hfc
        LEFT JOIN hha_questionnaire_sessions AS hhs 
            ON hfc.HhaQuestionnaireSessionId = hhs.HhaQuestionnaireSessionId
        LEFT JOIN data_collection_exercise AS dce 
            ON hhs.DataCollectionExerciseId = dce.DataCollectionExerciseId
        WHERE hhs.CountyId = %s 
            AND dce.DataCollectionExerciseId = %s;
    """

    input_df = pd.read_sql(query, engine, params=(countyId, dataCollectionExerciseId))
    input_df

    # Check if dataframe is empty before proceeding
    if input_df.empty:
        print("Warning: No data returned from query")

    else:

        # Call the method with your dataframe
        outliers_df = detect_outliers(input_df, id_column='FoodTypeId', 
                                value_columns=['NumDaysEaten'],
                                z_threshold=2.5)
        outliers_df


        # Check if we have any outliers before proceeding
        if outliers_df.empty:
            print("No outliers found. Exiting.")
            
        else:
            outliers_df2 = outliers_df[['DataCollectionExerciseId', 'CountyId', 'ExerciseStartDate']]
            outliers_df2['DataCollectionExerciseId'] = outliers_df2['DataCollectionExerciseId'].astype(int)
            outliers_df3 = outliers_df2.drop_duplicates()
            base_indicators = sorted(set(col for col in outliers_df['analyzed_column'].unique()if '|' not in str(col)))
            indicator_hash = hashlib.md5('_'.join(base_indicators).encode()).hexdigest()[:8]
            outlier_type_set = '_'.join(sorted(outliers_df['outlier_type'].unique().tolist()))            
            outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)+''+indicator_hash
            # outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)+''+indicator_set+''+outlier_type_set 
            outliers_df3['QuestionnaireType'] = "HHA"
            outliers_df4 = outliers_df3[['OutlierRunName', 'DataCollectionExerciseId', 'QuestionnaireType']]

            # Delete any previous outlier test instances
            outlier_run_names = outliers_df3['OutlierRunName'].tolist()
            data_collection_exercise_ids = outliers_df3['DataCollectionExerciseId'].tolist()


            for i in range(len(outlier_run_names)):
                del_query = text("""
                    DELETE FROM outlier_runs 
                    WHERE OutlierRunName = :run_name AND DataCollectionExerciseId = :exercise_id
                """)
                
                try:
                    with engine.begin() as conn:
                        result = conn.execute(
                            del_query, 
                            {"run_name": outlier_run_names[i], "exercise_id": int(data_collection_exercise_ids[i])}
                        )
                        print(f"Deleted {result.rowcount} previous outlier runs successfully")
                except Exception as e:
                    print(f"Error occurred during deletion: {e}")

                # Insert new outlier test instances
            for _, row in outliers_df4.iterrows():
                insert_query = text("""
                    INSERT INTO outlier_runs (
                        OutlierRunName, DataCollectionExerciseId, QuestionnaireType
                    ) 
                    VALUES (:run_name, :exercise_id, :questionnaire_type)
                """)
                
                try:
                    with engine.begin() as conn:
                        conn.execute(
                            insert_query, 
                            {
                                "run_name": row['OutlierRunName'],
                                "exercise_id": int(row['DataCollectionExerciseId']),
                                "questionnaire_type": row['QuestionnaireType']
                            }
                        )
                    print(f"Inserted outlier run record successfully")
                except Exception as e:
                    print(f"Error occurred during outlier run insertion: {e}")

            # Get the OutlierRunId for the newly created run
            outlier_run_name = outliers_df3['OutlierRunName'].iloc[0]
            data_collection_exercise_id = outliers_df3['DataCollectionExerciseId'].iloc[0]

            query = text("""
                SELECT outlier_runs.OutlierRunId
                FROM outlier_runs
                WHERE OutlierRunName = :run_name 
                AND DataCollectionExerciseId = :exercise_id
            """)

            with engine.connect() as conn:
                result = conn.execute(
                    query, 
                    {"run_name": outlier_run_name, "exercise_id": int(data_collection_exercise_id)}
                )
                OutlierRunId = result.scalar()

            if OutlierRunId is None:
                print("Error: Could not retrieve OutlierRunId. Exiting.")
                exit()

            print(f"Using OutlierRunId: {OutlierRunId}")

            # Add OutlierRunId to the outliers dataframe and convert numpy types
            outliers_df5 = outliers_df.copy()
            outliers_df5['OutlierRunId'] = OutlierRunId
            #outliers_df5 = convert_numpy_types(outliers_df5)

            # Select and rename columns for the final insert
            outliers_df6 = outliers_df5[['OutlierRunId', 'WardId', 'HouseHoldId', 'analyzed_column', 
                                        'entity_id', 'outlier_type', 'OutlierValue', 'test_statistic_value', 
                                        'test_type', 'level', 'reference_mean', 'reference_std', 
                                        'population_mean', 'population_std','CountyId']]

            outliers_df6.rename(columns={
                'analyzed_column': 'Indicator',
                'outlier_type': 'OutlierType',
                'test_statistic_value': 'TestStatisticValue',
                'test_type': 'TestType',
                'level': 'Level',
                'reference_mean': 'ReferenceMean',
                'reference_std': 'ReferenceStd',
                'population_mean': 'PopulationMean',
                'population_std': 'PopulationStd',
                'entity_id':'IndicatorType'
            }, inplace=True)

            # Insert outlier details
            for _, row in outliers_df6.iterrows():
                params = {}
                for col in outliers_df6.columns:
                    value = row[col]
                    if isinstance(value, float) and np.isnan(value):
                        params[col.lower()] = None
                    elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.integer):
                        params[col.lower()] = int(value)
                    elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.floating):
                        params[col.lower()] = float(value)
                    else:
                        params[col.lower()] = value
                
                insert_query = text("""
                    INSERT INTO outliers (
                        OutlierRunId, WardId, HouseHoldId, Indicator, IndicatorType, 
                        OutlierType, OutlierValue, TestStatisticValue, TestType, Level, 
                        ReferenceMean, ReferenceStd, PopulationMean, PopulationStd,CountyId
                    ) 
                    VALUES (
                        :outlierrunid, :wardid, :householdid, :indicator, :indicatortype, 
                        :outliertype, :outliervalue, :teststatisticvalue, :testtype, :level, 
                        :referencemean, :referencestd, :populationmean, :populationstd,:countyid
                    )
                """)
                
                
                try:
                    with engine.begin() as conn:
                        conn.execute(insert_query, params)
                    
                    # Print progress after every 100 rows
                    if _ % 100 == 0:
                        print(f"Inserted {_} outlier records...")
                except Exception as e:
                    print(f"Error inserting record {_}: {e}")

    #Livestock Production
    query = """
        SELECT 
            hlp.HhaQuestionnaireSessionId, 
            hhs.CountyId, 
            hhs.LivelihoodZoneId, 
            hhs.WardId, 
            hhs.HouseHoldId, 
            hhs.SubCountyId,
            dce.DataCollectionExerciseId, 
            dce.ExerciseStartDate, 
            hlp.AnimalId,
            hlp.NumberKeptToday, 
            hlp.NumberBornInLastFourWeeks,
            hlp.NumberPurchasedInLastFourWeeks, 
            hlp.NumberSoldInLastFourWeeks,
            hlp.AveragePricePerAnimalSold, 
            hlp.NumberDiedDuringLastFourWeeks
        FROM hh_livestock_production_by_species AS hlp
        LEFT JOIN hha_questionnaire_sessions AS hhs 
            ON hlp.HhaQuestionnaireSessionId = hhs.HhaQuestionnaireSessionId
        LEFT JOIN data_collection_exercise AS dce 
            ON hhs.DataCollectionExerciseId = dce.DataCollectionExerciseId
        WHERE hhs.CountyId = %s 
            AND dce.DataCollectionExerciseId = %s;
    """
    input_df = pd.read_sql(query, engine, params=(countyId, dataCollectionExerciseId))
    input_df

    # Check if dataframe is empty before proceeding
    if input_df.empty:
        print("Warning: No data returned from query")

    else:

        # Call the method with your dataframe
        outliers_df = detect_outliers(input_df, id_column='AnimalId', 
                                value_columns=['NumberKeptToday', 'NumberBornInLastFourWeeks', 
                                'NumberPurchasedInLastFourWeeks', 'NumberSoldInLastFourWeeks',
                                'AveragePricePerAnimalSold', 'NumberDiedDuringLastFourWeeks'])
        outliers_df


        # Check if we have any outliers before proceeding
        if outliers_df.empty:
            print("No outliers found. Exiting.")
            
        else:
            outliers_df2 = outliers_df[['DataCollectionExerciseId', 'CountyId', 'ExerciseStartDate']]
            outliers_df2['DataCollectionExerciseId'] = outliers_df2['DataCollectionExerciseId'].astype(int)
            outliers_df3 = outliers_df2.drop_duplicates()
            base_indicators = sorted(set(col for col in outliers_df['analyzed_column'].unique()if '|' not in str(col)))
            indicator_hash = hashlib.md5('_'.join(base_indicators).encode()).hexdigest()[:8]
            outlier_type_set = '_'.join(sorted(outliers_df['outlier_type'].unique().tolist()))            
            outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)+''+indicator_hash
            # outliers_df3['OutlierRunName'] = outliers_df3['CountyId'].astype(str) + '' + outliers_df3['ExerciseStartDate'].astype(str)+''+indicator_set+''+outlier_type_set 
            outliers_df3['QuestionnaireType'] = "HHA"
            outliers_df4 = outliers_df3[['OutlierRunName', 'DataCollectionExerciseId', 'QuestionnaireType']]

            # Delete any previous outlier test instances
            outlier_run_names = outliers_df3['OutlierRunName'].tolist()
            data_collection_exercise_ids = outliers_df3['DataCollectionExerciseId'].tolist()


            for i in range(len(outlier_run_names)):
                del_query = text("""
                    DELETE FROM outlier_runs 
                    WHERE OutlierRunName = :run_name AND DataCollectionExerciseId = :exercise_id
                """)
                
                try:
                    with engine.begin() as conn:
                        result = conn.execute(
                            del_query, 
                            {"run_name": outlier_run_names[i], "exercise_id": int(data_collection_exercise_ids[i])}
                        )
                        print(f"Deleted {result.rowcount} previous outlier runs successfully")
                except Exception as e:
                    print(f"Error occurred during deletion: {e}")

                # Insert new outlier test instances
            for _, row in outliers_df4.iterrows():
                insert_query = text("""
                    INSERT INTO outlier_runs (
                        OutlierRunName, DataCollectionExerciseId, QuestionnaireType
                    ) 
                    VALUES (:run_name, :exercise_id, :questionnaire_type)
                """)
                
                try:
                    with engine.begin() as conn:
                        conn.execute(
                            insert_query, 
                            {
                                "run_name": row['OutlierRunName'],
                                "exercise_id": int(row['DataCollectionExerciseId']),
                                "questionnaire_type": row['QuestionnaireType']
                            }
                        )
                    print(f"Inserted outlier run record successfully")
                except Exception as e:
                    print(f"Error occurred during outlier run insertion: {e}")

            # Get the OutlierRunId for the newly created run
            outlier_run_name = outliers_df3['OutlierRunName'].iloc[0]
            data_collection_exercise_id = outliers_df3['DataCollectionExerciseId'].iloc[0]

            query = text("""
                SELECT outlier_runs.OutlierRunId
                FROM outlier_runs
                WHERE OutlierRunName = :run_name 
                AND DataCollectionExerciseId = :exercise_id
            """)

            with engine.connect() as conn:
                result = conn.execute(
                    query, 
                    {"run_name": outlier_run_name, "exercise_id": int(data_collection_exercise_id)}
                )
                OutlierRunId = result.scalar()

            if OutlierRunId is None:
                print("Error: Could not retrieve OutlierRunId. Exiting.")
                exit()

            print(f"Using OutlierRunId: {OutlierRunId}")

            # Add OutlierRunId to the outliers dataframe and convert numpy types
            outliers_df5 = outliers_df.copy()
            outliers_df5['OutlierRunId'] = OutlierRunId
            #outliers_df5 = convert_numpy_types(outliers_df5)

            # Select and rename columns for the final insert
            outliers_df6 = outliers_df5[['OutlierRunId', 'WardId', 'HouseHoldId', 'analyzed_column', 
                                        'entity_id', 'outlier_type', 'OutlierValue', 'test_statistic_value', 
                                        'test_type', 'level', 'reference_mean', 'reference_std', 
                                        'population_mean', 'population_std','CountyId']]

            outliers_df6.rename(columns={
                'analyzed_column': 'Indicator',
                'outlier_type': 'OutlierType',
                'test_statistic_value': 'TestStatisticValue',
                'test_type': 'TestType',
                'level': 'Level',
                'reference_mean': 'ReferenceMean',
                'reference_std': 'ReferenceStd',
                'population_mean': 'PopulationMean',
                'population_std': 'PopulationStd',
                'entity_id':'IndicatorType'
            }, inplace=True)

            # Insert outlier details
            for _, row in outliers_df6.iterrows():
                params = {}
                for col in outliers_df6.columns:
                    value = row[col]
                    if isinstance(value, float) and np.isnan(value):
                        params[col.lower()] = None
                    elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.integer):
                        params[col.lower()] = int(value)
                    elif hasattr(value, 'dtype') and np.issubdtype(value.dtype, np.floating):
                        params[col.lower()] = float(value)
                    else:
                        params[col.lower()] = value
                
                insert_query = text("""
                    INSERT INTO outliers (
                        OutlierRunId, WardId, HouseHoldId, Indicator, IndicatorType, 
                        OutlierType, OutlierValue, TestStatisticValue, TestType, Level, 
                        ReferenceMean, ReferenceStd, PopulationMean, PopulationStd,CountyId
                    ) 
                    VALUES (
                        :outlierrunid, :wardid, :householdid, :indicator, :indicatortype, 
                        :outliertype, :outliervalue, :teststatisticvalue, :testtype, :level, 
                        :referencemean, :referencestd, :populationmean, :populationstd,:countyid
                    )
                """)
                
                
                try:
                    with engine.begin() as conn:
                        conn.execute(insert_query, params)
                    
                    # Print progress after every 100 rows
                    if _ % 100 == 0:
                        print(f"Inserted {_} outlier records...")
                except Exception as e:
                    print(f"Error inserting record {_}: {e}")


    #     print(f"Inserted {len(outliers_df6)} outlier records successfully")
