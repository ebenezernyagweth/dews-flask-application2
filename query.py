import pandas as pd
from sqlalchemy import create_engine


def fetch_milk_production():
  engine = create_engine(
    "mysql+mysqlconnector://root:*Database630803240081@127.0.0.1/livelihoodzones")

  query = """
    SELECT hh_crop_production_per_species.HhaQuestionnaireSessionId, hha_questionnaire_sessions.CountyId, hha_questionnaire_sessions.LivelihoodZoneId,hha_questionnaire_sessions.WardId,hha_questionnaire_sessions.HouseHoldId, hha_questionnaire_sessions.SubCountyId,
            data_collection_exercise.DataCollectionExerciseId, data_collection_exercise.ExerciseStartDate, hh_crop_production_per_species.CropId,hh_crop_production_per_species.AcresPlantedInLastFourWks,hh_crop_production_per_species.AcresHarvestedInLastFourWks,hh_crop_production_per_species.KgsHarvestedInLastFourWks,hh_crop_production_per_species.OwnProductionStockInKg,hh_crop_production_per_species.KgsSoldInLastFourWks,hh_crop_production_per_species.PricePerKg
    FROM (hh_crop_production_per_species
          LEFT JOIN hha_questionnaire_sessions ON (hh_crop_production_per_species.HhaQuestionnaireSessionId = hha_questionnaire_sessions.HhaQuestionnaireSessionId))
          LEFT JOIN data_collection_exercise ON (hha_questionnaire_sessions.DataCollectionExerciseId = data_collection_exercise.DataCollectionExerciseId)
    WHERE (hha_questionnaire_sessions.CountyId = '32' AND data_collection_exercise.DataCollectionExerciseId = '4')
    
"""
  df = pd.read_sql(query, engine)

  # Fill NaT/NaN values
  df = df.fillna("")

  # Convert datetime columns to string
  for col in df.select_dtypes(include=['datetime64']):
    df[col] = df[col].astype(str)

  return df.to_dict(orient='records')
