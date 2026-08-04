import hopsworks
import os
from dotenv import load_dotenv
import pandas as pd


# =================================== Connection to Hopsworks ===================================
load_dotenv() # load .env

def connect_to_hopsworks():
    # read variables inside the .env
    project = hopsworks.login( 
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        host="eu-west.cloud.hopsworks.ai",
        port=443,
        project="bahawalpur_aqi_2026"
    )
    return project


# =================================== Feature group ===================================

def get_or_create_feature_group(project):
    fs = project.get_feature_store()
    
    feature_group = fs.get_or_create_feature_group(
        name="bahawalpur_aqi_features_v2",
        version=1,
        description="Hourly AQI and weather features for Bahawalpur",
        primary_key=["timestamp"],
        event_time="timestamp"
    )
    return feature_group



# =================================== Add data to feature group ===================================
def insert_historical_data(feature_group, csv_path):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    feature_group.insert(df)
    print(f"Inserted {len(df)} rows into feature group.")



if __name__ == "__main__":
    project = connect_to_hopsworks()
    fg = get_or_create_feature_group(project)
    insert_historical_data(fg, "data/processed/bahawalpur_historical.csv")


