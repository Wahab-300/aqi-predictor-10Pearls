import os
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd

from fetch_data import fetch_aqi
from process_data import calculate_aqi, PM25_BREAKPOINTS, PM10_BREAKPOINTS
from hopsworks_connection import connect_to_hopsworks, get_or_create_feature_group

load_dotenv()


# =================================== calculating on live api data ===================================
def build_live_row(city_name, lat, lon):
    data = fetch_aqi(lat, lon)
    
    components = data["list"][0]["components"]
    pm25 = components["pm2_5"]
    pm10 = components["pm10"]
    
    aqi_pm25 = calculate_aqi(pm25, PM25_BREAKPOINTS)
    aqi_pm10 = calculate_aqi(pm10, PM10_BREAKPOINTS)
    overall_aqi = max(aqi_pm25, aqi_pm10)
    
    timestamp = datetime.now()
    hour = timestamp.hour
    day_of_week = timestamp.weekday()
    month = timestamp.month
    
    row = {
        "city": city_name,
        "timestamp": timestamp,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "pm2_5": pm25,
        "pm10": pm10,
        "aqi_pm25": aqi_pm25,
        "aqi_pm10": aqi_pm10,
        "overall_aqi": overall_aqi
    }
    return row


# =================================== saving row to hopsworks ===================================
def run_live_pipeline():
    project = connect_to_hopsworks()
    fg = get_or_create_feature_group(project)
    
    row = build_live_row("bahawalpur", 29.3956, 71.6836)
    df = pd.DataFrame([row])
    fg.insert(df)
    print("Live row inserted:", row)

if __name__ == "__main__":
    run_live_pipeline()