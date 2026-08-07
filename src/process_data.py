import json
import os
import pandas as pd
import glob
from datetime import datetime


# ============================= This file is 1st prototype of this project =============================

# EPA Breakpoint tables: (Conc_low, Conc_high, AQI_low, AQI_high)
# These tables are use in live_pipeline.py
PM25_BREAKPOINTS = [
    (0.0, 12.0, 0, 50),
    (12.1, 35.4, 51, 100),
    (35.5, 55.4, 101, 150),
    (55.5, 150.4, 151, 200),
    (150.5, 250.4, 201, 300),
    (250.5, 500.4, 301, 500),
]

PM10_BREAKPOINTS = [
    (0.0, 54.9, 0, 50),
    (55.0, 154.9, 51, 100),
    (155.0, 254.9, 101, 150),
    (255.0, 354.9, 151, 200),
    (355.0, 424.9, 201, 300),
    (425.0, 604.9, 301, 500),
]

# =================================== calculating AQI ===================================
# This fun is used in the live pipeline (live_pipeline.py)
def calculate_aqi(concentration, breakpoints):
    for conc_low, conc_high, aqi_low, aqi_high in breakpoints:
        if conc_low <= concentration <= conc_high:
            aqi = ((aqi_high - aqi_low) / (conc_high - conc_low)) * (concentration - conc_low) + aqi_low
            return round(aqi)


        # If concentration exceeds the highest bracket, cap at 500
    if concentration > breakpoints[-1][1]:
        return 500    
    return None


# ///////// All the fun below are used locally only for testing and debugging not use in live_pipeline.py /////////

# reading raw json and processing to find AQI
# def process_raw_file(filepath):
#     with open(filepath, "r") as f:
#         data = json.load(f)
    
#     components = data["list"][0]["components"]
#     pm25 = components["pm2_5"]
#     pm10 = components["pm10"]
    
#     aqi_pm25 = calculate_aqi(pm25, PM25_BREAKPOINTS)
#     aqi_pm10 = calculate_aqi(pm10, PM10_BREAKPOINTS)
    
#     overall_aqi = max(aqi_pm25, aqi_pm10)
    
#     return {
#         "pm2_5": pm25,
#         "pm10": pm10,
#         "aqi_pm25": aqi_pm25,
#         "aqi_pm10": aqi_pm10,
#         "overall_aqi": overall_aqi
#     }


# # extracting the city and timestamp
# def parse_filename(filename):
#     # e.g. karachi_2026-07-24_17-58-38.json
#     name = filename.replace(".json", "")
#     parts = name.split("_", 1)
#     city = parts[0]
#     timestamp = parts[1]
#     return city, timestamp


# # processing all the raw json files
# def process_all_raw_files():
#     raw_files = glob.glob("data/raw/*.json")
#     rows = []

#     for filepath in raw_files:
#         filename_only = os.path.basename(filepath)
#         city, timestamp = parse_filename(filename_only)
#         result = process_raw_file(filepath)

#         hour, day_of_week, month = extract_time_features(timestamp)



#         row = {
#             "city": city,
#             "timestamp": timestamp,
#             "hour": hour,
#             "day_of_week": day_of_week,
#             "month": month,
#             **result
#         }
#         rows.append(row) 

#     df = pd.DataFrame(rows)    
#     df.to_csv("data/processed/aqi_data.csv", index=False)
#     print(f"Processed {len(rows)} files → saved to data/processed/aqi_data.csv")


# # extract time features for timestamp string
# def extract_time_features(timestamp_str):
#     dt = datetime.strptime(timestamp_str, "%Y-%m-%d_%H-%M-%S")

#     hour = dt.hour
#     day_of_week = dt.weekday()
#     month = dt.month

#     return hour, day_of_week, month



# if __name__ == "__main__":
#     process_all_raw_files()

