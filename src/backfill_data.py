import pandas as pd
import requests
from process_data import calculate_aqi, PM25_BREAKPOINTS, PM10_BREAKPOINTS


def fetch_historical_data(lat, lon, start_date, end_date):
    url = "https://air-quality-api.open-meteo.com/v1/air-quality"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm10,pm2_5",
        "start_date": start_date,
        "end_date": end_date
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data


def process_historical_data(data, city_name):
    times = data["hourly"]["time"]
    pm10_values = data["hourly"]["pm10"]
    pm25_values = data["hourly"]["pm2_5"]

    rows = []

    for i in range(len(times)):
        rows.append({
            "city": city_name,
            "timestamp": times[i],
            "pm10": pm10_values[i],
            "pm2_5": pm25_values[i]
        })

    return rows



from datetime import datetime

def extract_time_features_iso(timestamp_str):
    dt = datetime.fromisoformat(timestamp_str)
    
    hour = dt.hour
    day_of_week = dt.weekday()
    month = dt.month
    
    return hour, day_of_week, month


def build_backfill_dataset(lat, lon, start_date, end_date, city_name):
    data = fetch_historical_data(lat, lon, start_date, end_date)
    raw_rows = process_historical_data(data, city_name)
    
    final_rows = []
    for row in raw_rows:
        aqi_pm25 = calculate_aqi(row["pm2_5"], PM25_BREAKPOINTS)
        aqi_pm10 = calculate_aqi(row["pm10"], PM10_BREAKPOINTS)

        if aqi_pm25 is None or aqi_pm10 is None:

            print(f"Problem row: pm2_5={row['pm2_5']}, pm10={row['pm10']}, timestamp={row['timestamp']}")
            overall_aqi = None
        else:
            overall_aqi = max(aqi_pm25, aqi_pm10)    
        
        hour, day_of_week, month = extract_time_features_iso(row["timestamp"])
        
        final_rows.append({
            "city": row["city"],
            "timestamp": row["timestamp"],
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            "pm2_5": row["pm2_5"],
            "pm10": row["pm10"],
            "aqi_pm25": aqi_pm25,
            "aqi_pm10": aqi_pm10,
            "overall_aqi": overall_aqi
        })
    
    return final_rows




if __name__ == "__main__":
    rows = build_backfill_dataset(29.3956, 71.6836, "2024-07-27", "2026-07-24", "bahawalpur")
    df = pd.DataFrame(rows)
    df.to_csv("data/processed/bahawalpur_historical.csv", index=False)
    print(f"Saved {len(rows)} rows to data/processed/bahawalpur_historical.csv")

