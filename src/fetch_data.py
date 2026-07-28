import requests
from dotenv import load_dotenv
import os
import json
from datetime import datetime

load_dotenv() # Load the environment variables from the .env file
api_key = os.getenv("OPENWEATHER_API_KEY")

def fetch_aqi(lat, lon):
   
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params ={
        "lat": lat,
        "lon": lon,
        "appid": api_key,
    }
    response = requests.get(url, params=params)
    data = response.json()
    return data



def save_raw_data(city_name, data):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"data/raw/{city_name.lower()}_{timestamp}.json"
    
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)
    
    print(f"Saved: {filename}")


if __name__ == "__main__":
    cities = {
        "Karachi": (24.8607, 67.0011),
        "Lahore": (31.5497, 74.3436),
        "Islamabad": (33.6844, 73.0479),
        "Bahawalpur": (29.3956, 71.6836)
    }
    for city, (lat, lon) in cities.items():
        result = fetch_aqi(city, lat, lon)
        print(city, result)
        save_raw_data(city, result)    

