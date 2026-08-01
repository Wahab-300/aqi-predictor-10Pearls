import streamlit as st
import pandas as pd
import joblib
from src.hopsworks_connection import connect_to_hopsworks, get_or_create_feature_group

st.set_page_config(page_title="Bahawalpur AQI Predictor", page_icon="🌫️", layout="wide")

st.title("🌫️ Bahawalpur AQI Predictor")
st.caption("AI-powered air quality forecasting for Bahawalpur, Pakistan")

# ===================================  Connecting to Hopsworks ===================================
@st.cache_resource
def get_project():
    return connect_to_hopsworks()


@st.cache_data(ttl=3600)
def load_data(_fg):
    df = _fg.read() # _ used to avoid hashing as it is complex obj (feature group)
    return df.sort_values("timestamp")

#  Loading Data 
project = get_project()
fg = get_or_create_feature_group(project)
df = load_data(fg)

latest_row = df.iloc[-1]



# ===================================  AQI Category mapping ===================================
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good", "green"
    elif aqi <= 100:
        return "Moderate", "yellow"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "orange"
    elif aqi <= 200:
        return "Unhealthy", "red"
    elif aqi <= 300:
        return "Very Unhealthy", "purple"
    else:
        return "Hazardous", "maroon"



# ===================================  Displaying AQI Category ===================================
category, color = get_aqi_category(latest_row["overall_aqi"])

st.metric("Current AQI", int(latest_row["overall_aqi"]))
st.markdown(f"<h3 style='color:{color}'>{category}</h3>", unsafe_allow_html=True)


# ===================================  Current Pollutants ===================================
st.subheader("Current Pollutants")
col1, col2 = st.columns(2)

with col1:
    st.metric("PM2.5", f"{latest_row['pm2_5']:.1f} µg/m³")

with col2:
    st.metric("PM10", f"{latest_row['pm10']:.1f} µg/m³")


# ===================================  24-Hour Trend ===================================
st.subheader("24-Hour AQI Trend")

last_24h = df.iloc[-24:]

st.line_chart(last_24h.set_index("timestamp")["overall_aqi"])    

# ===================================  Building Live Input Features ===================================
aqi_24h_ago = df.iloc[-25]["overall_aqi"]
aqi_change_rate = latest_row["overall_aqi"] - aqi_24h_ago
aqi_rolling_avg_24h = df.iloc[-24:]["overall_aqi"].mean()

live_features = pd.DataFrame([{
    "hour": latest_row["hour"],
    "day_of_week": latest_row["day_of_week"],
    "month": latest_row["month"],
    "pm2_5": latest_row["pm2_5"],
    "pm10": latest_row["pm10"],
    "aqi_pm25": latest_row["aqi_pm25"],
    "aqi_pm10": latest_row["aqi_pm10"],
    "overall_aqi": latest_row["overall_aqi"],
    "aqi_24h_ago": aqi_24h_ago,
    "aqi_change_rate": aqi_change_rate,
    "aqi_rolling_avg_24h": aqi_rolling_avg_24h
}])


# ===================================  Loading Models from Registry ===================================
@st.cache_resource
def load_models(_project):
    mr = _project.get_model_registry()
    
    model1_meta = mr.get_model("ridge_day1", version=1)
    model1_dir = model1_meta.download()
    model_day1 = joblib.load(f"{model1_dir}/ridge_day1.pkl")
    
    model2_meta = mr.get_model("ridge_day2", version=1)
    model2_dir = model2_meta.download()
    model_day2 = joblib.load(f"{model2_dir}/ridge_day2.pkl")
    
    model3_meta = mr.get_model("ridge_day3", version=1)
    model3_dir = model3_meta.download()
    model_day3 = joblib.load(f"{model3_dir}/ridge_day3.pkl")
    
    return model_day1, model_day2, model_day3

model_day1, model_day2, model_day3 = load_models(project)

#  Predicting AQI for Next 3 Days 
pred_day1 = model_day1.predict(live_features)[0]
pred_day2 = model_day2.predict(live_features)[0]
pred_day3 = model_day3.predict(live_features)[0]


# ===================================  Displaying 3-Day Forecast ===================================
st.subheader("3-Day AQI Forecast")

col1, col2, col3 = st.columns(3)

with col1:
    cat1, color1 = get_aqi_category(pred_day1)
    st.metric("Day 1 (24h)", f"{pred_day1:.1f}")
    st.markdown(f"<span style='color:{color1}'>{cat1}</span>", unsafe_allow_html=True)

with col2:
    cat2, color2 = get_aqi_category(pred_day2)
    st.metric("Day 2 (48h)", f"{pred_day2:.1f}")
    st.markdown(f"<span style='color:{color2}'>{cat2}</span>", unsafe_allow_html=True)

with col3:
    cat3, color3 = get_aqi_category(pred_day3)
    st.metric("Day 3 (72h)", f"{pred_day3:.1f}")
    st.markdown(f"<span style='color:{color3}'>{cat3}</span>", unsafe_allow_html=True)