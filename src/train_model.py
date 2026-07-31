import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge
from hopsworks_connection import connect_to_hopsworks, get_or_create_feature_group


# connect to hopsworks project
project = connect_to_hopsworks()
fg = get_or_create_feature_group(project)

# shift lastest 72 rows (3 days) to avoid NaN error
df = fg.read()
df['aqi_24h_ago'] = df['overall_aqi'].shift(24)
df['aqi_change_rate'] = df['overall_aqi'] - df['aqi_24h_ago']
df['aqi_rolling_avg_24h'] = df['overall_aqi'].rolling(window=24).mean()
df['target_day1'] = df['overall_aqi'].shift(-24) # for day 1 prediction
df['target_day2'] = df['overall_aqi'].shift(-48) # for day 2
df['target_day3'] = df['overall_aqi'].shift(-72) # for day 3


df = df.dropna()

# print(df.shape)
# print(df.head())
# print(df.tail())



# =================================== Feature cols ===================================
feature_cols = ['hour', 'day_of_week', 'month', 'pm2_5', 'pm10', 'aqi_pm25', 'aqi_pm10', 
                 'overall_aqi', 'aqi_24h_ago', 'aqi_change_rate', 'aqi_rolling_avg_24h']

X = df[feature_cols]
y_day1 = df['target_day1']
y_day2 = df['target_day2']
y_day3 = df['target_day3']



# =================================== Splitting data ===================================
split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y1_train, y1_test = y_day1[:split_index], y_day1[split_index:]
y2_train, y2_test = y_day2[:split_index], y_day2[split_index:]
y3_train, y3_test = y_day3[:split_index], y_day3[split_index:]

print(X_train.shape, X_test.shape)

# =================================== Training Machine Learning Models ===================================

# ============= Evaluation Fun =============
def train_and_evaluate(model, X_train, X_test, y_train, y_test, label):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"--- {label} ---")
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.2f}")
    return model, {"rmse": rmse, "mae": mae, "r2": r2}



# //////// Random Forest ////////
model_day1_rf, metrics_day1_rf = train_and_evaluate(RandomForestRegressor(random_state=42), X_train, X_test, y1_train, y1_test, "Day 1 - Random Forest")
model_day2_rf, metrics_day2_rf = train_and_evaluate(RandomForestRegressor(random_state=42), X_train, X_test, y2_train, y2_test, "Day 2 - Random Forest")
model_day3_rf, metrics_day3_rf = train_and_evaluate(RandomForestRegressor(random_state=42), X_train, X_test, y3_train, y3_test, "Day 3 - Random Forest")


# //////// Ridge Regression ///////
model_day1_ridge, metrics_day1_ridge = train_and_evaluate(Ridge(), X_train, X_test, y1_train, y1_test, "Day 1 - Ridge")
model_day2_ridge, metrics_day2_ridge = train_and_evaluate(Ridge(), X_train, X_test, y2_train, y2_test, "Day 2 - Ridge")
model_day3_ridge, metrics_day3_ridge = train_and_evaluate(Ridge(), X_train, X_test, y3_train, y3_test, "Day 3 - Ridge")



# =================================== Saving models to local file ===================================
os.makedirs("models", exist_ok=True)

joblib.dump(model_day1_ridge, "models/ridge_day1.pkl")
joblib.dump(model_day2_ridge, "models/ridge_day2.pkl")
joblib.dump(model_day3_ridge, "models/ridge_day3.pkl")

print("Models saved successfully.")


# =================================== Register model to the model Registery ===================================
def register_model(project, model_path, model_name, metrics):
    mr = project.get_model_registry()
    
    model = mr.python.create_model(
        name=model_name,
        metrics=metrics,
        description=f"Ridge Regression model for {model_name}"
    )
    model.save(model_path)
    print(f"Registered model: {model_name}")


## Call the registered models
register_model(project, "models/ridge_day1.pkl", "ridge_day1", metrics_day1_ridge)
register_model(project, "models/ridge_day2.pkl", "ridge_day2", metrics_day2_ridge)
register_model(project, "models/ridge_day3.pkl", "ridge_day3", metrics_day3_ridge)