# 🍃 Pearls AQI Predictor

AI-powered air quality forecasting dashboard for **Bahawalpur, Pakistan** — built as a Data Science internship project at **10Pearls Pakistan**.

**🔗 Live app:** https://aqi-predictor-10pearls-jhxpungzhgjnhdmzspzgft.streamlit.app/

---

## What it does

Predicts Air Quality Index (AQI) up to **3 days ahead** using a fully automated, serverless ML pipeline — live hourly data collection, daily model retraining, and a real-time interactive dashboard with explainable predictions (SHAP).

- **24h / 48h / 72h forecasts** — three separate models, one per horizon
- **Live current AQI** with health guidance and hazard alerts
- **"Why this prediction?"** — SHAP waterfall chart explaining each forecast
- **Fully automated** — hourly data fetch + daily retraining via GitHub Actions

## Tech stack

| Layer                          | Tool                                                    |
| ------------------------------ | ------------------------------------------------------- |
| Live data                      | OpenWeather Air Pollution API                           |
| Historical backfill            | Open-Meteo (2 years, hourly)                            |
| Feature Store / Model Registry | Hopsworks (serverless, free tier)                       |
| Model                          | Ridge Regression (beat Random Forest on all 3 horizons) |
| Explainability                 | SHAP (LinearExplainer)                                  |
| Dashboard                      | Streamlit + Plotly                                      |
| Automation                     | GitHub Actions (hourly fetch, daily retrain)            |

## Results

| Horizon     | R²   | RMSE  |
| ----------- | ---- | ----- |
| Day 1 (24h) | 0.31 | 34.13 |
| Day 2 (48h) | 0.19 | 36.80 |
| Day 3 (72h) | 0.15 | 37.67 |

Accuracy decreasing with horizon is expected and realistic — analogous to real weather forecasting (24h forecasts are always more reliable than 72h ones).

## Project structure

```
aqi-predictor/
├── app.py                  # Streamlit dashboard
├── src/
│   ├── fetch_data.py       # Live OpenWeather API calls
│   ├── backfill_data.py    # Historical Open-Meteo backfill
│   ├── process_data.py     # EPA AQI calculation
│   ├── live_pipeline.py    # Hourly production pipeline
│   ├── train_model.py      # Daily training + registration
│   ├── hopsworks_connection.py
│   └── check_data.py       # Debug/verification tool
├── notebooks/
│   └── eda_bahawalpur_aqi.ipynb
├── .github/workflows/
│   ├── fetch_data.yml      # Hourly cron
│   └── train_model.yml     # Daily cron
└── requirements.txt
```

## Full project write-up

See [`PROJECT_SUMMARY.md`](./PROJECT_SUMMARY.md) for the complete journey — architecture decisions, every bug encountered and fixed, and the full MLOps pipeline design. A printable PDF version is in [`docs/Project_Summary.pdf`](./docs/Project_Summary.pdf).

## Author

**Abdul Wahab** · Data Science Intern @ 10Pearls Pakistan
