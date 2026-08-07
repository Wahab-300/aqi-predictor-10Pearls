import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from src.hopsworks_connection import connect_to_hopsworks, get_or_create_feature_group

st.set_page_config(page_title="Pearls AQI Predictor", page_icon="🍃", layout="wide")

# ===================================  Theme Colors ===================================
PRIMARY = "#ef4444"
PRIMARY_DARK = "#b91c1c"
PRIMARY_LIGHT = "#7f1d1d"

BG = "#0a0a0a"
CARD_BG = "#18181b"
BORDER = "#27272a"
TEXT = "#f4f4f5"
TEXT_MUTED = "#a1a1aa"
TEXT_FAINT = "#71717a"
GRID_LINE = "#27272a"

AQI_COLORS = {
    "Good": "#22c55e",
    "Moderate": "#eab308",
    "Unhealthy for Sensitive Groups": "#f97316",
    "Unhealthy": "#ef4444",
    "Very Unhealthy": "#a855f7",
    "Hazardous": "#dc2626",
}

# ===================================  Custom CSS ===================================
st.markdown(f"""
<style>

    /* Custom branded spinner */
    div[data-testid="stSpinner"] {{
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        background-color: {BG} !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        z-index: 9999 !important;
    }}
    div[data-testid="stSpinner"] > div {{
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        gap: 16px !important;
    }}
    span[data-testid="stSpinnerIcon"] {{
        display: none !important;
    }}
    div[data-testid="stSpinner"] > div::before {{
        content: "";
        width: 48px;
        height: 48px;
        border: 4px solid {PRIMARY_LIGHT};
        border-top: 4px solid {PRIMARY};
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }}
    div[data-testid="stSpinner"] p {{
        color: {TEXT};
        font-weight: 700;
        font-size: 16px;
    }}
    @keyframes spin {{
        0% {{ transform: rotate(0deg); }}
        100% {{ transform: rotate(360deg); }}
    }}


    .stApp {{ background-color: {BG}; }}

    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background-color: {CARD_BG};
        border-radius: 16px;
        border: 1px solid {BORDER} !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        margin-bottom: 4px;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"] > div {{
        padding: 26px 28px;
    }}

    /* Tighten default Streamlit block gaps */
    div[data-testid="stVerticalBlock"] > div {{
        gap: 0.5rem;
    }}

    .sticky-header {{
        position: sticky; top: 0; z-index: 999;
        background-color: {BG};
        padding: 16px 28px;
        margin: -1rem -1rem 24px -1rem;
        border-bottom: 1px solid {BORDER};
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }}

    .header-icon {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 46px; height: 46px; border-radius: 12px;
        background: linear-gradient(135deg, {PRIMARY}, {PRIMARY_DARK});
        font-size: 22px; margin-right: 14px; vertical-align: middle;
    }}
    .app-title {{ font-size: 24px; font-weight: 800; color: {TEXT}; margin: 0; }}
    .app-subtitle {{ color: {TEXT_MUTED}; font-size: 14px; margin-top: 2px; }}

    .location-badge {{
        background-color: {PRIMARY_LIGHT}; border-radius: 20px; padding: 7px 16px;
        font-size: 13px; color: #fecaca; font-weight: 600; display: inline-block;
    }}
    .updated-text {{ color: {TEXT_FAINT}; font-size: 13px; margin-top: 8px; text-align: right; }}

    .badge {{
        display: inline-block; padding: 6px 18px; border-radius: 20px;
        font-weight: 700; font-size: 13px;
    }}
    .label {{ color: {TEXT_MUTED}; font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.7px; }}
    .big-number {{ font-size: 50px; font-weight: 800; margin: 4px 0; line-height: 1; color: {TEXT}; }}

    .pollutant-value {{ font-size: 26px; font-weight: 800; color: {TEXT}; margin: 0; }}
    .pollutant-name {{ color: {TEXT_MUTED}; font-size: 13px; font-weight: 600; margin: 2px 0 0 0; }}

    .guidance-box {{
        background-color: #111113; border-radius: 10px; padding: 16px 18px;
        margin-top: 16px; font-size: 14px; color: {TEXT_MUTED}; line-height: 1.6;
        border-left: 4px solid {PRIMARY};
    }}
    .guidance-box b {{ color: {TEXT}; }}

    h1, h2, h3, h4, h5, p {{ color: {TEXT}; }}
    .section-title {{ font-size: 24px; font-weight: 800; color: {TEXT}; margin: 0 0 4px 0; }}
    .section-subtitle {{ color: {TEXT_MUTED}; font-size: 14px; margin-bottom: 18px; }}

    .stat-label {{ color: {TEXT_FAINT}; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }}
    .stat-value {{ color: {TEXT}; font-size: 20px; font-weight: 800; }}

    div[data-testid="stButton"] button {{
        background-color: {PRIMARY} !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        padding: 10px 24px !important;
        box-shadow: 0 2px 6px rgba(239,68,68,0.35);
    }}
    div[data-testid="stButton"] button:hover {{
        background-color: {PRIMARY_DARK} !important;
    }}
    div[data-testid="stButton"] button p {{
        color: #ffffff !important;
    }}

    /* Consistent spacing wrapper for section titles that sit directly on the page */
    .section-block {{
        margin-top: 8px;
        margin-bottom: 8px;
    }}

    /* Subtle divider between major page sections */
    .section-divider {{
        border: none;
        border-top: 1px solid {BORDER};
        margin: 36px 0 28px 0;
    }}

    /* Streamlit's own text elements (labels, markdown outside our custom classes) */
    .stMarkdown, .stText, label {{ color: {TEXT}; }}

   
</style>
""", unsafe_allow_html=True)

def get_aqi_category(aqi):
    if aqi <= 50: return "Good"
    elif aqi <= 100: return "Moderate"
    elif aqi <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi <= 200: return "Unhealthy"
    elif aqi <= 300: return "Very Unhealthy"
    else: return "Hazardous"

def badge_html(category):
    c = AQI_COLORS[category]
    return f"<span class='badge' style='background-color:{c}33; color:{c};'>{category}</span>"

def styled_chart(fig, height=280):
    fig.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font=dict(color=TEXT_MUTED, size=13),
        margin=dict(l=10, r=10, t=10, b=10),
        height=height,
        hovermode="closest",
        hoverlabel=dict(bgcolor="#000000", font_color="white", font_size=13),
        xaxis=dict(gridcolor=GRID_LINE, fixedrange=True, tickfont=dict(color=TEXT_MUTED)),
        yaxis=dict(gridcolor=GRID_LINE, fixedrange=True, tickfont=dict(color=TEXT_MUTED)),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False, "scrollZoom": False})

def make_gauge(value, category):
    color = AQI_COLORS[category]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 38, "color": TEXT}},
        gauge={
            "axis": {"range": [0, 300], "tickwidth": 0, "tickvals": [0, 100, 200, 300], "tickfont": {"color": TEXT_MUTED}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": CARD_BG,
            "borderwidth": 0,
            "steps": [{"range": [0, 300], "color": "#27272a"}],
        },
    ))
    fig.update_layout(height=200, margin=dict(l=30, r=30, t=20, b=10), paper_bgcolor=CARD_BG, font=dict(color=TEXT))
    return fig

# ===================================  Data + Models (cached) ===================================
@st.cache_resource(show_spinner=False)
def get_project():
    return connect_to_hopsworks()

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(_fg):
    df = _fg.read()
    return df.sort_values("timestamp")

@st.cache_resource(show_spinner=False)
def load_models(_project):
    mr = _project.get_model_registry()
    models, metrics = {}, {}
    for day, name in [(1, "ridge_day1"), (2, "ridge_day2"), (3, "ridge_day3")]:
        meta = mr.get_model(name, version=1)
        path = meta.download()
        models[day] = joblib.load(f"{path}/{name}.pkl")
        try:
            metrics[day] = meta.training_metrics or {}
        except Exception:
            metrics[day] = {}
    return models, metrics

with st.spinner("Fetching live air quality data..."):
    project = get_project()
    fg = get_or_create_feature_group(project)
    df = load_data(fg)

with st.spinner("Loading forecasting models..."):
    models, model_metrics = load_models(project)

latest_row = df.iloc[-1]


@st.cache_resource(show_spinner=False)
def load_explainers(_models, _df):
    """Builds one SHAP explainer per day, reusing already-loaded data."""
    df = _df.copy()
    df["aqi_24h_ago"] = df["overall_aqi"].shift(24)
    df["aqi_change_rate"] = df["overall_aqi"] - df["aqi_24h_ago"]
    df["aqi_rolling_avg_24h"] = df["overall_aqi"].rolling(window=24).mean()
    df = df.dropna().reset_index(drop=True)

    feature_cols = ['hour', 'day_of_week', 'month', 'pm2_5', 'pm10', 'aqi_pm25', 'aqi_pm10',
                     'overall_aqi', 'aqi_24h_ago', 'aqi_change_rate', 'aqi_rolling_avg_24h']

    split_index = int(len(df) * 0.8)
    X_train = df[feature_cols][:split_index]

    explainers = {}
    for day in [1, 2, 3]:
        masker = shap.maskers.Independent(X_train, max_samples=len(X_train))
        explainers[day] = shap.LinearExplainer(_models[day], masker)
    return explainers

with st.spinner("Preparing explanations..."):
    explainers = load_explainers(models, df)

# ===================================  Sticky Header ===================================
hcol1, hcol2, hcol3 = st.columns([2.5, 1, 0.8])
with hcol1:
    st.markdown(f"""
    <div class="sticky-header" style="display:flex; align-items:center;">
        <span class="header-icon">🍃</span>
        <div>
            <p class="app-title">Pearls AQI Predictor</p>
            <p class="app-subtitle">AI-powered air quality forecasting for Bahawalpur, Pakistan</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

with hcol2:
    now_str = datetime.now().strftime("%I:%M %p")
    st.markdown(f"""
    <div style="text-align:right; margin-top:10px;">
        <span class="location-badge">📍 Bahawalpur · Pakistan</span>
        <p class="updated-text">Updated {now_str}</p>
    </div>
    """, unsafe_allow_html=True)

with hcol3:
    st.write("")
    if st.button("Refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

st.write("")

# ===================================  Reserved Space for Alert Banner ===================================
alert_placeholder = st.empty()


# ===================================  Current AQI + Health Guidance ===================================
category = get_aqi_category(latest_row["overall_aqi"])

aqi_24h_ago_preview = df.iloc[-25]["overall_aqi"]
points_change = latest_row["overall_aqi"] - aqi_24h_ago_preview
direction = "↑" if points_change >= 0 else "↓"
sign = "+" if points_change >= 0 else "-"
direction_color = "#ef4444" if points_change >= 0 else "#22c55e"
reading_time = latest_row["timestamp"].strftime("%I:%M %p") if hasattr(latest_row["timestamp"], "strftime") else ""

col1, col2 = st.columns([1.6, 1])

with col1:
    with st.container(border=True):
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.markdown(f"""
            <p class="label">📍 Bahawalpur</p>
            <h2 style="margin:6px 0 0 0; font-size:28px; color:{TEXT};">Current Air Quality</h2>
            <p style="color:{TEXT_MUTED}; margin-top:6px; font-size:15px;">{category}</p>
            <div style="margin-top:26px;">
                <span style="color:{direction_color}; font-weight:800; font-size:17px;">{direction} {sign}{abs(points_change):.0f} points</span>
                <p style="color:{TEXT_FAINT}; font-size:13px; margin:3px 0 0 0;">vs previous reading · {reading_time}</p>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.plotly_chart(make_gauge(latest_row["overall_aqi"], category), use_container_width=True, config={"displayModeBar": False})

with col2:
    guidance_text = {
        "Good": "Air quality is satisfactory, and air pollution poses little or no risk.",
        "Moderate": "Air quality is acceptable. Unusually sensitive people should consider reducing prolonged outdoor exertion.",
        "Unhealthy for Sensitive Groups": "Sensitive groups may experience health effects. General public is less likely to be affected.",
        "Unhealthy": "Everyone may begin to experience health effects. Sensitive groups may experience more serious effects.",
        "Very Unhealthy": "Health alert: everyone may experience more serious health effects.",
        "Hazardous": "Health warning of emergency conditions. Everyone is more likely to be affected.",
    }
    with st.container(border=True):
        st.markdown(f"""
        <p class="label">🛡️ Air Quality Status</p>
        <div style="margin-top:10px;">{badge_html(category)}</div>
        <div class="guidance-box"><b>Health guidance:</b> {guidance_text[category]}</div>
        """, unsafe_allow_html=True)

# ===================================  Live Pollutants Grid ===================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="section-block">
    <p class="section-title">Current Pollutants</p>
    <p class="section-subtitle">Live pollutant concentrations at Bahawalpur</p>
</div>
""", unsafe_allow_html=True)
p1, p2 = st.columns(2)

with p1:
    with st.container(border=True):
        st.markdown(f"""
        <p class="pollutant-value">{latest_row['pm2_5']:.1f} <span style="font-size:13px; color:{TEXT_FAINT}; font-weight:500;">µg/m³</span></p>
        <p class="pollutant-name">PM2.5</p>
        """, unsafe_allow_html=True)

with p2:
    with st.container(border=True):
        st.markdown(f"""
        <p class="pollutant-value">{latest_row['pm10']:.1f} <span style="font-size:13px; color:{TEXT_FAINT}; font-weight:500;">µg/m³</span></p>
        <p class="pollutant-name">PM10</p>
        """, unsafe_allow_html=True)

# ===================================  24-Hour Trend Chart ===================================
last_24h = df.iloc[-24:]
current_v = latest_row["overall_aqi"]
avg_v = last_24h["overall_aqi"].mean()
min_v = last_24h["overall_aqi"].min()
max_v = last_24h["overall_aqi"].max()

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

with st.container(border=True):
    tcol1, tcol2 = st.columns([2, 1.8])
    with tcol1:
        st.markdown('<p class="section-title" style="font-size:20px;">24-Hour AQI Trend</p>', unsafe_allow_html=True)
        st.markdown('<p class="section-subtitle">Air quality changes over the last 24 hours</p>', unsafe_allow_html=True)
    with tcol2:
        s1, s2, s3, s4 = st.columns(4)
        for col, label, val, clr in [
            (s1, "CURRENT", current_v, PRIMARY), (s2, "AVG", avg_v, TEXT), (s3, "MIN", min_v, TEXT), (s4, "MAX", max_v, TEXT)
        ]:
            with col:
                st.markdown(f'<p class="stat-label">{label}</p><p class="stat-value" style="color:{clr};">{val:.0f}</p>', unsafe_allow_html=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=last_24h["timestamp"], y=last_24h["overall_aqi"],
        mode="lines+markers", fill="tozeroy",
        line=dict(color=PRIMARY, width=3, shape="spline"),
        marker=dict(size=6, color=PRIMARY, opacity=0),
        fillcolor="rgba(239,68,68,0.12)",
        hovertemplate="<b>%{y:.0f} AQI</b><extra></extra>"
    ))

    # Set the y-axis range to 15% above the maximum value
    y_max_24h = last_24h["overall_aqi"].max()
    fig.update_layout(yaxis=dict(range=[0, y_max_24h * 1.15]))
    styled_chart(fig)

# ===================================  Predictions ===================================
aqi_24h_ago = df.iloc[-25]["overall_aqi"]
aqi_change_rate = latest_row["overall_aqi"] - aqi_24h_ago
aqi_rolling_avg_24h = df.iloc[-24:]["overall_aqi"].mean()

live_features = pd.DataFrame([{
    "hour": latest_row["hour"], "day_of_week": latest_row["day_of_week"], "month": latest_row["month"],
    "pm2_5": latest_row["pm2_5"], "pm10": latest_row["pm10"],
    "aqi_pm25": latest_row["aqi_pm25"], "aqi_pm10": latest_row["aqi_pm10"],
    "overall_aqi": latest_row["overall_aqi"], "aqi_24h_ago": aqi_24h_ago,
    "aqi_change_rate": aqi_change_rate, "aqi_rolling_avg_24h": aqi_rolling_avg_24h
}])

predictions = {day: models[day].predict(live_features)[0] for day in [1, 2, 3]}


# ============  Hazard Alert Banner ============    
HAZARD_THRESHOLD = 201

alert_messages = []
if latest_row["overall_aqi"] >= HAZARD_THRESHOLD:
    alert_messages.append(f"Current AQI is {latest_row['overall_aqi']:.0f} — {get_aqi_category(latest_row['overall_aqi'])}")

for day in [1, 2, 3]:
    if predictions[day] >= HAZARD_THRESHOLD:
        alert_messages.append(f"Day {day} forecast is {predictions[day]:.0f} — {get_aqi_category(predictions[day])}")

if alert_messages:
    alert_text = " · ".join(alert_messages)
    alert_placeholder.markdown(f"""
    <div style="background-color:#2a0f0f; border-left:5px solid #ef4444; border-radius:10px; padding:14px 20px; margin-bottom:20px;">
        <span style="font-weight:800; color:#fca5a5;">⚠️ Hazard Alert:</span>
        <span style="color:#fecaca;"> {alert_text}</span>
    </div>
    """, unsafe_allow_html=True)

# ============  3-Day Forecast Cards ============

st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="section-block">
    <p class="section-title">3-Day AQI Forecast</p>
    <p class="section-subtitle">Predicted AQI for the next three days</p>
</div>
""", unsafe_allow_html=True)
cols = st.columns(3)
labels = {1: "24H · DAY 1", 2: "48H · DAY 2", 3: "72H · DAY 3"}

for i, day in enumerate([1, 2, 3]):
    pred = predictions[day]
    cat = get_aqi_category(pred)
    c = AQI_COLORS[cat]
    with cols[i]:
        with st.container(border=True):
            st.markdown(f"""
            <div style="height:5px; background-color:{c}; border-radius:4px; margin:-26px -28px 18px -28px;"></div>
            <p class="label">{labels[day]}</p>
            <p class="big-number" style="font-size:40px; color:{c};">{pred:.1f}</p>
            {badge_html(cat)}
            """, unsafe_allow_html=True)


# ===================================  SHAP Explanation ===================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)
st.markdown("""
<div class="section-block">
    <p class="section-title">Why This Prediction?</p>
    <p class="section-subtitle">How each factor built up today's Day 1 forecast</p>
</div>
""", unsafe_allow_html=True)

with st.container(border=True):
    shap_values = explainers[1](live_features)
    shap_row = pd.Series(shap_values.values[0], index=live_features.columns).sort_values(key=abs, ascending=False)
    base_value = shap_values.base_values[0]
    final_value = base_value + shap_row.sum()

    fig3 = go.Figure(go.Waterfall(
        measure=["relative"] * len(shap_row),
        orientation="h",
        y=shap_row.index,
        x=shap_row.values,
        base=base_value,
        connector=dict(line=dict(color=GRID_LINE, width=1)),
        increasing=dict(marker=dict(color="#ef4444")),
        decreasing=dict(marker=dict(color="#22c55e")),
        text=[f"{v:+.1f}" for v in shap_row.values],
        textposition="outside",
        textfont=dict(color=TEXT, size=12),
        hoverinfo="text",
        hovertext=[f"<b>{name}</b><br>Impact: {v:+.2f}" for name, v in zip(shap_row.index, shap_row.values)],
    ))

    fig3.update_layout(
        plot_bgcolor=CARD_BG, paper_bgcolor=CARD_BG,
        font=dict(color=TEXT_MUTED, size=13),
        margin=dict(l=10, r=10, t=30, b=10),
        height=400,
        showlegend=False,
        hovermode="closest",
        hoverdistance=50,
        hoverlabel=dict(bgcolor="#000000", font_color="white", font_size=13),
        xaxis=dict(gridcolor=GRID_LINE, fixedrange=True, title="AQI value",
                tickfont=dict(color=TEXT_MUTED, size=12),
                title_font=dict(color=TEXT_MUTED, size=13)),
        yaxis=dict(fixedrange=True, tickfont=dict(color=TEXT, size=13)),
    )
    st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    st.markdown(f"""
    <p style="color:{TEXT_MUTED}; font-size:13px; margin-top:10px;">
        Starting from the average prediction (<b style="color:{TEXT};">{base_value:.1f}</b>),
        each factor adds or subtracts, landing on today's forecast of
        <b style="color:{PRIMARY};">{final_value:.1f}</b>.<br>
        <span style="color:#ef4444; font-weight:700;">Red</span> = pushed it higher &nbsp;·&nbsp;
        <span style="color:#22c55e; font-weight:700;">Green</span> = pulled it lower
    </p>
    """, unsafe_allow_html=True)

# ===================================  Forecast Trend + Prediction System ===================================
st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

fcol1, fcol2 = st.columns([1.6, 1])

with fcol1:
    with st.container(border=True):
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <p class="section-title" style="font-size:20px; margin-bottom:0;">Predicted AQI Trend</p>
                <p class="section-subtitle">Today through 72-hour AI forecast</p>
            </div>
            {badge_html(get_aqi_category(predictions[3]))}
        </div>
        """, unsafe_allow_html=True)

        trend_x = ["Today", "Day 1", "Day 2", "Day 3"]
        trend_y = [current_v, predictions[1], predictions[2], predictions[3]]

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=trend_x, y=trend_y, mode="lines+markers",
            line=dict(color=PRIMARY, width=3, shape="spline"),
            marker=dict(size=9, color=PRIMARY),
            hovertemplate="<b>%{y:.1f} AQI</b><extra></extra>"
        ))

        # Set the y-axis range to 15% above the maximum value
        y_max_pred = max(trend_y)
        fig2.update_layout(yaxis=dict(range=[0, y_max_pred * 1.15]))
        styled_chart(fig2)

with fcol2:
    with st.container(border=True):
        st.markdown(f"""
        <p class="label">⚙️ Prediction System</p>
        <p style="color:{TEXT_MUTED}; font-size:13px; margin-top:-6px;">Machine learning forecast · Ridge Regression</p>
        <div style="margin-top:18px;">
        """, unsafe_allow_html=True)
        for day in [1, 2, 3]:
            rmse = model_metrics[day].get("rmse") if model_metrics[day] else None
            rmse_str = f"±{rmse:.2f}" if rmse is not None else "N/A"
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:10px 0; border-bottom:1px solid {BORDER};">
                <span style="color:{TEXT_MUTED}; font-size:13px;">{day*24}h model error</span>
                <span style="font-weight:800; font-size:13px; color:{PRIMARY};">RMSE {rmse_str}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ===================================  Footer ===================================
st.markdown(f"""
<hr class="section-divider">
<div style="text-align:center; padding: 20px 0 40px 0;">
    <p style="color:{TEXT_FAINT}; font-size:13px; margin:0;">
        Built by <b style="color:{TEXT_MUTED};">Abdul Wahab</b> · Data Science Intern @ 10Pearls Pakistan
    </p>
    <p style="color:{TEXT_FAINT}; font-size:12px; margin-top:4px;">
        Pearls AQI Predictor · Powered by Ridge Regression, Hopsworks & Streamlit
    </p>
</div>
""", unsafe_allow_html=True)