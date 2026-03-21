import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from scipy.io import arff

st.set_page_config(
    page_title="OzoneWatch",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #0a0a0a;
        color: #f0ede6;
    }

    [data-testid="stHeader"] {
        background: #0a0a0a;
    }

    section[data-testid="stSidebar"] {
        background: #0a0a0a;
    }

    .hero-wrap {
        border-bottom: 1px solid #222;
        padding: 4rem 0 3rem 0;
        margin-bottom: 3rem;
    }

    .hero-tag {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #555;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: #0d1f0d;
        color: #22c55e;
        font-size: 0.65rem;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        border: 1px solid #1a3a1a;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    .pulse {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #22c55e;
        animation: pulse 2s infinite;
        display: inline-block;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }

    .hero-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: clamp(50px, 8vw, 110px);
        line-height: 0.9;
        letter-spacing: 3px;
        color: #f0ede6;
        margin: 0.5rem 0 1.5rem 0;
    }

    .hero-title span { color: #333; }

    .hero-sub {
        font-size: 0.9rem;
        color: #555;
        font-weight: 300;
        letter-spacing: 0.5px;
        max-width: 500px;
    }

    .section-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #444;
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #1a1a1a;
    }

    .weather-card {
        background: #111;
        border: 1px solid #1e1e1e;
        border-radius: 4px;
        padding: 1.5rem;
        margin-bottom: 0.5rem;
    }

    .w-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #444;
        margin-bottom: 0.4rem;
    }

    .w-value {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 2.8rem;
        letter-spacing: 2px;
        color: #f0ede6;
        line-height: 1;
    }

    .w-unit {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: #444;
        font-weight: 300;
    }

    .result-wrap-normal {
        background: #0a130a;
        border: 1px solid #1a2e1a;
        border-radius: 4px;
        padding: 3rem;
        text-align: center;
        margin-top: 2rem;
    }

    .result-wrap-ozone {
        background: #130a0a;
        border: 1px solid #2e1a1a;
        border-radius: 4px;
        padding: 3rem;
        text-align: center;
        margin-top: 2rem;
    }

    .result-icon {
        font-size: 2rem;
        margin-bottom: 1rem;
    }

    .result-title-normal {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.5rem;
        letter-spacing: 4px;
        color: #22c55e;
        line-height: 1;
    }

    .result-title-ozone {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 3.5rem;
        letter-spacing: 4px;
        color: #ef4444;
        line-height: 1;
    }

    .result-score-normal {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 5rem;
        color: #22c55e;
        letter-spacing: 3px;
        line-height: 1;
        margin: 0.5rem 0;
    }

    .result-score-ozone {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 5rem;
        color: #ef4444;
        letter-spacing: 3px;
        line-height: 1;
        margin: 0.5rem 0;
    }

    .result-desc {
        font-size: 0.85rem;
        color: #555;
        margin-top: 0.8rem;
        font-weight: 300;
        letter-spacing: 0.5px;
    }

    div[data-testid="stButton"] button {
        background: #f0ede6 !important;
        color: #0a0a0a !important;
        border: none !important;
        border-radius: 2px !important;
        padding: 0.8rem 2rem !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }

    div[data-testid="stButton"] button:hover {
        opacity: 0.85 !important;
    }

    .footer-wrap {
        border-top: 1px solid #1a1a1a;
        margin-top: 4rem;
        padding-top: 2rem;
        text-align: center;
    }

    .footer-text {
        font-size: 0.65rem;
        color: #333;
        letter-spacing: 2px;
        text-transform: uppercase;
    }

    [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"],
    [data-testid="stMetricDelta"] {
        color: #f0ede6 !important;
    }

    .stSlider label { color: #555 !important; }
    .stSlider [data-testid="stMarkdownContainer"] p { color: #555 !important; }

    div[data-baseweb="slider"] div[role="slider"] {
        background: #f0ede6 !important;
    }

    [data-testid="stSlider"] > div > div > div {
        background: #333 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    features = joblib.load('features.pkl')
    return model, features

@st.cache_data
def load_data():
    data, meta = arff.loadarff('data/ozone-level-8hr.arff')
    df = pd.DataFrame(data)
    df = df.map(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    df.replace('?', np.nan, inplace=True)
    X = df.drop('Class', axis=1).astype(float)
    return X

def get_houston_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 29.7604,
            "longitude": -95.3698,
            "current": ["temperature_2m", "relative_humidity_2m",
                       "surface_pressure", "wind_speed_10m"],
            "hourly": ["temperature_850hPa", "temperature_700hPa"],
            "forecast_days": 1
        }
        response = requests.get(url, params=params)
        data = response.json()
        current = data['current']
        hourly = data['hourly']
        return {
            'temp_ground': current['temperature_2m'],
            'humidity': current['relative_humidity_2m'],
            'pressure': current['surface_pressure'],
            'wind_speed': current['wind_speed_10m'],
            'temp_850': hourly['temperature_850hPa'][0],
            'temp_700': hourly['temperature_700hPa'][0],
        }, True
    except:
        return None, False

model, features = load_model()
X = load_data()

# HERO
st.markdown("""
<div class="hero-wrap">
    <div class="hero-tag">
        Houston, TX · Atmospheric Risk Detection
        <span class="live-badge"><span class="pulse"></span> Live</span>
    </div>
    <div class="hero-title">
        OZONE<br>
        <span>WATCH</span>
    </div>
    <div class="hero-sub">
        Real-time air quality risk prediction using live atmospheric data and machine learning.
    </div>
</div>
""", unsafe_allow_html=True)

weather, success = get_houston_weather()

if success:
    st.markdown('<div class="section-label">Current Conditions — Houston, TX</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="weather-card">
            <div class="w-label">Temperature</div>
            <div class="w-value">{weather['temp_ground']}<span class="w-unit"> °C</span></div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="weather-card">
            <div class="w-label">Humidity</div>
            <div class="w-value">{weather['humidity']}<span class="w-unit"> %</span></div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="weather-card">
            <div class="w-label">Pressure</div>
            <div class="w-value">{weather['pressure']}<span class="w-unit"> hPa</span></div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="weather-card">
            <div class="w-label">Wind Speed</div>
            <div class="w-value">{weather['wind_speed']}<span class="w-unit"> km/h</span></div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    inputs = {col: float(X[col].mean()) for col in X.columns}
    inputs['V62'] = weather['temp_ground']
    inputs['V67'] = weather['temp_850'] if weather['temp_850'] else float(X['V67'].mean())
    inputs['V57'] = weather['temp_700'] if weather['temp_700'] else float(X['V57'].mean())
    inputs['V53'] = weather['humidity']
    inputs['V56'] = weather['pressure']

    if st.button("PREDICT TODAY'S OZONE RISK"):
        input_df = pd.DataFrame([inputs])
        proba = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]

        if prediction == 1:
            st.markdown(f"""
            <div class="result-wrap-ozone">
                <div class="result-title-ozone">OZONE DAY DETECTED</div>
                <div class="result-score-ozone">{proba:.1%}</div>
                <div class="result-desc">Risk score based on current atmospheric conditions.<br>
                Authorities should consider issuing air quality warnings.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-wrap-normal">
                <div class="result-title-normal">NORMAL DAY</div>
                <div class="result-score-normal">{proba:.1%}</div>
                <div class="result-desc">Risk score based on current atmospheric conditions.<br>
                Air quality is within safe limits today.</div>
            </div>""", unsafe_allow_html=True)

else:
    st.markdown('<div class="section-label">Manual Input Mode</div>', unsafe_allow_html=True)
    inputs = {col: float(X[col].mean()) for col in X.columns}
    col1, col2 = st.columns(2)
    with col1:
        inputs['V62'] = st.slider('Solar Radiation', float(X['V62'].min()), float(X['V62'].max()), float(X['V62'].mean()))
        inputs['V67'] = st.slider('Upper Atmosphere Temp', float(X['V67'].min()), float(X['V67'].max()), float(X['V67'].mean()))
        inputs['V57'] = st.slider('Mid Atmosphere Temp', float(X['V57'].min()), float(X['V57'].max()), float(X['V57'].mean()))
    with col2:
        inputs['V53'] = st.slider('Humidity Index', float(X['V53'].min()), float(X['V53'].max()), float(X['V53'].mean()))
        inputs['V56'] = st.slider('Pressure Reading', float(X['V56'].min()), float(X['V56'].max()), float(X['V56'].mean()))

    if st.button("PREDICT OZONE RISK"):
        input_df = pd.DataFrame([inputs])
        proba = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.markdown(f"""
            <div class="result-wrap-ozone">
                <div class="result-title-ozone">OZONE DAY DETECTED</div>
                <div class="result-score-ozone">{proba:.1%}</div>
                <div class="result-desc">Authorities should consider issuing air quality warnings.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-wrap-normal">
                <div class="result-title-normal">NORMAL DAY</div>
                <div class="result-score-normal">{proba:.1%}</div>
                <div class="result-desc">Air quality is within safe limits today.</div>
            </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer-wrap">
    <div class="footer-text">
        OzoneWatch · Logistic Regression + SMOTE + SHAP · Open-Meteo API · 2534 days Houston data
    </div>
</div>
""", unsafe_allow_html=True)