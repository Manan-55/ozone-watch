import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
from scipy.io import arff

# Load saved model
@st.cache_resource
def load_model():
    model = joblib.load('model.pkl')
    features = joblib.load('features.pkl')
    return model, features

# Load data for slider ranges only
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
        weather = {
            'temp_ground': current['temperature_2m'],
            'humidity': current['relative_humidity_2m'],
            'pressure': current['surface_pressure'],
            'wind_speed': current['wind_speed_10m'],
            'temp_850': hourly['temperature_850hPa'][0],
            'temp_700': hourly['temperature_700hPa'][0],
        }
        return weather, True
    except:
        return None, False

model, features = load_model()
X = load_data()

# UI
st.title("🌫️ OzoneWatch")
st.subheader("Live Air Quality Risk Prediction — Houston, TX")
st.markdown("---")

weather, success = get_houston_weather()

if success:
    st.success("✅ Live weather data fetched from Houston, TX")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Temperature", f"{weather['temp_ground']}°C")
    col2.metric("💧 Humidity", f"{weather['humidity']}%")
    col3.metric("🔵 Pressure", f"{weather['pressure']} hPa")
    col4.metric("💨 Wind Speed", f"{weather['wind_speed']} km/h")
    
    st.markdown("---")
    
    inputs = {col: float(X[col].mean()) for col in X.columns}
    inputs['V62'] = weather['temp_ground']
    inputs['V67'] = weather['temp_850'] if weather['temp_850'] else float(X['V67'].mean())
    inputs['V57'] = weather['temp_700'] if weather['temp_700'] else float(X['V57'].mean())
    inputs['V53'] = weather['humidity']
    inputs['V56'] = weather['pressure']
    
    if st.button("🔍 Predict Today's Ozone Risk", use_container_width=True):
        input_df = pd.DataFrame([inputs])
        proba = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        
        st.markdown("---")
        
        if prediction == 1:
            st.error("⚠️ OZONE DAY DETECTED")
            st.metric("Risk Score", f"{proba:.1%}", delta="High Risk")
            st.write("🚨 Authorities should issue air quality warnings immediately.")
        else:
            st.success("✅ NORMAL DAY")
            st.metric("Risk Score", f"{proba:.1%}", delta="Low Risk")
            st.write("😊 Air quality is within safe limits today.")
else:
    st.warning("⚠️ Could not fetch live data. Using manual input mode.")
    inputs = {col: float(X[col].mean()) for col in X.columns}
    col1, col2 = st.columns(2)
    with col1:
        inputs['V62'] = st.slider('☀️ Solar Radiation', float(X['V62'].min()), float(X['V62'].max()), float(X['V62'].mean()))
        inputs['V67'] = st.slider('🌡️ Upper Atmosphere Temp', float(X['V67'].min()), float(X['V67'].max()), float(X['V67'].mean()))
        inputs['V57'] = st.slider('🌡️ Mid Atmosphere Temp', float(X['V57'].min()), float(X['V57'].max()), float(X['V57'].mean()))
    with col2:
        inputs['V53'] = st.slider('💧 Humidity Index', float(X['V53'].min()), float(X['V53'].max()), float(X['V53'].mean()))
        inputs['V56'] = st.slider('🔵 Pressure Reading', float(X['V56'].min()), float(X['V56'].max()), float(X['V56'].mean()))
    
    if st.button("🔍 Predict Ozone Risk", use_container_width=True):
        input_df = pd.DataFrame([inputs])
        proba = model.predict_proba(input_df)[0][1]
        prediction = model.predict(input_df)[0]
        
        if prediction == 1:
            st.error("⚠️ OZONE DAY DETECTED")
            st.metric("Risk Score", f"{proba:.1%}", delta="High Risk")
            st.write("🚨 Authorities should issue air quality warnings immediately.")
        else:
            st.success("✅ NORMAL DAY")
            st.metric("Risk Score", f"{proba:.1%}", delta="Low Risk")
            st.write("😊 Air quality is within safe limits today.")

st.markdown("---")
st.caption("Built with Logistic Regression | SMOTE | SHAP | Live data from Open-Meteo API | Trained on 2534 days of Houston atmospheric data")