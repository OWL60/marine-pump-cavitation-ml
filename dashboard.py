"""Streamlit dashboard for marine pump cavitation prediction."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import streamlit as st

from src.data import MarinePumpVibrationDataGenerator
from src.features.frequency_features import FrequencyFeatureExtractor
from src.features.time_features import extract_time_features

st.set_page_config(page_title="Marine Cavitation Dashboard", layout="wide")
st.title("Marine Pump Cavitation Predictor")

st.sidebar.header("Signal Parameters")
sample_rate = st.sidebar.slider("Sample Rate", 2000, 20000, 10000, step=500)
shaft_freq = st.sidebar.slider("Shaft RPM", 500, 3600, 1750, step=50)
duration = st.sidebar.slider("Duration (s)", 1.0, 4.0, 1.0, step=0.5)
severity = st.sidebar.selectbox("Cavitation Severity", ["mild", "moderate", "severe"])

model_path = Path("artifacts/models/random_forest.joblib")

if st.button("Generate and Predict"):
    generator = MarinePumpVibrationDataGenerator(sample_rate=sample_rate, shaft_freq=shaft_freq, duration=duration)
    normal = generator.generate_vibration_signal()
    signal = generator.add_cavitation_effect(normal, severity=severity)

    freq_extractor = FrequencyFeatureExtractor(generator)
    time_values = list(extract_time_features(signal).values())
    freq_values = list(freq_extractor.extract_frequency_features(signal).values())
    features = np.array(time_values + freq_values, dtype=float).reshape(1, -1)

    st.subheader("Generated Signal")
    st.line_chart(signal)

    if model_path.exists():
        model = joblib.load(model_path)
        prob = float(model.predict_proba(features)[0, 1])
        pred = int(prob >= 0.5)
        label = "CAVITATION" if pred == 1 else "NORMAL"
        st.metric("Prediction", label)
        st.metric("Cavitation Probability", f"{prob:.2%}")
    else:
        st.warning("Model not found. Run: `python main.py --mode run_all`")
