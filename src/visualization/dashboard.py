"""Streamlit dashboard and monitoring page with cavitation explainability."""

import streamlit as st

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.generator import MarinePumpVibrationDataGenerator
from src.features.frequency_features import FrequencyFeatureExtractor
from src.features.time_features import extract_time_features
from src.visualization.explainability import explain_cavitation_prediction


def _run_prediction(severity: str):
    generator = MarinePumpVibrationDataGenerator(sample_rate=10000, duration=1.0)
    baseline = generator.generate_vibration_signal()
    signal = (
        generator.add_cavitation_effect(baseline, severity=severity)
        if severity != "none"
        else baseline
    )

    time_features = extract_time_features(signal)
    freq_extractor = FrequencyFeatureExtractor(generator)
    freq_features = freq_extractor.extract_frequency_features(signal)
    explanation = explain_cavitation_prediction(time_features, freq_features)
    return explanation, time_features, freq_features


def _render_explainability(predicted: bool, reasons):
    if predicted:
        st.error("⚠️ Cavitation predicted")
        st.subheader("Why cavitation is predicted")
        for reason in reasons:
            st.markdown(f"- {reason}")
    else:
        st.success("✅ No cavitation predicted")


def create_dashboard() -> None:
    """Create the Streamlit dashboard UI."""
    st.set_page_config(page_title="Marine Pump Cavitation", layout="wide")
    st.title("Marine Pump Cavitation Dashboard")

    dashboard_tab, monitoring_tab = st.tabs(["Dashboard", "Monitoring"])

    with dashboard_tab:
        st.header("Dashboard")
        severity = st.selectbox("Simulated condition", ["none", "mild", "moderate", "severe"], index=2)
        explanation, time_features, freq_features = _run_prediction(severity)

        st.metric("Risk score", f"{explanation.risk_score:.2f}")
        _render_explainability(explanation.cavitation_predicted, explanation.reasons)

        st.caption("Top indicators")
        st.json(
            {
                "crest_factor": round(float(time_features.get("crest_factor", 0.0)), 3),
                "kurtosis": round(float(time_features.get("kurtosis", 0.0)), 3),
                "cavitation_indicator": round(float(freq_features.get("cavitation_indicator", 0.0)), 3),
                "spectral_bandwidth_hz": round(float(freq_features.get("spectral_bandwidth_hz", 0.0)), 1),
            }
        )

    with monitoring_tab:
        st.header("Monitoring")
        st.write("Real-time style monitoring snapshot")
        severity = st.select_slider("Incoming vibration pattern", ["none", "mild", "moderate", "severe"], value="moderate")
        explanation, _, _ = _run_prediction(severity)
        st.metric("Current risk", f"{explanation.risk_score:.2f}")
        _render_explainability(explanation.cavitation_predicted, explanation.reasons)


if __name__ == "__main__":
    create_dashboard()
