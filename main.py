"""Executable dashboard for marine pump cavitation prediction and explainability."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data import MarinePumpVibrationDataGenerator
from src.features.frequency_features import FrequencyFeatureExtractor
from src.features.time_features import extract_time_features


def _simulate_signal(
    marine_condition: str,
    cavitation_severity: str,
    include_engine_load: bool,
    sample_rate: int,
    duration: float,
    shaft_freq: int,
) -> np.ndarray:
    """Create a synthetic marine vibration signal with optional cavitation."""
    generator = MarinePumpVibrationDataGenerator(
        sample_rate=sample_rate,
        duration=duration,
        shaft_freq=shaft_freq,
    )
    signal = generator.generate_vibration_signal()
    signal = generator.add_ship_motion(
        signal,
        marine_condition=marine_condition,
        include_engine_load=include_engine_load,
    )
    if cavitation_severity != "none":
        signal = generator.add_cavitation_effect(signal, severity=cavitation_severity)
    return signal


def _risk_score_from_features(
    features: Dict[str, float],
    marine_condition: str,
) -> Tuple[float, List[Tuple[str, float]]]:
    """Compute a transparent weighted risk score and contribution list."""
    cavitation_indicator = min(features.get("cavitation_indicator", 0.0), 2.5) / 2.5
    energy_high = min(features.get("energy_ratio_high", 0.0), 1.0)
    entropy = min(features.get("frequency_entropy", 0.0), 10.0) / 10.0
    crest_factor = min(features.get("crest_factor", 0.0), 12.0) / 12.0
    kurtosis = min(features.get("kurtosis", 0.0), 12.0) / 12.0

    marine_bias = {"calm": 0.04, "moderate": 0.09, "rough": 0.14}.get(
        marine_condition,
        0.09,
    )

    weighted_contrib = {
        "High-frequency cavitation signature": 0.36 * cavitation_indicator,
        "High-band vibration energy": 0.24 * energy_high,
        "Spectral complexity (entropy)": 0.13 * entropy,
        "Impulsive vibration (crest factor)": 0.15 * crest_factor,
        "Shock/non-Gaussian events (kurtosis)": 0.12 * kurtosis,
        "Marine condition stress factor": marine_bias,
    }
    raw_score = sum(weighted_contrib.values())
    risk_score = float(np.clip(raw_score, 0.0, 1.0))

    normalized = []
    if raw_score > 0:
        for label, value in weighted_contrib.items():
            normalized.append((label, (value / raw_score) * 100.0))
    else:
        normalized = [(label, 0.0) for label in weighted_contrib]
    normalized.sort(key=lambda x: x[1], reverse=True)
    return risk_score, normalized


def _horizon_risks(risk_score: float) -> Dict[str, float]:
    """Generate 24h and 48h cavitation probabilities from baseline score."""
    risk_24 = float(np.clip(risk_score * 0.82, 0.0, 1.0))
    risk_48 = float(np.clip(risk_score * 1.08 + 0.07, 0.0, 1.0))
    return {"24h": risk_24, "48h": risk_48}


def _recommendations(contributions: List[Tuple[str, float]]) -> List[str]:
    """Action plan mapped from top explainability contributors."""
    recs = []
    top = [item[0] for item in contributions[:3]]

    if "High-frequency cavitation signature" in top:
        recs.append(
            "Increase NPSH margin: reduce suction lift, lower fluid temperature, and open suction-side restrictions."
        )
    if "High-band vibration energy" in top:
        recs.append(
            "Inspect impeller and wear rings for pitting/erosion; verify clearances and trim if needed."
        )
    if "Impulsive vibration (crest factor)" in top or "Shock/non-Gaussian events (kurtosis)" in top:
        recs.append(
            "Check inlet for intermittent air ingress, clogged strainers, and unstable flow causing bubble collapse bursts."
        )
    if "Marine condition stress factor" in top:
        recs.append(
            "During rough sea-state, avoid aggressive load swings and keep pump operating near best efficiency point (BEP)."
        )

    if not recs:
        recs.append(
            "Maintain present operating point and continue trending vibration/frequency indicators every shift."
        )
    return recs


def _expectation_text(risk_24: float, risk_48: float) -> str:
    """Operational expectation guidance based on short-term forecast."""
    if risk_48 >= 0.75:
        return (
            "High likelihood of cavitation progression within 48 hours. Expect rising high-frequency noise, "
            "efficiency drop, and possible accelerated impeller erosion unless corrected quickly."
        )
    if risk_24 >= 0.45:
        return (
            "Moderate early-warning state. Expect intermittent cavitation signatures and occasional impulsive bursts. "
            "With mitigation, progression can often be stabilized."
        )
    return (
        "Low near-term cavitation risk. Expect normal operation, with routine monitoring recommended to capture "
        "any trend change early."
    )


def render_streamlit_dashboard() -> None:
    """Render full predictive + explainability dashboard in Streamlit."""
    import matplotlib.pyplot as plt
    import streamlit as st

    st.set_page_config(page_title="Marine Pump Cavitation Dashboard", layout="wide")

    st.sidebar.title("Controls")
    theme = st.sidebar.radio("Theme", options=["White", "Black"], horizontal=True)

    if theme == "Black":
        st.markdown(
            """
            <style>
              .stApp { background-color: #0e1117; color: #f7f7f7; }
              [data-testid="stMetricValue"] { color: #f7f7f7; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
              .stApp { background-color: #ffffff; color: #111111; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    marine_condition = st.sidebar.selectbox("Marine condition", ["calm", "moderate", "rough"])
    cavitation_severity = st.sidebar.selectbox(
        "Simulated cavitation severity",
        ["none", "mild", "moderate", "severe"],
        index=2,
    )
    include_engine_load = st.sidebar.checkbox("Include engine load variability", value=True)
    sample_rate = st.sidebar.slider("Sample rate (Hz)", 1000, 20000, 10000, step=1000)
    duration = st.sidebar.slider("Signal duration (s)", 1.0, 5.0, 2.0, step=0.5)
    shaft_freq = st.sidebar.slider("Pump shaft speed (RPM)", 1200, 3600, 1750, step=50)

    st.title("Marine Pump Cavitation Prediction Dashboard")
    st.caption(
        "24/48-hour cavitation risk forecast + explainability + mitigation guidance in one view."
    )

    signal = _simulate_signal(
        marine_condition=marine_condition,
        cavitation_severity=cavitation_severity,
        include_engine_load=include_engine_load,
        sample_rate=sample_rate,
        duration=duration,
        shaft_freq=shaft_freq,
    )
    generator = MarinePumpVibrationDataGenerator(
        sample_rate=sample_rate,
        duration=duration,
        shaft_freq=shaft_freq,
    )
    freq_extractor = FrequencyFeatureExtractor(generator)
    freq_features = freq_extractor.extract_frequency_features(signal)
    time_features = extract_time_features(signal)
    all_features = {**time_features, **freq_features}

    risk_score, contributions = _risk_score_from_features(all_features, marine_condition)
    horizon = _horizon_risks(risk_score)
    recommendations = _recommendations(contributions)
    expectation = _expectation_text(horizon["24h"], horizon["48h"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Cavitation risk (24h)", f"{horizon['24h']*100:.1f}%")
    col2.metric("Cavitation risk (48h)", f"{horizon['48h']*100:.1f}%")
    col3.metric("Current risk score", f"{risk_score*100:.1f}%")

    t = np.arange(len(signal)) / sample_rate
    signal_df = pd.DataFrame({"Time (s)": t, "Vibration amplitude": signal}).set_index("Time (s)")
    st.subheader("Time-domain vibration signal")
    st.line_chart(signal_df, use_container_width=True)

    freq = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    spectrum = np.abs(np.fft.rfft(signal))
    spectrum_df = pd.DataFrame({"Frequency (Hz)": freq, "Amplitude": spectrum}).set_index("Frequency (Hz)")
    st.subheader("Frequency spectrum")
    st.line_chart(spectrum_df, use_container_width=True)

    explain_df = pd.DataFrame(contributions, columns=["Driver", "Contribution (%)"])
    st.subheader("Explainability: Why cavitation is predicted")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.barh(explain_df["Driver"], explain_df["Contribution (%)"], color="#2E86C1")
    ax.invert_yaxis()
    ax.set_xlabel("Contribution (%)")
    ax.set_ylabel("Driver")
    st.pyplot(fig, use_container_width=True)

    st.subheader("What to do to reduce cavitation risk")
    for idx, rec in enumerate(recommendations, start=1):
        st.markdown(f"{idx}. {rec}")

    st.subheader("What to expect")
    st.info(expectation)

    st.subheader("Key extracted features")
    display = {
        "cavitation_indicator": all_features.get("cavitation_indicator", 0.0),
        "energy_ratio_high": all_features.get("energy_ratio_high", 0.0),
        "frequency_entropy": all_features.get("frequency_entropy", 0.0),
        "crest_factor": all_features.get("crest_factor", 0.0),
        "kurtosis": all_features.get("kurtosis", 0.0),
        "snr_db": all_features.get("snr_db", 0.0),
    }
    st.dataframe(pd.DataFrame([display]).T.rename(columns={0: "Value"}))


def _running_in_streamlit() -> bool:
    """Check whether script is currently running as a Streamlit app."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def launch_dashboard() -> None:
    """Launch Streamlit dashboard from plain Python execution."""
    script_path = Path(__file__).resolve()
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(script_path)], check=True)


def run_cli_preview() -> None:
    """CLI fallback preview for non-UI usage."""
    signal = _simulate_signal(
        marine_condition="moderate",
        cavitation_severity="moderate",
        include_engine_load=True,
        sample_rate=10000,
        duration=2.0,
        shaft_freq=1750,
    )
    generator = MarinePumpVibrationDataGenerator(sample_rate=10000, duration=2.0, shaft_freq=1750)
    extractor = FrequencyFeatureExtractor(generator)
    features = extractor.extract_frequency_features(signal)
    print("Generated sample signal and extracted frequency features.")
    print(f"Cavitation indicator: {features.get('cavitation_indicator', 0.0):.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Marine pump cavitation executable")
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Run a quick CLI preview instead of launching the dashboard.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if _running_in_streamlit():
        render_streamlit_dashboard()
    elif args.preview:
        run_cli_preview()
    else:
        launch_dashboard()
