"""Streamlit dashboard for browser-based cavitation analysis."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib import pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.generator import MarinePumpVibrationDataGenerator
from src.features.frequency_features import FrequencyFeatureExtractor
from src.features.time_features import extract_time_features
from src.features.time_frequency_features import WaveletTimeFrequencyFeatureExtractor


st.set_page_config(page_title="Marine Pump Cavitation Dashboard", layout="wide")


@st.cache_resource
def get_generator(sample_rate: int, shaft_freq: int, duration: float):
    return MarinePumpVibrationDataGenerator(
        sample_rate=sample_rate, shaft_freq=shaft_freq, duration=duration
    )


def main() -> None:
    st.title("Marine Pump Cavitation Dashboard")
    st.caption("Browser-style UI with time, frequency, and wavelet time-frequency views")

    with st.sidebar:
        st.header("Signal Controls")
        sample_rate = st.slider("Sample Rate (Hz)", 2000, 20000, 10000, 500)
        shaft_freq = st.slider("Shaft Speed (RPM)", 600, 3600, 1750, 50)
        duration = st.slider("Duration (s)", 1.0, 5.0, 1.0, 0.5)
        severity = st.selectbox(
            "Cavitation severity", ["normal", "mild", "moderate", "severe"]
        )

    generator = get_generator(
        sample_rate=sample_rate, shaft_freq=shaft_freq, duration=duration
    )
    baseline = generator.generate_vibration_signal()
    signal = (
        baseline
        if severity == "normal"
        else generator.add_cavitation_effect(baseline, severity=severity)
    )

    time_axis = np.linspace(0, duration, len(signal), endpoint=False)

    freq_extractor = FrequencyFeatureExtractor(generator)
    wavelet_extractor = WaveletTimeFrequencyFeatureExtractor(sample_rate=sample_rate)

    freq_features = freq_extractor.extract_frequency_features(signal)
    time_features = extract_time_features(signal)
    wavelet_features = wavelet_extractor.extract_features(signal)

    tab_time, tab_freq, tab_wavelet, tab_metrics = st.tabs(
        [
            "Time Domain",
            "Frequency Domain",
            "Wavelet Time-Frequency",
            "Feature Metrics",
        ]
    )

    with tab_time:
        st.line_chart(
            pd.DataFrame({"time_s": time_axis, "amplitude": signal}).set_index("time_s")
        )

    with tab_freq:
        details = freq_extractor._compute_frequency_features(signal)
        st.line_chart(
            pd.DataFrame(
                {
                    "frequency_hz": details["frequency_spectrum_freqs"],
                    "psd": details["frequency_spectrum_psd"],
                }
            ).set_index("frequency_hz")
        )

    with tab_wavelet:
        coeffs, frequencies = wavelet_extractor._compute_cwt(signal)
        power = np.abs(coeffs) ** 2
        fig, ax = plt.subplots(figsize=(12, 4))
        im = ax.imshow(
            power,
            aspect="auto",
            origin="lower",
            extent=[time_axis.min(), time_axis.max(), frequencies.min(), frequencies.max()],
            cmap="viridis",
        )
        ax.set_title("Wavelet Scalogram (CWT)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")
        fig.colorbar(im, ax=ax, label="Power")
        st.pyplot(fig)

    with tab_metrics:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("Time Features")
            st.dataframe(pd.DataFrame([time_features]).T.rename(columns={0: "value"}))
        with col2:
            st.subheader("Frequency Features")
            st.dataframe(pd.DataFrame([freq_features]).T.rename(columns={0: "value"}))
        with col3:
            st.subheader("Wavelet Features")
            st.dataframe(pd.DataFrame([wavelet_features]).T.rename(columns={0: "value"}))


if __name__ == "__main__":
    main()
