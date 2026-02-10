"""Tests for time-frequency feature extraction."""

import numpy as np
import pytest

from src.features.time_frequency_features import extract_time_frequency_features


def test_extract_time_frequency_features_returns_expected_keys(baseline_vibration):
    signal, generator, _ = baseline_vibration

    features = extract_time_frequency_features(signal, sample_rate=generator.sample_rate)

    expected_keys = {
        "tf_total_power",
        "tf_power_mean",
        "tf_power_std",
        "tf_spectral_entropy",
        "tf_dominant_frequency_hz",
        "tf_spectral_flux_mean",
        "tf_bandwidth_mean_hz",
        "tf_low_band_ratio",
        "tf_mid_band_ratio",
        "tf_high_band_ratio",
    }

    assert set(features.keys()) == expected_keys
    assert all(np.isfinite(v) for v in features.values())


def test_time_frequency_features_detect_dominant_frequency():
    sample_rate = 4000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    f0 = 200.0
    signal = np.sin(2 * np.pi * f0 * t)

    features = extract_time_frequency_features(signal, sample_rate=sample_rate)

    assert abs(features["tf_dominant_frequency_hz"] - f0) < 20.0


def test_time_frequency_features_invalid_input_raises():
    with pytest.raises(ValueError):
        extract_time_frequency_features(np.array([0.1, 0.2]), sample_rate=1000)
