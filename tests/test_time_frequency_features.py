"""Tests for wavelet-based time-frequency features."""

import numpy as np

from src.features.time_frequency_features import WaveletTimeFrequencyFeatureExtractor


def test_wavelet_extract_features(baseline_vibration):
    signal, generator, _ = baseline_vibration
    extractor = WaveletTimeFrequencyFeatureExtractor(sample_rate=generator.sample_rate)

    features = extractor.extract_features(signal)

    required = [
        "wavelet_total_energy",
        "wavelet_peak_frequency_hz",
        "wavelet_spectral_centroid_hz",
        "wavelet_entropy",
        "wavelet_high_low_energy_ratio",
        "wavelet_temporal_variability",
    ]
    for key in required:
        assert key in features
        assert isinstance(features[key], float)
        assert not np.isnan(features[key])
        assert features[key] >= 0


def test_wavelet_cavitation_changes_signature(cavitation_vibration):
    cav_signal, normal_signal, generator, _ = cavitation_vibration
    extractor = WaveletTimeFrequencyFeatureExtractor(sample_rate=generator.sample_rate)

    cav = extractor.extract_features(cav_signal)
    normal = extractor.extract_features(normal_signal)

    assert abs(cav["wavelet_spectral_centroid_hz"] - normal["wavelet_spectral_centroid_hz"]) > 1e-6
