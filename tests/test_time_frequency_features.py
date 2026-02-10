import numpy as np
import pytest

from src.features.time_frequency_features import extract_time_frequency_features


def test_extract_time_frequency_features_basic(baseline_vibration):
    signal, generator, _ = baseline_vibration
    feats = extract_time_frequency_features(signal, sample_rate=generator.sample_rate)

    assert "tf_total_energy" in feats
    assert feats["tf_total_energy"] > 0
    assert 0 <= feats["tf_high_band_ratio"] <= 1
    assert feats["tf_time_bins"] > 0


def test_extract_time_frequency_invalid_input():
    with pytest.raises(ValueError):
        extract_time_frequency_features(np.ones((10, 2)), sample_rate=10000)
