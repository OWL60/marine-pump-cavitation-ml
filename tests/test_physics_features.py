"""Tests for physics-informed feature extraction."""

import numpy as np
import pytest

from src.features.physics_features import extract_physics_features


COMMON_PARAMS = {
    "sample_rate": 10000,
    "shaft_rpm": 1750,
    "suction_pressure_pa": 200000.0,
    "vapor_pressure_pa": 3000.0,
    "fluid_density_kg_m3": 1025.0,
    "flow_velocity_m_s": 4.0,
    "impeller_diameter_m": 0.25,
}


def test_extract_physics_features_returns_expected_keys(baseline_vibration):
    signal, _, _ = baseline_vibration
    features = extract_physics_features(signal, **COMMON_PARAMS)

    expected_keys = {
        "shaft_frequency_hz",
        "tip_speed_m_s",
        "cavitation_number",
        "dominant_frequency_hz",
        "strouhal_number",
        "reynolds_number",
        "high_frequency_ratio",
        "harmonic_power_ratio",
        "vibration_intensity",
        "cavitation_risk_index",
    }

    assert set(features.keys()) == expected_keys
    assert all(np.isfinite(v) for v in features.values())
    assert features["cavitation_number"] > 0


def test_cavitation_risk_index_increases_for_cavitation_signal(cavitation_vibration):
    cav_signal, baseline_signal, _, _ = cavitation_vibration

    base_features = extract_physics_features(baseline_signal, **COMMON_PARAMS)
    cav_features = extract_physics_features(cav_signal, **COMMON_PARAMS)

    assert cav_features["high_frequency_ratio"] >= base_features["high_frequency_ratio"]
    assert cav_features["cavitation_risk_index"] >= base_features["cavitation_risk_index"]


def test_extract_physics_features_invalid_input_raises():
    with pytest.raises(ValueError):
        extract_physics_features(np.array([0.1, 0.2]), **COMMON_PARAMS)

    with pytest.raises(ValueError):
        invalid = dict(COMMON_PARAMS)
        invalid["shaft_rpm"] = 0
        extract_physics_features(np.random.randn(1000), **invalid)
