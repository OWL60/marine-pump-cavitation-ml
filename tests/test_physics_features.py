import numpy as np

from src.features.physics_features import (
    calculate_cavitation_number,
    extract_physics_features,
)


def test_calculate_cavitation_number_positive():
    sigma = calculate_cavitation_number(
        suction_pressure_pa=180_000,
        vapor_pressure_pa=3_000,
        fluid_density_kg_m3=1025,
        flow_velocity_m_s=3.0,
    )
    assert sigma > 0


def test_extract_physics_features_smoke(baseline_vibration):
    signal, generator, _ = baseline_vibration
    features = extract_physics_features(
        signal=signal,
        sample_rate=generator.sample_rate,
        suction_pressure_pa=180_000,
        vapor_pressure_pa=3_000,
        fluid_density_kg_m3=1025,
        flow_velocity_m_s=3.0,
        hydraulic_diameter_m=0.1,
        dynamic_viscosity_pa_s=1.2e-3,
        npsh_available_m=8.5,
        npsh_required_m=6.0,
    )

    assert features["physics_reynolds_number"] > 0
    assert features["physics_npsh_margin"] == 2.5
