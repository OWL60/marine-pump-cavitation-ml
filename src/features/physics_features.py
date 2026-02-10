"""Physics-informed feature extraction for marine pump cavitation analysis."""

from typing import Dict

import numpy as np


EPS = np.finfo(np.float64).eps


def extract_physics_features(
    signal: np.ndarray,
    sample_rate: int,
    shaft_rpm: float,
    suction_pressure_pa: float,
    vapor_pressure_pa: float,
    fluid_density_kg_m3: float,
    flow_velocity_m_s: float,
    impeller_diameter_m: float,
    kinematic_viscosity_m2_s: float = 1.0e-6,
) -> Dict[str, float]:
    """Extract physics-informed features from vibration signal and operating point."""
    x = np.asarray(signal, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size < 16:
        raise ValueError("Signal too short for physics feature extraction (min 16 samples).")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    if shaft_rpm <= 0:
        raise ValueError("shaft_rpm must be positive.")
    if fluid_density_kg_m3 <= 0 or flow_velocity_m_s <= 0 or impeller_diameter_m <= 0:
        raise ValueError("Fluid density, flow velocity, and impeller diameter must be positive.")

    x = x - np.mean(x)

    shaft_freq_hz = shaft_rpm / 60.0
    tip_speed_m_s = np.pi * impeller_diameter_m * shaft_freq_hz

    cavitation_number = (suction_pressure_pa - vapor_pressure_pa) / (
        0.5 * fluid_density_kg_m3 * flow_velocity_m_s**2 + EPS
    )

    spectrum = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    power = np.abs(spectrum) ** 2
    total_power = np.sum(power) + EPS

    dominant_frequency_hz = float(freqs[np.argmax(power)])
    strouhal_number = dominant_frequency_hz * impeller_diameter_m / (flow_velocity_m_s + EPS)
    reynolds_number = (
        flow_velocity_m_s * impeller_diameter_m / (kinematic_viscosity_m2_s + EPS)
    )

    hf_threshold = 10.0 * shaft_freq_hz
    hf_mask = freqs >= hf_threshold
    high_frequency_ratio = float(np.sum(power[hf_mask]) / total_power)

    harmonic_powers = []
    for harmonic in range(1, 6):
        target_freq = harmonic * shaft_freq_hz
        idx = int(np.argmin(np.abs(freqs - target_freq)))
        harmonic_powers.append(power[idx])

    harmonic_power_ratio = float(np.sum(harmonic_powers) / total_power)

    rms = float(np.sqrt(np.mean(x**2)))
    vibration_intensity = rms / (tip_speed_m_s + EPS)

    cavitation_risk_index = (
        high_frequency_ratio * (1.0 / (max(cavitation_number, EPS))) * (1.0 + vibration_intensity)
    )

    return {
        "shaft_frequency_hz": float(shaft_freq_hz),
        "tip_speed_m_s": float(tip_speed_m_s),
        "cavitation_number": float(cavitation_number),
        "dominant_frequency_hz": dominant_frequency_hz,
        "strouhal_number": float(strouhal_number),
        "reynolds_number": float(reynolds_number),
        "high_frequency_ratio": high_frequency_ratio,
        "harmonic_power_ratio": harmonic_power_ratio,
        "vibration_intensity": float(vibration_intensity),
        "cavitation_risk_index": float(cavitation_risk_index),
    }
