"""Physics-informed feature utilities for marine pump cavitation."""

from __future__ import annotations

from typing import Dict

import numpy as np

EPS = np.finfo(np.float64).eps


def calculate_cavitation_number(
    suction_pressure_pa: float,
    vapor_pressure_pa: float,
    fluid_density_kg_m3: float,
    flow_velocity_m_s: float,
) -> float:
    """Calculate cavitation number sigma."""
    dynamic_pressure = 0.5 * fluid_density_kg_m3 * (flow_velocity_m_s**2)
    if dynamic_pressure <= 0:
        raise ValueError("dynamic pressure must be positive")
    return float((suction_pressure_pa - vapor_pressure_pa) / (dynamic_pressure + EPS))


def calculate_npsh_margin(
    npsh_available_m: float,
    npsh_required_m: float,
) -> float:
    """Compute net positive suction head margin."""
    return float(npsh_available_m - npsh_required_m)


def estimate_reynolds_number(
    fluid_density_kg_m3: float,
    flow_velocity_m_s: float,
    hydraulic_diameter_m: float,
    dynamic_viscosity_pa_s: float,
) -> float:
    """Estimate Reynolds number for impeller inlet flow."""
    if dynamic_viscosity_pa_s <= 0:
        raise ValueError("dynamic viscosity must be positive")
    return float(
        fluid_density_kg_m3
        * flow_velocity_m_s
        * hydraulic_diameter_m
        / (dynamic_viscosity_pa_s + EPS)
    )


def energy_ratio_high_low(
    signal: np.ndarray,
    sample_rate: float,
    low_cut_hz: float = 100.0,
    high_cut_hz: float = 1000.0,
) -> float:
    """Compute high/low frequency energy ratio from FFT power spectrum."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be 1D")
    if len(x) < 16:
        raise ValueError("signal too short")
    x = x - np.mean(x)

    fft_vals = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sample_rate)
    power = np.abs(fft_vals) ** 2

    low_mask = (freqs >= 0.0) & (freqs <= low_cut_hz)
    high_mask = freqs >= high_cut_hz

    low_energy = float(np.sum(power[low_mask])) if np.any(low_mask) else 0.0
    high_energy = float(np.sum(power[high_mask])) if np.any(high_mask) else 0.0

    return float(high_energy / (low_energy + EPS))


def extract_physics_features(
    signal: np.ndarray,
    sample_rate: float,
    suction_pressure_pa: float,
    vapor_pressure_pa: float,
    fluid_density_kg_m3: float,
    flow_velocity_m_s: float,
    hydraulic_diameter_m: float,
    dynamic_viscosity_pa_s: float,
    npsh_available_m: float,
    npsh_required_m: float,
) -> Dict[str, float]:
    """Extract a compact set of physics-informed features."""
    cavitation_number = calculate_cavitation_number(
        suction_pressure_pa,
        vapor_pressure_pa,
        fluid_density_kg_m3,
        flow_velocity_m_s,
    )
    return {
        "physics_cavitation_number": cavitation_number,
        "physics_npsh_margin": calculate_npsh_margin(npsh_available_m, npsh_required_m),
        "physics_reynolds_number": estimate_reynolds_number(
            fluid_density_kg_m3,
            flow_velocity_m_s,
            hydraulic_diameter_m,
            dynamic_viscosity_pa_s,
        ),
        "physics_energy_ratio_high_low": energy_ratio_high_low(signal, sample_rate),
    }
