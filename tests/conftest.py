"""
configuration file for shared fixtures.
"""
import numpy as np
import pytest
from scipy import stats
from src.data import MarinePumpVibrationDataGenerator

@pytest.fixture
def baseline_vibration() -> list:
    """
    Test the generate_vibration_signal method.      
    """
    generator = MarinePumpVibrationDataGenerator(sample_rate=100000)
    signal = generator.generate_vibration_signal(rpm=1750, duration=1.0)

    baseline_stats = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'max': np.max(signal),
        'min': np.min(signal),
        'rms': np.sqrt(np.mean(signal**2)),
        'peak': np.max(np.abs(signal)),
        'kurtosis': stats.kurtosis(signal, fisher=False) - 3
    }
    threshold = {
        'mean': 0.1,
        'std': 0,
        'rms': 0,
        'peak': 3,
        'kurtosis': 1e-6
    }

    assert len(signal) == int(generator.sample_rate * 1.0), "Signal length mismatch"
    assert abs(baseline_stats['mean']) < threshold['mean'], "Mean value too high"
    assert baseline_stats['std'] > threshold['std'], "Standard deviation should be positive"
    assert baseline_stats['rms'] > threshold['rms'], "RMS value should be positive"
    assert baseline_stats['peak'] < threshold['peak'] * baseline_stats['rms'], "Peak value must be no exceed 2 - 3 * rms"
    assert baseline_stats['kurtosis'] < threshold['kurtosis']

    return signal, generator, baseline_stats


@pytest.fixture(params=['mild', 'moderate', 'severe'])
@pytest.fixture(param=['mild', 'moderate', 'severe'])
def cavitation_vibration(baseline_vibration, request) -> list:
    """
    For generating cavitation vibration
    """
    baseline_signal, generator_obj, baseline_stats = baseline_vibration
    severity = request.param
    cav_signal = generator_obj.add_cavitation_effect(baseline_signal, generator_obj.sample_rate, severity)

    rms = np.sqrt(np.mean(cav_signal**2))
    assert isinstance(cav_signal, np.ndarray), "cavitation signal is not of np.ndarray"
    assert len(cav_signal) == len(baseline_signal), "cavitation signal is not as the normal vibration" 
    assert rms > baseline_stats['rms'] * 0.8, "RMS is too high"

    return cav_signal, baseline_signal, generator_obj, baseline_stats