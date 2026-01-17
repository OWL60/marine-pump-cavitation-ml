"""
Tests for vibration generator engine
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import log
from src.data.generator import MarinePumpVibrationDataGenerator
from src.features import time_features

def test_add_cavitation_effect(baseline_vibration) -> None:
    """
    Test add_cavitation_effect method.
    """
    baseline_signal, generator_obj, baseline_stats = baseline_vibration

    severities = ['mild', 'moderate', 'severe']
    for severity in severities:
        cav_signal = generator_obj.add_cavitation_effect(baseline_signal, sample_rate=100000, severity=severity)
    
        rms = np.sqrt(np.mean(cav_signal**2))

        assert isinstance(cav_signal, np.ndarray), "cavitation signal is not of np.ndarray"
        assert len(cav_signal) == len(baseline_signal), "cavitation signal is not as the normal vibration" 
        assert rms > baseline_stats['rms'] * 0.8, "RMS is too high"


def test_add_ship_motion(baseline_vibration) -> None:
    """ 
    Test ship motion method
    """
    baseline_signal, generator_obj, _ = baseline_vibration
    conditions = ['mild', 'moderate', 'severe']

    for condition in conditions:
        marine_signal = generator_obj.add_ship_motion(baseline_signal, marine_condition=condition, include_engine_load=False)

        assert isinstance(marine_signal, np.ndarray), "signals is not of np.ndarray"
        assert len(marine_signal) == len(baseline_signal), "marine signal is not as the normal vibration" 

        assert not np.array_equal(marine_signal, baseline_signal), "marine signal has different shape as normal vibration"

        modulation = np.std(marine_signal) / np.std(baseline_signal)
        assert 0.5 < modulation < 2.0


def test_compare_signals(baseline_vibration) -> None:
    """
    Test compare_signals
    """
    baseline_signal, generator_obj, _ = baseline_vibration
    cav_signal = generator_obj.add_cavitation_effect(baseline_signal, sample_rate=100000, severity='moderate')
    stats = generator_obj.compare_signals(baseline_signal, cav_signal)

    for _, value in stats.items():
        assert isinstance(value, (int, float, np.number))

    assert stats['cavitation_rms'] > stats['normal_rms'], 'cavitation rms is low'
    assert stats['rms_increase'] > 1.0, 'rms increase is too low'
    assert np.abs(stats['cavitation_kurtosis'] - stats['normal_kurtosis']) < 1e-6


def test_compare_land_v_marine(baseline_vibration) -> None:
    """
    Test compare_land_v_marine method
    """
    pass