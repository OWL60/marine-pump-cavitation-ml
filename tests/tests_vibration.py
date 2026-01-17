"""
Tests for vibration generator engine
"""
import numpy as np
import pytest

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


def test_generate_dataset_small(baseline_vibration) -> None:
    """
    Perfom test for small dataset
    """
    _, generator_obj, _ = baseline_vibration
    X_raw, X_features, y, metadata = generator_obj.generate_dataset(n_samples =20, include_marine_conditions=True, save_to_disk=True)

    assert X_raw.shape[0] == 20
    assert X_features.shape[0] == 20
    assert len(y) == 20
    assert len(metadata) == 20
    assert np.sum(y==0) == 10
    assert np.sum(y==1) == 10

    for meta in metadata:
        assert 'sample_id' in meta
        assert 'rpm' in meta
        assert 'signal_peak' in meta
        assert 'signal_rms' in meta
        assert 'marine_condition' in meta


@pytest.mark.slow
def test_generate_dataset_large(baseline_vibration):
    """
    Test generate dataset function for large dataset
    """
    _, generator_obj, _ = baseline_vibration
    X_raw, X_features, _, metadata = generator_obj.generate_dataset(n_samples=1000, include_marine_conditions=True, save_to_disk=False)
    
    assert X_raw.shape == (1000, 100000)
    assert X_features.shape[0] == 1000
    assert X_features.shape[1] > 5

    rpm = [meta['rpm'] for meta in metadata]
    assert len(set(rpm)) > 1
    conditions = [meta['marine_condition'] for meta in metadata]
    assert len(set(conditions)) > 1


def test_generate_dataset_land_only(baseline_vibration):
    """
    test generation of dataset for land
    """
    _, generator_obj, _ = baseline_vibration
    _, _, _, metadata = generator_obj.generate_dataset(n_samples=10, include_marine_conditions=False, save_to_disk=False)
    for meta in metadata:
        assert meta['marine_condition'] == 'land'
        assert not meta['is_marine']


def test_error_handling(baseline_vibration):
    """
    Test for error handling in case of invalid argument passed to vibration generator
    """
    _, generator_obj, _ = baseline_vibration
    with pytest.raises(Exception):
        generator_obj.generate_vibration_signal(rpm=-1000, duration=1.0)
    with pytest.raises(Exception):
        generator_obj.generate_vibration_signal(rpm=1750, duration=0)
    with pytest.raises(Exception):
        generator_obj.generate_vibration_signal(rpm=1750, duration=-1.0)


def test_generate_marine_scenarios(baseline_vibration):
    """
    Test for generate marine scenario method
    """
    scenarios = ['mild', 'moderate', 'severe']
    normal_vibaration_signal, generator_obj, _ = baseline_vibration
    expected_scenarios = generator_obj.generate_marine_scenarios(normal_vibaration_signal)
    
    for scenario in expected_scenarios:
        assert scenario in scenarios
        assert isinstance(expected_scenarios[scenario], np.ndarray)
        assert len(expected_scenarios[scenario]) == len(normal_vibaration_signal)

        diff_scenarios_signals = list(expected_scenarios.values())
        for index, back_scenario_signal in enumerate(diff_scenarios_signals):
            for scenario_signal in enumerate(diff_scenarios_signals, start=index+1):
                assert not np.array_equal(back_scenario_signal, scenario_signal)