"""
Test for time-domain features
"""
import numpy as np
from src.features import time_features

def test_extract_time_features(baseline_vibration):
    """
    Test extract_time_feature method
    """
    baseline_signal, _, _ = baseline_vibration
    features = time_features.extract_time_features(baseline_signal)
    
    assert isinstance(features, dict)
    assert len(features) > 5

    required_features = ['mean', 'std', 'rms', 'peak', 'crest_factor']
    for required_feature in required_features:
        assert required_feature in features 
        assert isinstance(features[required_feature], (int, float, np.number))

    assert np.abs(features['mean']) < 0.1  
    assert features['std'] > 0  
    assert features['rms'] > 0  
    assert features['peak'] > 0


def test_extract_time_features_cavitation(cavitation_vibration):
    """
    Test extract time-domain features from cavitation vibration method.
    """
    cav_signal, baseline_signal, _, _ = cavitation_vibration
    baseline_features = time_features.extract_time_features(baseline_signal)
    cav_features = time_features.extract_time_features(cav_signal)

    assert not np.allclose(list(baseline_features.values()), list(cav_features.values()), rtol=0.1)
    assert cav_features['peak'] > baseline_features['peak']
    assert cav_features['crest_factor'] > baseline_features['crest_factor']


def test_extract_time_features_edge_cases():
        """
        Test feature extraction with edge cases
        """
        # len(short_signal) < 10, will rise the exception.
        short_signal = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 7.0])
        features = time_features.extract_time_features(short_signal)
        assert isinstance(features, dict)
        
        constant_signal = np.ones(100)
        features = time_features.extract_time_features(constant_signal)
        assert features['std'] == 0
        assert features['crest_factor'] == 1.0
        
        zero_signal = np.zeros(100)
        features = time_features.extract_time_features(zero_signal)
        assert features['mean'] == 0
        assert features['rms'] == 0


def test_batch_extract(cavitation_vibration):
    """
    Test batch extract method
    """
    cav_signal, baseline_signal, _, _ = cavitation_vibration
    batch_signal = [baseline_signal, cav_signal, baseline_signal * 2, cav_signal]
    features_matrix = time_features.batch_extract(batch_signal, verbose=False)

    assert isinstance(features_matrix, np.ndarray)
    assert features_matrix.shape[0] == len(batch_signal)
    assert features_matrix.shape[1] > 5

    assert not np.any(np.isinf(features_matrix))
    assert not np.any(np.isnan(features_matrix))

    assert features_matrix.shape[0] == 4 # label, ensure all samples are perfectly processed.