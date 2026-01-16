"""
Test for time-domain features
"""
import pytest
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


def test_batch_extract_with_empty_list():
     """
     Test test batch extract method with empty list
     """
     feature_matrix = time_features.batch_extract([], verbose=False)
     assert isinstance(feature_matrix, np.ndarray)
     assert feature_matrix.shape == (0, ) or feature_matrix.size == 0


def test_batch_extract_with_invalid_signal():
    """
    Test batch extract with invalid signal
    """
    signals = [
         np.random.randn(100),
         np.array([]),
         np.array([1.0]),
         np.random.randn(50),

    ]
    features_matrix = time_features.batch_extract(signals, verbose=False)
    assert isinstance(features_matrix, np.ndarray)
    assert features_matrix.shape[0] == len(signals)


def test_get_feature_names():
    """
    Test for get_feature_names method.
    """
    feature_names = time_features.get_feature_names()

    assert isinstance(feature_names, list)
    assert len(feature_names) > 5
    expected_names = ['mean', 'std', 'variance', 'peak', 'rms', 'crest_factor','shape_factor']

    for expected_name in expected_names:
        assert expected_name in feature_names

    for feature_name in feature_names:
        assert isinstance(feature_name, str)


def test_normalize_features():
    """
    Test standard normalization method
    """
    np.random.seed(42)
    features = np.random.randn(100, 10) + 5 * 10
    normalized = time_features.normalize_features(features, 'standard')

    assert isinstance(normalized, np.ndarray)
    assert normalized.shape == features.shape

    col_means = np.mean(normalized, axis=0) 
    assert np.all(np.abs(col_means) < 1e-10)

    col_stds = np.std(normalized, axis=0)
    assert np.all(np.abs(col_stds - 1) < 1e-10)


def test_normalize_features_minmax():
    """
    Test minmax normalization
    """
    np.random.seed(42)
    features = np.random.randn(100, 5) + 10
    normalized = time_features.normalize_features(features, 'minmax')

    assert np.all(normalized) >= 0
    assert np.all(normalized) <= 1

    column_mins = np.min(normalized, axis=0)
    column_maxs = np.max(normalized, axis=0)
    assert np.all(np.abs(column_mins) < 1e-10)
    assert np.all(np.abs(column_maxs- 1) < 1e-10)


def test_normalize_features_robust():
    """
    Test robust normalization
    """
    np.random.seed(42)
    features = np.random.randn(100, 10)
    normalized = time_features.normalize_features(features, 'robust')
    column_median = np.median(normalized, axis=0)

    assert np.all(np.abs(column_median) < 1e-10)


def test_normalize_features_invalid_method():
    """
    Test normalize features for invalid method
    """
    np.random.seed(42)
    features = np.random.randn(10, 5)
    with pytest.raises(ValueError) as e:
        time_features.normalize_features(features, 'ivalid_method')
    assert "Unknown normalization method: " in str(e.value)


def test_feature_important_analysis():
    """
    Test feature_important_analysis method
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    feature_names = [f'features_{i}' for i in range(n_features)]
    results = time_features.feature_important_analysis(X, y, feature_names)

    assert 'sorted_features' in results
    assert 'sorted_importances' in results
    assert 'top_features' in results

    assert len(results['sorted_features']) > 0
    assert len(results['sorted_importances']) > 0
        
    # First feature should be in top features
    assert 'features_0' in results['sorted_features'] or 'features_1' in results['sorted_features']


def test_features_important_analysis_no_names():
    """
    Test features importance without feauture names
    """
    np.random.seed(42)
    X = np.random.randn(50, 5)
    y = np.random.randint(0, 2, 50)
    results = time_features.feature_important_analysis(X, y)

    assert 'sorted_features' in results
    assert len(results['sorted_features']) == 5

@pytest.mark.parametrize('signal_length', [100, 1000, 10000])
def test_extract_features_with_different_length(signal_length):
    """
    Test features extraction with different length
    """
    np.random.seed(42)
    signals = np.random.randn(signal_length)
    features = time_features.extract_time_features(signals)
    assert isinstance(features, dict)
    assert 'mean' in features
    assert 'std' in features