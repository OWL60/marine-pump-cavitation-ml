"""Test frequency feature extraction functions."""

from typing import Any, Dict, List

import numpy as np
import pytest


def test_extract_frequency_features_basic(
    baseline_vibration, frequency_feature_extractor
):
    """Test basic functionality of extract_frequency_features"""

    norm_vibration, _, _ = baseline_vibration
    extractor = frequency_feature_extractor

    # Extract frequency features
    features: Dict[str, float] = extractor.extract_frequency_features(norm_vibration)

    assert isinstance(features, dict)

    required_features: List[str] = [
        "peak_frequency_hz",
        "total_power",
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
    ]
    for feature in required_features:
        assert feature in features
        assert isinstance(features[feature], (int, float, np.float64, np.int64))
        assert features[feature] >= 0, f"{feature} should be non-negative"
        assert not np.isnan(features[feature]), f"{feature} should not be NaN"
    assert (features["peak_frequency_hz"] - 29.3) < np.finfo(
        np.float64
    ).eps  # shaft frequency = 1750 rpm


def test_extract_frequency_features_cav(
    cavitation_vibration, frequency_feature_extractor
):
    """Test for extract features from the cavitation signals"""
    cav_vibration, normal_vibration, _, _ = cavitation_vibration
    extractor = frequency_feature_extractor

    cav_features: Dict[str, float] = extractor.extract_frequency_features(cav_vibration)
    norm_features: Dict[str, float] = extractor.extract_frequency_features(
        normal_vibration
    )

    assert cav_features["cavitation_indicator"] > norm_features["cavitation_indicator"]
    assert (
        abs(
            cav_features["spectral_centroid_hz"] - norm_features["spectral_centroid_hz"]
        )
        > 10
    )


def test_extract_frequency_specific_features(
    baseline_vibration, frequency_feature_extractor
):
    """Test frequency specific features"""
    normal_vibration, _, _ = baseline_vibration
    extractor = frequency_feature_extractor
    specific_features: List[str] = [
        "peak_frequency_hz",
        "peak_amplitude",
        "total_power",
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_skewness",
        "spectral_kurtosis",
        "rms_frequency_hz",
        "frequency_entropy",
    ]
    specific_results: Dict[str, float] = extractor.extract_frequency_features(
        normal_vibration, specific_features
    )
    assert len(specific_results) == len(specific_features)
    for specific_result in specific_results:
        assert specific_result in specific_features


def test_batch_extract_frequency_features(
    cavitation_vibration, frequency_feature_extractor
):
    """Test batch extract method"""
    cav_signal, baseline_signal, _, _ = cavitation_vibration
    extractor = frequency_feature_extractor

    batch_signal = [baseline_signal, cav_signal, baseline_signal * 2, cav_signal]
    features_matrix: np.ndarray = extractor.batch_extract_frequency_features(
        batch_signal, verbose=False
    )

    assert isinstance(features_matrix, np.ndarray)
    assert features_matrix.shape[0] == len(batch_signal)
    assert features_matrix.shape[1] > 5

    assert not np.any(np.isinf(features_matrix))
    assert not np.any(np.isnan(features_matrix))

    assert (
        features_matrix.shape[0] == 4
    )  # label, ensure all samples are perfectly processed.


def test_normalize_features(frequency_feature_extractor):
    """Test batch extract method"""
    extractor = frequency_feature_extractor
    np.random.seed(42)
    features_matrix: np.ndarray = np.random.randn(100, 5)
    methods: List[str] = ["standard", "minmax", "robust", "unit", "log"]

    for method in methods:
        norm_features: np.ndarray = extractor.normalize_features(
            features_matrix, method
        )

        assert norm_features.shape == features_matrix.shape
        assert not np.any(np.isnan(features_matrix))
        assert not np.any(np.isnan(features_matrix))

        if method is "standard":
            means: np.ndarray = np.abs(np.mean(norm_features, axis=0))
            stds: np.ndarray = np.abs(np.std(norm_features, axis=0))
            assert np.allclose(means, 0, atol=1e-10)
            assert np.allclose(stds, 1, atol=1e-10)
        elif method is "minmax":
            assert np.all(norm_features >= 0) and np.all(norm_features <= 1)
        elif method is "robust":
            medians: np.ndarray = np.abs(np.median(norm_features))
            assert np.allclose(medians, 0, atol=1e-10)
        elif method is "unit":
            norms: np.ndarray = np.linalg.norm(norm_features, axis=0)
            assert np.allclose(norms, 1, atol=1e-10)
        elif method is "log":
            assert np.all(norm_features > 0)


def test_normalize_features_with_names(frequency_feature_extractor):
    """Test batch extract method"""
    extractor = frequency_feature_extractor

    features_matrix: np.ndarray = np.array(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float
    )
    norm_features: np.ndarray = extractor.normalize_features(
        features_matrix, method="standard"
    )

    assert norm_features.shape == features_matrix.shape

    means: np.ndarray = np.abs(np.mean(norm_features, axis=0))
    assert np.allclose(means, 0, atol=1e-10)

    large_feat_matrix: np.ndarray = features_matrix * 1e6
    large_norm_features: np.ndarray = extractor.normalize_features(
        large_feat_matrix, method="standard"
    )

    assert large_norm_features.shape == features_matrix.shape


def test_feature_importance_analysis_class(frequency_feature_extractor):
    """Test for feature importance analysis"""
    extractor = frequency_feature_extractor
    np.random.seed(42)
    samples_x: np.ndarray = np.random.randn(200, 10)
    labels_y: np.ndarray = (
        samples_x[:, 0] + 0.5 * samples_x[:, 1] - 0.3 * samples_x[:, 2] > 0
    ).astype(int)
    methods: List[str] = [
        "f-score",
        "mutual_info",
        "random_forest",
        "correlation",
        "variance",
    ]

    for method in methods:
        results: Dict[str, Any] = extractor.feature_importance_analysis(
            samples_x, labels_y, task="classification", method=method
        )

        assert "importance_scores" in results
        assert "ranked_features" in results
        assert "ranked_scores" in results

        ranked_features = results["ranked_features"]
        top_5 = ranked_features[:5]
        selected_features = [
            "feature_0",
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
        ]

        important_top: int = sum(1 for feat in selected_features if feat in top_5)
        assert important_top >= 2


def test_analyze_feature_importance_regression(frequency_feature_extractor):
    """Test feature importance for regression"""
    extractor = frequency_feature_extractor

    np.random.seed(42)
    samples_x = np.random.randn(200, 10)
    labels_y = (
        2 * samples_x[:, 0]
        + 1.5 * samples_x[:, 1]
        + samples_x[:, 2]
        + np.random.randn(200) * 0.5
    )

    feature_names = [f"feature_{i}" for i in range(10)]

    results: Dict[str, Any] = extractor.feature_importance_analysis(
        samples_x, labels_y, feature_names, task="regression", method="f-score"
    )

    assert len(results["importance_scores"]) == 10
    assert len(results["ranked_features"]) == 10

    # Features 0, 1, 2 should be top ranked
    ranked_features = results["ranked_features"]
    assert "feature_0" in ranked_features[:3]


def test_error_handling(frequency_feature_extractor):
    """
    Test for error handling in case of invalid method passed
    """
    np.random.seed(42)
    samples_x: np.ndarray = np.random.randn(200, 10)
    labels_y: np.ndarray = (
        samples_x[:, 0] + 0.5 * samples_x[:, 1] - 0.3 * samples_x[:, 2] > 0
    ).astype(int)

    with pytest.raises(Exception):
        frequency_feature_extractor(samples_x, labels_y, method="ivalid-method")
