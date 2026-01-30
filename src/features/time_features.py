"""
Time-domain feature extraction for pump vibration signal
"""

import warnings
from typing import Dict, List, Union

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

from utils import log

warnings.filterwarnings("ignore", category=RuntimeWarning)
EPS = np.finfo(np.float64).eps


def extract_time_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extract comprehensive time-domain features extraction for pump vibration signal
    """
    signal = np.asarray(signal, dtype=np.float64)
    signal -= np.mean(signal)
    cleaning_mask = np.isfinite(signal)
    signal = signal[cleaning_mask]
    abs_signal = np.abs(signal)
    if len(signal) < 10:
        log.log_error(f"Signal too short: {len(signal)} samples. Need at least 10")
        raise ValueError(f"Signal too short: {len(signal)} samples. Need at least 10")

    features = {}
    features["mean"] = np.mean(signal)
    features["std"] = np.std(signal)
    features["variance"] = np.var(signal)
    features["peak"] = np.max(abs_signal)
    features["peak_to_peak"] = np.ptp(signal)
    features["rms"] = np.sqrt(np.mean(signal**2))
    features["mean_abs"] = np.mean(abs_signal)

    if features["rms"] > EPS:
        features["crest_factor"] = features["peak"] / features["rms"] + EPS
    else:
        features["crest_factor"] = 0.0

    if features["mean_abs"] > EPS:
        features["shape_factor"] = features["rms"] / features["mean_abs"] + EPS
    else:
        features["shape_factor"] = 0.0

    if features["mean_abs"] > EPS:
        features["impulsive_factor"] = features["peak"] / features["mean_abs"] + EPS
    else:
        features["impulsive_factor"] = 0.0

    # Higher order statistics(HOS), Skewness and kurtosis of signal

    if features["std"] > 0 and len(signal) >= 3:
        features["skewness"] = stats.skew(signal)
    else:
        features["skewness"] = 0.0

    if features["std"] > 0 and len(signal) >= 4:
        features["kurtosis"] = stats.kurtosis(signal)
    else:
        features["kurtosis"] = 0.0

    features["energy"] = np.sum(signal**2)
    features["power"] = features["energy"] / len(signal)

    # Zero crossing.
    zero_crossing = np.where(np.diff(np.sign(signal)))[0]
    features["zero_crossing_rate"] = len(zero_crossing) / (len(signal) + EPS)

    # slope sign change
    slope_changes = np.sum(np.diff(np.sign(np.diff(signal))) != 0)
    features["slope_changes"] = slope_changes / (len(signal) + EPS)

    if features["rms"] > EPS:
        features["peak_to_rms"] = features["peak"] / features["rms"] + EPS
    else:
        features["peak_to_rms"] = 0.0

    sorted_signal = sorted(abs_signal)
    if len(sorted_signal) >= 10:
        top_10_percent = sorted_signal[-len(sorted_signal) // 10 :]
        features["spike_index"] = (
            np.mean(top_10_percent) / features["rms"] if features["rms"] > EPS else 0.0
        )
    else:
        features["spike_index"] = 0.0

    # Median Absolute Deviation, MAD (robust to outliers)
    features["MAD"] = np.median(np.abs(signal - np.median(signal)))
    # Interquartile Range, IQR (robust spread measure)
    q75, q25 = np.percentile(signal, [75, 25])
    features["IQR"] = q75 - q25
    # Coefficient of Variation, CoV
    if np.abs(features["mean"]) > EPS:
        features["CoV"] = features["std"] / np.abs(features["mean"]) + EPS
    else:
        features["CoV"] = 0.0
    return features


def batch_extract(
    signal: Union[List[np.ndarray], np.ndarray], verbose: bool = True
) -> np.ndarray:
    """
    Extract features from different signals
    """
    if isinstance(signal, np.ndarray):
        if signal.ndim == 1:
            signal = [signal]
        elif signal.ndim == 2:
            signal = list(signal)

    n_signal = len(signal)
    if verbose:
        log.log_info(f"Extracting features from {n_signal} signals...")

    all_features = []
    feature_names = None

    for i, sig in enumerate(signal):
        if verbose and i % 100 == 0 and i > 0:
            log.log_success(f"Processed {i}/{n_signal} signals...")

        try:
            features = extract_time_features(sig)
            if feature_names is None:
                feature_names = list(features.keys())
            feature_vector = [features[name] for name in feature_names]
            all_features.append(feature_vector)

        except Exception as e:
            if verbose:
                log.log_error(f"Error processing signal {i}: {str(e)}")
            if feature_names is not None:
                all_features.append([0.0] * len(feature_names))
    if verbose:
        log.log_info("Feature extraction complete!")
    if feature_names:
        log.log_info(f"Extracted {len(feature_names)} features per signal")
        log.log_info(
            f"Feature matrix shape: ({len(all_features)}, {len(feature_names)})"
        )
    return np.array(all_features)


def get_feature_names() -> List[str]:
    """
    get all feature names in consistent order.
    """
    # create dummy signal to get feature names
    dummy_signal = np.random.randn(100)
    feature_names = list(extract_time_features(dummy_signal).keys())
    return feature_names


def normalize_features(features: np.ndarray, method: str) -> np.ndarray:
    """
    Normalize feature matrix.
    """
    if method == "standard":
        mean = np.mean(features, axis=0)
        std = np.std(features, axis=0)
        std[std == 0] = 1
        return (features - mean) / std

    if method == "minmax":
        min_values = np.min(features, axis=0)
        max_values = np.max(features, axis=0)
        range_values = max_values - min_values
        range_values[range_values == 0] = 1
        return (features - min_values) / range_values

    if method == "robust":
        median = np.median(features, axis=0)
        q75 = np.percentile(features, 75, axis=0)
        q25 = np.percentile(features, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr == 0] = 1
        return (features - median) / iqr

    log.log_debug(f"Unknown normalization method: {method}")
    raise ValueError(f"Unknown normalization method: {method}")


def feature_important_analysis(
    features: np.ndarray, label: np.ndarray, feature_names: List[str] = None
) -> Dict:
    """
    Extract the important features.
    """
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(features.shape[1])]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, label)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]

    result = {"sorted_features": [], "sorted_importances": [], "top_features": []}

    log.log_info("Feature Importance Analysis:")
    log.log_info("=" * 50)

    for i, idx in enumerate(indices[:10]):  # Show top 10
        result["sorted_features"].append(feature_names[idx])
        result["sorted_importances"].append(importances[idx])
        result["top_features"].append((feature_names[idx], importances[idx]))
        log.log_info(f"{i+1:2d}. {feature_names[idx]:<20} : {importances[idx]:.4f}")
    return result
