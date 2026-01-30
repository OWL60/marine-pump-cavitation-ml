"""Module for computing frequency domain features from signals."""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import welch

EPS = np.finfo(np.float64).eps
__all__ = [
    "extract_frequency_features",
    "get_frequency_feature_names",
    "batch_extract_frequency_features",
    "plot_frequency_spectrum_example",
]


def _compute_frequency_features(
    signal: np.ndarray,
    nperseg: int = 1024,
    sampling_rate: float = 10000,
    shaft_freq: float = 1750,
) -> Dict[str, Union[float, np.ndarray, List[float]]]:
    """
    Compute frequency domain features from a given signal.

    Parameters:
    signal (array-like): The input signal from which to compute frequency features.
    sampling_rate (float): The sampling rate of the signal.

    Returns:
    dict: A dictionary containing frequency domain features.
    """
    if len(signal) < nperseg:
        warnings.warn(f"Signal length {len(signal)} is less than nperseg {nperseg}")
        nperseg = np.min(256, len(signal) // 2)

    if nperseg < 64:
        raise ValueError("nperseg is too small for frequency feature extraction.")

    freqs, psd = welch(signal, fs=sampling_rate, nperseg=nperseg, scaling="density")

    # Remove DC component
    psd: np.ndarray
    freqs: np.ndarray
    if freqs[0] == 0:
        psd = psd[1:]
        freqs = freqs[1:]

    total_power: float = np.trapz(psd, freqs)

    peak_frequency = freqs[np.argmax(psd)]
    peak_amplitude = np.max(psd)
    spectral_centroid = (
        np.trapz(freqs * psd, freqs) / total_power if total_power > 0 else 0.0
    )

    # spectral bandwidth (standard deviation)
    spectral_bandwidth = (
        np.sqrt(np.trapz(((freqs - spectral_centroid) ** 2) * psd, freqs) / total_power)
        if total_power > 0
        else 0.0
    )
    # spectral skewness
    spectral_skewness = (
        np.trapz(((freqs - spectral_centroid) ** 3) * psd, freqs)
        / (spectral_bandwidth**3 * total_power)
        if spectral_bandwidth > 0 and total_power > 0
        else 0.0
    )
    # spectral kurtosis
    spectral_kurtosis = (
        np.trapz(((freqs - spectral_centroid) ** 4) * psd, freqs)
        / (spectral_bandwidth**4 * total_power)
        - 3
        if spectral_bandwidth > 0 and total_power > 0
        else -3.0
    )
    shaft_freq /= 60.0
    band_definitions: Dict[float, Tuple[float, float]] = {
        "ultra_low_freq": (0, 0.5 * shaft_freq),
        "low_freq": (0.5 * shaft_freq, 2 * shaft_freq),
        "medium_freq": (2 * shaft_freq, 10 * shaft_freq),
        "high_freq": (10 * shaft_freq, 50 * shaft_freq),
        "ultra_high_freq": (50 * shaft_freq, sampling_rate / 2),
    }
    band_powers: Dict[str, float] = {}
    for band_name, (f_lower, f_upper) in band_definitions.items():
        band_mask = (freqs >= f_lower) & (freqs <= f_upper)
        if np.any(band_mask):
            band_power = np.trapz(psd[band_mask], freqs[band_mask])
            band_powers[band_name] = band_power
        else:
            band_powers[band_name] = 0.0

    band_ratios: Dict[str, float] = {}
    for req_band_name, req_band_power in band_powers.items():
        if total_power > 0:
            band_ratios[f"energy_ratio_{req_band_name}"] = req_band_power / total_power
        else:
            band_ratios[f"energy_ratio_{req_band_name}"] = 0.0

    if shaft_freq > 0:
        dominant_ratio: float = peak_frequency / shaft_freq
    else:
        dominant_ratio = 0.0

    harmonic_ratios: List[float] = []
    for i in range(1, 6):
        harmonics_freq: float = i * shaft_freq
        idx = np.argmin(np.abs(freqs - harmonics_freq))
        if idx < len(psd):
            harmonic_power = psd[idx]
            harmonic_ratios.append(
                harmonic_power / peak_amplitude if peak_amplitude > 0 else 0.0
            )
        else:
            harmonic_ratios.append(0.0)

    clean_power: float = total_power - band_powers.get("ultra_low_freq", 0.0)
    noise_power: float = band_powers.get("ultra_low_freq", 0.0)
    if noise_power > 0:
        snr_db: float = 10 * np.log10(
            clean_power / noise_power
        )  # signal-to-noise ratio in dB
    else:
        snr_db = 0.0

    cavitation_indicator: float = band_powers.get("high_freq", 0.0) + band_powers.get(
        "ultra_high_freq", 0.0
    )

    rms: float = (
        np.sqrt(np.trapz(freqs**2 * psd, freqs) / total_power)
        if total_power > 0
        else 0.0
    )
    if total_power > 0:
        norm_pd: np.ndarray = psd / total_power
        norm_pd = norm_pd[norm_pd > 0]  # avoid log(0)
        entropy_frequency: float = -np.sum(norm_pd * np.log2(norm_pd))
    else:
        entropy_frequency = 0.0

    features: Dict[str, Union[float, np.ndarray, List[float]]] = {
        "total_power": total_power,
        "peak_frequency_hz": peak_frequency,
        "peak_amplitude": peak_amplitude,
        "spectral_centroid_hz": spectral_centroid,
        "spectral_bandwidth_hz": spectral_bandwidth,
        "spectral_skewness": spectral_skewness,
        "spectral_kurtosis": spectral_kurtosis,
        "dominant_ratio": dominant_ratio,
        "snr_db": snr_db,
        "cavitation_indicator": cavitation_indicator,
        "rms_frequency_hz": rms,
        "frequency_entropy": entropy_frequency,
        "frequency_spectrum_freqs": freqs,
        "frequency_spectrum_psd": psd,
    }
    features.update(band_powers)
    features.update(band_ratios)
    for i, ratio in enumerate(harmonic_ratios, start=1):
        features[f"harmonic_ratio_{i}_ratio"] = ratio
    return features


def extract_frequency_features(
    signal: np.ndarray,
    sampling_rate: float = 10000,
    shaft_freq: float = 1750,
    nperseg: int = 1024,
    features_list: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Wrapper function to extract a specific given frequency domain features from a signal.
    Extract selected frequency features from signal
    Args:
        signal: Input vibration signal
        fs: Sampling frequency in Hz
        features_list: List of feature names to extract. If None, extracts all.

    Returns:
        Dictionary with selected features as float values
    """
    all_features: Dict[str, Union[float, np.ndarray, List[float]]] = (
        _compute_frequency_features(signal, nperseg, sampling_rate, shaft_freq)
    )
    float_features: Dict[str, float] = {}
    for key, value in all_features.items():
        if isinstance(value, (float, int, np.float64, np.int64)):
            float_features[key] = float(value)
        elif features_list and key in features_list:
            if isinstance(value, (np.ndarray, list)):
                value = np.mean(value)
                float_features[key] = float(value)

    if features_list:
        filtered_features: Dict[str, float] = {}
        for feature_name in features_list:
            if feature_name in float_features:
                filtered_features[feature_name] = float_features[feature_name]
            elif feature_name in all_features:
                feature_value = all_features[feature_name]
                if isinstance(feature_value, (np.ndarray, list)):
                    feature_value = np.mean(feature_value)
                    filtered_features[feature_name] = float(feature_value)
                elif isinstance(feature_value, (float, int, np.float64, np.int64)):
                    filtered_features[feature_name] = float(feature_value)
        return filtered_features
    return float_features


def get_frequency_feature_names(
    feature_list: Optional[Union[List[str], str]] = None,
) -> List[str]:
    """
    Get the list of all available frequency domain feature names.
    """
    default_features: List[str] = [
        "peak_frequency_hz",
        "peak_amplitude",
        "total_power",
        "spectral_centroid_hz",
        "spectral_bandwidth_hz",
        "spectral_skewness",
        "spectral_kurtosis",
        "rms_frequency_hz",
        "frequency_entropy",
        "dominant_frequency_ratio",
        "snr_db",
        "cavitation_indicator",
        "energy_ratio_ultra_low",
        "energy_ratio_low",
        "energy_ratio_medium",
        "energy_ratio_high",
        "energy_ratio_ultra_high",
        "harmonic_1_ratio",
        "harmonic_2_ratio",
        "harmonic_3_ratio",
        "harmonic_4_ratio",
        "harmonic_5_ratio",
    ]
    if feature_list:
        return [feature for feature in feature_list if feature in default_features]
    return default_features


def batch_extract_frequency_features(
    signal: Union[List[np.ndarray], np.ndarray],
    sampling_rate: float = 10000,
    shaft_freq: float = 1750,
    nperseg: int = 1024,
    features_list: Optional[List[str]] = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Extract frequency domain features from different signals in batch.
    Returns:
        2D numpy array of features (n_signals x n_features)
    """
    all_features: List[Dict[str, float]] = []
    try:
        for idx, sig in enumerate(signal):
            features: Dict[str, float] = extract_frequency_features(
                sig,
                sampling_rate=sampling_rate,
                shaft_freq=shaft_freq,
                nperseg=nperseg,
                features_list=features_list,
            )
            all_features.append(features)
            if verbose:
                print(f"Extracted features from signal {idx + 1}/{len(signal)}")
    except Exception as e:
        if verbose:
            print(f"Error during batch feature extraction: {str(e)}")
        if all_features:
            zero_features = {key: 0.0 for key in all_features[0].keys()}
            all_features.append(zero_features)
        else:
            all_features.append({})

    if not all_features:
        return np.array([])

    feature_names: List[str] = [feat for feat in all_features[0].keys()]
    feature_matrix: np.ndarray = np.zeros((len(all_features), len(feature_names)))
    for i, feat_dict in enumerate(all_features):
        for j, feat_name in enumerate(feature_names):
            feature_matrix[i, j] = feat_dict.get(feat_name, 0.0)
    if verbose:
        print(
            f"Extracted {len(all_features)} frequency features from {len(feature_names)} signals"
        )

    return feature_matrix


def plot_frequency_spectrum_example(
    signal: np.ndarray,
    sampling_rate: float = 10000,
    nperseg: int = 1024,
    shaft_freq: float = 1750,
    title: str = "Frequency Spectrum",
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the frequency spectrum of a given signal.
    """
    features: Dict[str, Union[float, np.ndarray, List[float]]] = (
        _compute_frequency_features(
            signal, shaft_freq=shaft_freq, sampling_rate=sampling_rate, nperseg=nperseg
        )
    )
    freqs: np.ndarray = features["frequency_spectrum_freqs"]
    psd: np.ndarray = features["frequency_spectrum_psd"]

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].semilogy(freqs, psd, color="blue")
    ax[0, 0].set_title("Power Spectral Density")
    ax[0, 0].set_xlabel("Frequency (Hz)")
    ax[0, 0].set_ylabel("PSD (V^2/Hz)")
    ax[0, 0].grid(True, alpha=0.3)

    # Highlight key frequencies
    ax[0, 1].plot(freqs, psd, color="blue")
    ax[0, 1].set_title("Frequency Spectrum with Key Frequencies")
    ax[0, 1].set_xlabel("Frequency (Hz)")
    ax[0, 1].set_ylabel("PSD (V^2/Hz)")
    ax[0, 1].grid(True, alpha=0.3)

    # mark peak frequency
    peak_freq: float = features["peak_frequency_hz"]
    peak_idx: int = np.argmin(np.abs(freqs - peak_freq))
    ax[0, 1].axvline(
        x=peak_freq,
        color="red",
        linestyle="--",
        alpha=0.7,
        label=f"Peak: {peak_freq:.1f} Hz",
    )
    shaft_freq_hz: float = shaft_freq / 60.0
    ax[0, 1].axvline(
        x=shaft_freq_hz,
        color="green",
        linestyle="--",
        alpha=0.7,
        label=f"Shaft Freq: {shaft_freq_hz:.1f} Hz",
    )

    # mark harmonics
    colors = ["orange", "purple", "brown", "pink", "gray"]
    for i in range(1, 6):
        harmonic_freq: float = i * shaft_freq_hz
        if harmonic_freq < freqs[-1]:
            ax[0, 1].axvline(
                x=harmonic_freq,
                color=colors[i - 1],
                linestyle="--",
                alpha=0.7,
                label=f"Harmonic {i}: {harmonic_freq:.1f} Hz",
            )

    ax[0, 1].legend(fontsize="small")

    # Energy distribution across bands
    band_names = [
        "ultra_low_freq",
        "low_freq",
        "medium_freq",
        "high_freq",
        "ultra_high_freq",
    ]
    band_colors = ["cyan", "magenta", "yellow", "orange", "red"]
    band_powers = [
        features.get(f"energy_ratio_{band}", 0.0) * 100 for band in band_names
    ]
    ax[1, 0].bar(range(len(band_names)), band_powers, color=band_colors)
    ax[1, 0].set_title("Energy Distribution Across Frequency Bands")
    ax[1, 0].set_xlabel("Frequency Bands")
    ax[1, 0].set_ylabel("Energy Percentage (%)")
    ax[1, 0].set_xticks(range(len(band_names)))
    ax[1, 0].set_xticklabels(band_names, rotation=45, ha="right")
    ax[1, 0].grid(True, alpha=0.3)

    # Add percentage label
    for i, power in enumerate(band_powers):
        ax[1, 0].text(
            i, power + 1, f"{power:.1f}%", ha="center", va="bottom", fontsize=8
        )

    key_features = {
        "Peak Freq (Hz)": f"{features['peak_frequency_hz']:.1f}",
        "Spectral Centroid (Hz)": f"{features['spectral_centroid_hz']:.1f}",
        "Total Power": f"{features['total_power']:.2e}",
        "SNR (dB)": f"{features['snr_db']:.1f}",
        "Cavitation Indicator": f"{features['cavitation_indicator']:.3f}",
    }
    summary_text = "\n".join([f"{k}: {v}" for k, v in key_features.items()])
    ax[1, 1].text(
        0.1,
        0.5,
        summary_text,
        fontsize=12,
        verticalalignment="center",
        fontfamily="monospace",
    )

    ax[1, 1].set_title("Feature Summary")
    ax[1, 1].axis("off")

    # Main title
    fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")

    plt.show()
