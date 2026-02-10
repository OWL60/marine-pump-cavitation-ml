"""Time-frequency domain feature extraction utilities."""

from typing import Dict

import numpy as np
from scipy.signal import stft


EPS = np.finfo(np.float64).eps


def _safe_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize a positive vector safely."""
    total = np.sum(values)
    if total <= EPS:
        return np.zeros_like(values)
    return values / total


def extract_time_frequency_features(
    signal: np.ndarray,
    sample_rate: int,
    nperseg: int = 256,
    noverlap: int = 128,
) -> Dict[str, float]:
    """Extract compact time-frequency features using STFT."""
    x = np.asarray(signal, dtype=np.float64)
    x = x[np.isfinite(x)]

    if x.size < 16:
        raise ValueError("Signal too short for time-frequency extraction (min 16 samples).")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    x = x - np.mean(x)
    nperseg = int(min(max(32, nperseg), len(x)))
    noverlap = int(min(max(0, noverlap), nperseg - 1))

    freqs, _, zxx = stft(
        x,
        fs=sample_rate,
        nperseg=nperseg,
        noverlap=noverlap,
        boundary=None,
        padded=False,
    )

    power = np.abs(zxx) ** 2
    total_power = float(np.sum(power))

    if total_power <= EPS:
        return {
            "tf_total_power": 0.0,
            "tf_power_mean": 0.0,
            "tf_power_std": 0.0,
            "tf_spectral_entropy": 0.0,
            "tf_dominant_frequency_hz": 0.0,
            "tf_spectral_flux_mean": 0.0,
            "tf_bandwidth_mean_hz": 0.0,
            "tf_low_band_ratio": 0.0,
            "tf_mid_band_ratio": 0.0,
            "tf_high_band_ratio": 0.0,
        }

    mean_spectrum = np.mean(power, axis=1)
    norm_spec = _safe_normalize(mean_spectrum)
    valid = norm_spec > 0
    entropy = float(-np.sum(norm_spec[valid] * np.log2(norm_spec[valid])))

    dominant_idx = int(np.argmax(mean_spectrum))
    dominant_frequency_hz = float(freqs[dominant_idx])

    if power.shape[1] > 1:
        spectral_flux = np.sqrt(np.sum(np.diff(power, axis=1) ** 2, axis=0))
        spectral_flux_mean = float(np.mean(spectral_flux))
    else:
        spectral_flux_mean = 0.0

    per_time_energy = np.sum(power, axis=0) + EPS
    centroid_t = np.sum(freqs[:, None] * power, axis=0) / per_time_energy
    bandwidth_t = np.sqrt(
        np.sum(((freqs[:, None] - centroid_t[None, :]) ** 2) * power, axis=0)
        / per_time_energy
    )

    nyquist = sample_rate / 2.0
    low_mask = freqs <= 0.2 * nyquist
    mid_mask = (freqs > 0.2 * nyquist) & (freqs <= 0.6 * nyquist)
    high_mask = freqs > 0.6 * nyquist

    low_ratio = float(np.sum(power[low_mask]) / total_power)
    mid_ratio = float(np.sum(power[mid_mask]) / total_power)
    high_ratio = float(np.sum(power[high_mask]) / total_power)

    return {
        "tf_total_power": float(total_power),
        "tf_power_mean": float(np.mean(power)),
        "tf_power_std": float(np.std(power)),
        "tf_spectral_entropy": entropy,
        "tf_dominant_frequency_hz": dominant_frequency_hz,
        "tf_spectral_flux_mean": spectral_flux_mean,
        "tf_bandwidth_mean_hz": float(np.mean(bandwidth_t)),
        "tf_low_band_ratio": low_ratio,
        "tf_mid_band_ratio": mid_ratio,
        "tf_high_band_ratio": high_ratio,
    }
