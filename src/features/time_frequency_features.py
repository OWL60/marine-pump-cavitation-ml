"""Time-frequency domain feature extraction utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.signal import spectrogram


EPS = np.finfo(np.float64).eps


def extract_time_frequency_features(
    signal: np.ndarray,
    sample_rate: float,
    nperseg: int = 256,
    noverlap: int = 128,
) -> Dict[str, float]:
    """Extract compact STFT/spectrogram-based features for cavitation analysis."""
    x = np.asarray(signal, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("signal must be a 1D numpy array")
    if len(x) < 32:
        raise ValueError("signal must have at least 32 samples")

    freqs, times, sxx = spectrogram(
        x,
        fs=sample_rate,
        window="hann",
        nperseg=min(nperseg, len(x)),
        noverlap=min(noverlap, max(min(nperseg, len(x)) - 1, 0)),
        scaling="density",
        mode="psd",
    )

    if sxx.size == 0:
        raise ValueError("spectrogram failed to produce valid bins")

    total_energy = float(np.sum(sxx))
    low_band = (freqs >= 0) & (freqs < 300)
    mid_band = (freqs >= 300) & (freqs < 1500)
    high_band = freqs >= 1500

    def band_energy(mask: np.ndarray) -> float:
        return float(np.sum(sxx[mask])) if np.any(mask) else 0.0

    low_e = band_energy(low_band)
    mid_e = band_energy(mid_band)
    high_e = band_energy(high_band)

    flatness_num = np.exp(np.mean(np.log(sxx + EPS)))
    flatness_den = np.mean(sxx + EPS)
    spectral_flatness = float(flatness_num / (flatness_den + EPS))

    temporal_energy = np.sum(sxx, axis=0)
    temporal_var = float(np.var(temporal_energy))

    dominant_freq_over_time = freqs[np.argmax(sxx, axis=0)]
    freq_instability = float(np.std(dominant_freq_over_time))

    return {
        "tf_total_energy": total_energy,
        "tf_low_band_ratio": low_e / (total_energy + EPS),
        "tf_mid_band_ratio": mid_e / (total_energy + EPS),
        "tf_high_band_ratio": high_e / (total_energy + EPS),
        "tf_spectral_flatness": spectral_flatness,
        "tf_temporal_energy_var": temporal_var,
        "tf_dominant_freq_instability": freq_instability,
        "tf_time_bins": float(len(times)),
    }
