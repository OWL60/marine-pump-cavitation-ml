"""Wavelet-based time-frequency feature extraction for pump vibration signals."""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import fftconvolve


class WaveletTimeFrequencyFeatureExtractor:
    """Extract time-frequency features using a Morlet-CWT implementation."""

    EPS = np.finfo(np.float64).eps

    def __init__(
        self,
        sample_rate: float,
        scales: Optional[Sequence[float]] = None,
        w0: float = 6.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.scales = np.array(scales if scales is not None else np.arange(1, 128))
        self.w0 = w0

    def _morlet_wavelet(self, scale: float) -> np.ndarray:
        width = int(max(8 * scale, 16))
        t = np.arange(-width, width + 1, dtype=float)
        x = t / scale
        wavelet = (np.pi ** -0.25) * np.exp(1j * self.w0 * x) * np.exp(-(x**2) / 2)
        return wavelet / np.sqrt(scale)

    def _compute_cwt(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        coefficients = []
        for scale in self.scales:
            wavelet = self._morlet_wavelet(float(scale))
            coeff = fftconvolve(signal, np.conj(wavelet[::-1]), mode="same")
            coefficients.append(coeff)

        coeffs = np.asarray(coefficients)
        frequencies = self.w0 * self.sample_rate / (2.0 * np.pi * self.scales)
        return coeffs, frequencies

    def extract_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Compute wavelet-domain summary features from a 1D signal."""
        coeffs, frequencies = self._compute_cwt(signal)
        power = np.abs(coeffs) ** 2

        total_wavelet_energy = float(np.sum(power))
        if total_wavelet_energy <= self.EPS:
            return {
                "wavelet_total_energy": 0.0,
                "wavelet_peak_frequency_hz": 0.0,
                "wavelet_spectral_centroid_hz": 0.0,
                "wavelet_entropy": 0.0,
                "wavelet_high_low_energy_ratio": 0.0,
                "wavelet_temporal_variability": 0.0,
            }

        energy_per_scale = np.sum(power, axis=1)
        peak_frequency = float(frequencies[np.argmax(energy_per_scale)])
        spectral_centroid = float(
            np.sum(frequencies * energy_per_scale) / (np.sum(energy_per_scale) + self.EPS)
        )

        distribution = energy_per_scale / (np.sum(energy_per_scale) + self.EPS)
        distribution = distribution[distribution > 0]
        wavelet_entropy = float(-np.sum(distribution * np.log2(distribution)))

        cutoff = np.median(frequencies)
        high_energy = float(np.sum(energy_per_scale[frequencies >= cutoff]))
        low_energy = float(np.sum(energy_per_scale[frequencies < cutoff]))
        high_low_energy_ratio = float(high_energy / (low_energy + self.EPS))

        time_energy = np.sum(power, axis=0)
        temporal_variability = float(np.std(time_energy) / (np.mean(time_energy) + self.EPS))

        return {
            "wavelet_total_energy": total_wavelet_energy,
            "wavelet_peak_frequency_hz": peak_frequency,
            "wavelet_spectral_centroid_hz": spectral_centroid,
            "wavelet_entropy": wavelet_entropy,
            "wavelet_high_low_energy_ratio": high_low_energy_ratio,
            "wavelet_temporal_variability": temporal_variability,
        }

    def batch_extract(self, signals: List[np.ndarray]) -> np.ndarray:
        """Extract features for multiple signals as a dense matrix."""
        feature_rows = [self.extract_features(signal) for signal in signals]
        if not feature_rows:
            return np.array([])

        feature_names = list(feature_rows[0].keys())
        return np.array([[row[name] for name in feature_names] for row in feature_rows])
