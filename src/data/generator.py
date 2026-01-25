"""
Marine pump vibration data generator
Generates synthetic vibration data for marine pumps
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import signal

from utils import log

from ..features.time_features import batch_extract


class MarinePumpVibrationDataGenerator:
    """
    Marine Pump Vibration Data Generator
    Generates synthetic vibration data for marine pumps,
    including cavitation effects and marine conditions.
    Args:
        sample_rate (int): Sampling rate in Hz.
    """

    def __init__(self, sample_rate: int = 1000):
        self.sample_rate = sample_rate

    def generate_vibration_signal(
        self, rpm: int = 1750, duration: float = 1.0
    ) -> np.ndarray:
        """
        Generate synthetic vibration signal for a marine pump.
        Args:
            rpm (int): Rotations per minute of the pump.
            duration (float): Duration of the signal in seconds.
        Returns:
            np.ndarray: Generated vibration signal.

        """
        if rpm < 0:
            raise ValueError("rpm must be positive")
        if duration < 1:
            raise ValueError("duration must be positive")

        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        shaft_frequency = rpm / 60  # Convert RPM to Hz
        signals = 0.5 * np.sin(2 * np.pi * shaft_frequency * t)  # Base vibration signal
        # Add harmonics
        signals += 0.2 * np.sin(2 * np.pi * 2 * shaft_frequency * t)  # 2nd harmonic
        signals += 0.1 * np.sin(2 * np.pi * 3 * shaft_frequency * t)  # 3rd harmonic

        # Add some noise
        noise = 0.05 * np.random.normal(size=signals.shape)
        signals += noise

        return signals

    def save_signal_to_csv(
        self, signals: np.ndarray, filename: str = "normal_vibration.csv"
    ) -> None:
        """
        Save the generated vibration signal to a CSV file.
        Args:
            signal (np.ndarray): Vibration signal to save.
            filename (str): Name of the CSV file.
        """
        df = pd.DataFrame(
            {
                "time": np.arange(len(signals)) / self.sample_rate,
                "vibration": signals,
            }
        )
        df.to_csv(filename, index=False)

    def plot_signal(
        self,
        axes: np.ndarray,
        signals: np.ndarray,
        pos: int,
        title: str = "Normal pump vibration",
    ) -> None:
        """
        Plot the vibration signal on given axes.
        Args:
            axes (np.ndarray): Array of matplotlib axes to plot on.
            signal (np.ndarray): Vibration signal to plot.
            pos (int): Position index in the axes array.
            title (str): Title of the plot.
        Returns:
            None
        """
        axes[pos].plot(signals, "b-", label="Vibration Signal", linewidth=1)
        axes[pos].set_title(title)
        axes[pos].set_xlabel("sample number")
        axes[pos].set_ylabel("Vibration Amplitude")
        axes[pos].grid(True, alpha=0.3)

    def add_cavitation_effect(
        self,
        signals: np.ndarray,
        sample_rate,
        severity: float = "mild",
        cavitation_start: float = 0.5,
    ) -> np.ndarray:
        """
        Add cavitation effect to the vibration signal.
        Args:
            signal (np.ndarray): Original vibration signal.
            severity (float): Severity of the cavitation effect [How bad the cavitation](mild, moderate, severe).
            cavitation_start (float): Time in seconds when cavitation starts.
        Returns:
            np.ndarray: Vibration signal with cavitation effect.
        """
        # Frequency noise
        cavitation_signal = signals.copy()
        n_sample = len(signals)
        time_to_take_sample = np.linspace(0, n_sample / sample_rate, n_sample)
        start_index = int(cavitation_start * sample_rate)

        if start_index >= n_sample:
            log.log_warning("Start index exceeds number of sample.")
            start_index = n_sample // 2  # Default to middle of the signal

        cav_freq = 8000  # Cavitation frequency component
        severity_levels = {"mild": 0.3, "moderate": 0.6, "severe": 1.0}
        hf_noise = np.zeros(n_sample)
        hf_noise[start_index:] = 0.3 * np.sin(
            2 * np.pi * cav_freq * time_to_take_sample[start_index:]
        )
        hf_noise *= severity_levels.get(severity, 0.5)

        # Add Random spikes
        spike = np.zeros_like(signals)
        duration_after_cavitation = (n_sample - start_index) / sample_rate
        spikes_per_second = {"mild": 5, "moderate": 20, "severe": 30}
        num_spikes = int(
            duration_after_cavitation * spikes_per_second.get(severity, 10)
        )

        if num_spikes > 0:
            spike_indices = np.random.choice(
                np.arange(start_index, n_sample),
                size=min(num_spikes, n_sample - start_index),
                replace=False,
            )
            spike_amplitudes = {"mild": 0.5, "moderate": 1.2, "severe": 2.5}
            spike_amp = spike_amplitudes.get(severity, 1.0) * np.random.randn(
                len(spike_indices)
            )
            spike[spike_indices] = spike_amp

        # Amplitude modulation
        modulation = np.zeros_like(signals)
        mod_freq = 5  # 5 Hz modulation frequency
        modulation[start_index:] = 0.5 * np.sin(
            2 * np.pi * mod_freq * time_to_take_sample[start_index:]
        )
        modulation_effect = modulation + 1.0

        # Add sub-harmonics
        shaft_frequency = 1750 / 60  # Assuming rpm=1750 for sub-harmonic calculation
        sub_harmonic = np.zeros_like(signals)
        sub_harmonic_freq = shaft_frequency / 2  # Half of shaft frequency
        sub_harmonic[start_index:] = 0.2 * np.sin(
            2 * np.pi * sub_harmonic_freq * time_to_take_sample[start_index:]
        )

        # Combine all effects
        cavitation_signal[start_index:] = (
            cavitation_signal[start_index:]
            + hf_noise[start_index:]
            + spike[start_index:]
            + sub_harmonic[start_index:]
        ) * modulation_effect[start_index:]

        # Add some extra noise for severe cavitation
        if severity == "severe":
            extra_noise = np.zeros_like(signals)
            extra_noise[start_index:] = 0.08 * np.random.randn(n_sample - start_index)
            cavitation_signal += extra_noise

        # rms before and after Cavitation
        rms_before = np.sqrt(np.mean(signals[:start_index] ** 2))
        rms_after = np.sqrt(np.mean(cavitation_signal[start_index:] ** 2))

        # Frequency analysis of cavitation portion
        fft_cav = np.abs(
            np.fft.fft(cavitation_signal[start_index : start_index + 1000])
        )  # Energy (magnitude)
        freq = np.fft.fftfreq(1000, 1 / self.sample_rate)
        log.log_debug(f"freq_max = {np.max(freq)}")

        hf_mask = (np.abs(freq) >= 5000) & (np.abs(freq) <= 10000)
        hf_energy = np.sum(fft_cav[hf_mask])
        log.log_info(
            f"RMS before cavitation: {rms_before:.3f}, RMS after cavitation: {rms_after:.3f}"
        )
        log.log_info(f"High-frequency energy: {hf_energy:.3f}")

        lf_mask = np.abs(freq) <= 100
        lf_energy = np.sum(fft_cav[lf_mask])
        hf_ratio = hf_energy / (lf_energy + 1e-10)
        log.log_info(f"Low-frequency energy: {lf_energy:.3f}")
        log.log_info(f"High frequency ratio: {hf_ratio:.3f}")

        return cavitation_signal

    def compare_signals(
        self,
        normal_signal: np.ndarray,
        cavitation_signal: np.ndarray,
        cavitation_start: float = 0.5,
    ) -> dict:
        """
        Compare normal and cavitation signals using statistical metrics.
        Args:
            normal_signal (np.ndarray): Normal vibration signal.
            cavitation_signal (np.ndarray): Vibration signal with cavitation effect.
        Returns:
            dict: Dictionary containing comparison metrics.
        """
        start_index = int(cavitation_start * self.sample_rate)
        normal_rms = np.sqrt(np.mean(normal_signal[start_index:] ** 2))
        cavitation_rms = np.sqrt(np.mean(cavitation_signal[start_index:] ** 2))

        normal_peak = np.max(np.abs(normal_signal[start_index:]))
        cavitation_peak = np.max(np.abs(cavitation_signal[start_index:]))

        comparison_metrics = {
            "normal_rms": normal_rms,
            "cavitation_rms": cavitation_rms,
            "normal_peak": normal_peak,
            "cavitation_peak": cavitation_peak,
            "peak_increase": None,
            "rms_increase": None,
        }

        comparison_metrics["peak_increase"] = (
            comparison_metrics["cavitation_peak"] / comparison_metrics["normal_peak"]
        )
        comparison_metrics["rms_increase"] = (
            comparison_metrics["cavitation_rms"] / comparison_metrics["normal_rms"]
        )

        # Calculate kurtosis
        comparison_metrics["normal_kurtosis"] = np.mean(
            normal_signal - np.mean(normal_signal)
        ) ** 4 / (np.std(normal_signal) ** 4)
        comparison_metrics["cavitation_kurtosis"] = np.mean(
            cavitation_signal - np.mean(cavitation_signal)
        ) ** 4 / (np.std(cavitation_signal) ** 4)

        return comparison_metrics

    def add_ship_motion(
        self,
        signals: np.ndarray,
        marine_condition: str = "moderate",
        include_engine_load: bool = True,
    ) -> np.ndarray:
        """
        Add ship motion effect to the vibration signal.
        Args:
            signal (np.ndarray): Original vibration signal.
            marine_condition (str): Marine condition ('calm', 'moderate', 'rough').
            include_engine_load (bool): Whether to include engine load variations.
        Returns:
            np.ndarray: Vibration signal with ship motion effect.
        """
        log.log_warning(
            f"Adding ship motion effect to the signal (sea state: {marine_condition})"
        )

        motion_signal = signals.copy()
        n_sample = len(signals)
        t = np.linspace(0, n_sample / self.sample_rate, n_sample, endpoint=False)

        roll_params = {
            "mild": {"freq": 0.05, "amplitude": 0.05, "phase": 0.0},
            "moderate": {"freq": 0.12, "amplitude": 0.15, "phase": 0.0},
            "severe": {"freq": 0.18, "amplitude": 0.25, "phase": np.pi / 4},
        }
        params = roll_params.get(marine_condition, roll_params["moderate"])
        roll_frequency = params["freq"]
        roll_amplitude = params["amplitude"]
        roll_phase = params["phase"]

        ship_roll = roll_amplitude * np.sin(2 * np.pi * roll_frequency * t + roll_phase)

        # Pitch motion
        pitch_params = {
            "mild": {"freq": 0.07, "amplitude": 0.03},
            "moderate": {"freq": 0.10, "amplitude": 0.10},
            "severe": {"freq": 0.15, "amplitude": 0.18},
        }
        params = pitch_params.get(marine_condition, pitch_params["moderate"])
        pitch_frequency = params["freq"]
        pitch_amplitude = params["amplitude"]

        ship_pitch = pitch_amplitude * np.sin(
            2 * np.pi * pitch_frequency * t + np.pi / 2
        )

        # heave motion
        heave_params = {
            "mild": {"freq": 0.03, "amplitude": 0.02},
            "moderate": {"freq": 0.06, "amplitude": 0.08},
            "severe": {"freq": 0.09, "amplitude": 0.12},
        }
        params = heave_params.get(marine_condition, heave_params["moderate"])
        heave_frequency = params["freq"]
        heave_amplitude = params["amplitude"]

        ship_heave = heave_amplitude * np.sin(
            2 * np.pi * heave_frequency * t + np.pi / 4
        )

        # Engine load variations if enabled
        if include_engine_load:
            # Engine load based on ship operation, Manouvering, cruising, full speed
            load_profile = np.ones_like(signals)
            phase_duration = n_sample // 4
            load_profile[:phase_duration] *= np.random.uniform(0.6, 0.8)  # Manoeuvering
            load_profile[phase_duration : 2 * phase_duration] *= np.random.uniform(
                0.8, 1.0
            )  # Cruising
            load_profile[2 * phase_duration : 3 * phase_duration] *= np.random.uniform(
                1.0, 1.2
            )  # Full speed
            load_profile[3 * phase_duration :] *= np.random.uniform(
                0.7, 0.9
            )  # Slowing down

            # Add some random load fluctuations
            load_fluctuations = 0.05 * np.random.randn(n_sample)
            load_profile *= 1 + 0.1 * load_fluctuations

            load_profile = np.clip(load_profile, 0.3, 1.0)  # Limit load profile
        else:
            load_profile = np.ones_like(signals)  # No load variations

        # Seawater turbulence effect
        density_factor = 1.025  # Density of seawater

        # Sea temperature varies between 0 and 30 degrees Celsius
        temp_variation = 15 + 10 * np.sin(
            2 * np.pi * 0.01 * t
        )  # Slow variation over time
        temperature_factor = 1 + (temp_variation - 15) * 0.01  # Simplified effect

        # Bubble/Cavitation noise from seawater
        bubble_noise = (
            0.02
            * np.random.normal(size=signals.shape)
            * density_factor
            * temperature_factor
        )

        # Combine all effects
        motion_effect = 1 + ship_roll + ship_pitch + ship_heave
        motion_signal = motion_signal * motion_effect * load_profile + bubble_noise

        log.log_info("Marine Conditions Applied:")
        log.log_info(
            f"Ship rolling: {roll_frequency:.2f} Hz, amplitude: {roll_amplitude:.2f}"
        )
        log.log_info(
            f"Ship pitching: {pitch_frequency:.2f} Hz, amplitude: {pitch_amplitude:.2f}"
        )
        log.log_info(
            f"Ship heaving: {heave_frequency:.2f} Hz, amplitude: {heave_amplitude:.2f}"
        )

        if include_engine_load:
            avg_load = np.mean(load_profile)
            min_load = np.min(load_profile)
            max_load = np.max(load_profile)
            log.log_info(
                f"Engine load: {min_load:.1f}-{max_load:.1f} (avg: {avg_load:.2f})"
            )

        # Calculate modulation depth
        modulation_depth = np.max(motion_effect) - np.min(motion_effect)
        log.log_info(f"Motion modulation: {modulation_depth:.2f}")

        # Calculate signal change
        rms_original = np.sqrt(np.mean(signals**2))
        rms_marine = np.sqrt(np.mean(motion_signal**2))
        log.log_info(f"RMS change: {rms_marine/rms_original:.2f} x")
        return motion_signal

    def comapare_land_v_marine(
        self, land_signal: np.ndarray, marine_signal: np.ndarray
    ) -> dict:
        """
        Compare land-based and marine-based vibration signals using statistical metrics.
        Args:
            land_signal (np.ndarray): Land-based vibration signal.
            marine_signal (np.ndarray): Marine-based vibration signal.
        Returns:
            dict: Dictionary containing comparison metrics.
        """
        stats = {
            "land_rms": np.sqrt(np.mean(land_signal**2)),
            "marine_rms": np.sqrt(np.mean(marine_signal**2)),
            "land_std": np.std(land_signal),
            "marine_std": np.std(marine_signal),
            "land_kurtosis": np.mean(land_signal - np.mean(land_signal)) ** 4
            / (np.std(land_signal) ** 4),
            "marine_kurtosis": np.mean(marine_signal - np.mean(marine_signal)) ** 4
            / (np.std(marine_signal) ** 4),
            "variation_coefficient": np.std(marine_signal)
            / (np.mean(marine_signal) + 1e-6),
        }

        # Calculate PSD (Power Spectral Density) using Welch's method
        f_land, pxx_land = signal.welch(land_signal, fs=self.sample_rate, nperseg=1024)
        f_marine, pxx_marine = signal.welch(
            marine_signal, fs=self.sample_rate, nperseg=1024
        )

        # mask for frequencies <= 2.0 Hz
        low_freq_mask = f_land <= 2.0
        stats["land_low_freq_energy"] = np.sum(pxx_land[low_freq_mask])
        stats["marine_low_freq_energy"] = np.sum(pxx_marine[low_freq_mask])
        stats["low_freq_energy_ratio"] = stats["marine_low_freq_energy"] / (
            stats["land_low_freq_energy"] + 1e-6
        )

        # log results with calculated stats
        log.log_info(
            "Comparison between Land-based and Marine-based Vibration Signals:"
        )
        log.log_info(
            f"RMS: Land = {stats['land_rms']:.3f}, Marine = {stats['marine_rms']:.3f}, Ratio = {stats['marine_rms']/stats['land_rms']:.2f} x"
        )
        log.log_info(
            f"Std Dev: Land = {stats['land_std']:.3f}, Marine = {stats['marine_std']:.3f}, Ratio = {stats['marine_std']/stats['land_std']:.2f} x"
        )
        log.log_info(
            f"Kurtosis: Land = {stats['land_kurtosis']:.3f}, Marine = {stats['marine_kurtosis']:.3f}"
        )
        log.log_info(
            f"Low-Freq Energy (<=2.0 Hz): Land = {stats['land_low_freq_energy']:.3f}, Marine = {stats['marine_low_freq_energy']:.3f}, Ratio = {stats['low_freq_energy_ratio']:.2f} x"
        )

        return stats

    def generate_marine_scenarios(
        self, base_signal: np.ndarray, including_engine_load: bool = True
    ) -> dict:
        """
        Generate multiple marine scenarios from a base vibration signal.
        Args:
            signal (np.ndarray): Base vibration signal.
        Returns:
            dict: Dictionary of generated scenario signals.
        """
        scenarios = ["mild", "moderate", "severe"]
        marine_signals = {}
        for scenario in scenarios:
            marine_signal = self.add_ship_motion(
                signals=base_signal,
                marine_condition=scenario,
                include_engine_load=including_engine_load,
            )
            marine_signals[scenario] = marine_signal
            log.log_info(f"Generated marine scenario: {scenario}")

        # calculate statistics for each scenario
        for scenario, signals in marine_signals.items():
            rms = np.sqrt(np.mean(signals**2))
            std = np.std(signals)
            log.log_info(f"Scenario: {scenario} | RMS: {rms:.3f} | Std dev: {std:.3f}")

        return marine_signals

    def generate_dataset(
        self,
        n_samples: int = 1000,
        include_marine_conditions: bool = True,
        save_to_disk: bool = True,
    ) -> tuple:
        """
        Generate a dataset of vibration signals with various conditions.
        Args:
            n_samples (int): vibration data to generate.
            include_marine_conditions (bool): Whether to include marine conditions.
            save_to_disk (bool): Whether to save generated signals to disk.
        Returns:
            tuple: (X_raw, X_features, y, metadata)
            X_raw: Raw vibration signals (n_samples, signal_length)
            X_features: Extracted features (n_samples, n_features)
            y: Labels (0=normal, 1=cavitation)
            metadata: Dictionary with sample metadata
        """
        log.log_info(f"Generating dataset with {n_samples} samples...")
        X_raw = []
        y = []
        metadata = []
        duration = 1.0  # seconds
        signal_length = int(self.sample_rate * duration)
        cavitation_severities = ["mild"] * 3 + ["moderate"] * 4 + ["severe"] * 3
        marine_conditions = ["calm"] * 2 + ["moderate"] * 5 + ["rough"] * 3

        for i in range(n_samples):
            if i % 100 == 0:
                log.log_info(f"Generating sample {i+1}/{n_samples}...")

            rpm = np.random.randint(1150, 3601)
            is_cavitation = i >= n_samples // 2
            severity = (
                np.random.choice(cavitation_severities) if is_cavitation else "none"
            )
            cavitation_start = np.random.uniform(0.2, 0.7) if is_cavitation else 1.0
            marine_condition = (
                np.random.choice(marine_conditions)
                if include_marine_conditions
                else "land"
            )

            base_signal = self.generate_vibration_signal()
            signals = np.zeros_like(base_signal)

            if is_cavitation:
                signals = self.add_cavitation_effect(
                    signals=base_signal,
                    sample_rate=1000,
                    severity=severity,
                    cavitation_start=cavitation_start,
                )
            else:
                signals = base_signal

            if include_marine_conditions and marine_condition != "land":
                signals = self.add_ship_motion(
                    signals=base_signal,
                    marine_condition=marine_condition,
                    include_engine_load=True,
                )

            X_raw.append(signals)
            y.append(1 if is_cavitation else 0)
            metadata.append(
                {
                    "sample_id": i,
                    "rpm": rpm,
                    "cavitation_severity": severity,
                    "cavitation_start_time": (
                        cavitation_start if is_cavitation else None
                    ),
                    "marine_condition": marine_condition,
                    "is_marine": include_marine_conditions
                    and marine_condition != "land",
                    "signal_rms": np.sqrt(np.mean(signals**2)),
                    "signal_peak": np.max(np.abs(signals)),
                }
            )
        X_raw = np.array(X_raw)
        y = np.array(y)

        log.log_info(f"Extracting features from {len(X_raw)} signals...")

        X_features = batch_extract(X_raw)

        log.log_info(f"signal shape: {X_raw.shape}")
        log.log_info(f"features shape: {X_features.shape}")
        log.log_info(f"label: {sum(y == 0)} Normal, {sum(y == 1)} Cavitation")

        if save_to_disk:
            self._save_dataset(X_raw, X_features, y, metadata)

        self.print_dataset_summarry(X_raw, y, metadata)

        return X_raw, X_features, y, metadata

    def _save_dataset(
        self, X_raw: np.ndarray, X_features: np.ndarray, y: np.ndarray, metadata: list
    ) -> None:
        os.makedirs("data/raw", exist_ok=True)
        os.makedirs("data/processed", exist_ok=True)
        os.makedirs("data/metadata", exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        np.save(f"data/raw/X_raw_{timestamp}.npy", X_raw)
        np.save(f"data/raw/y_{timestamp}.npy", y)
        np.save(f"data/processed/X_features_{timestamp}.npy", X_features)
        np.save(f"data/processed/y_{timestamp}.npy", y)

        metadata_df = pd.DataFrame(metadata)
        metadata_df.to_csv(f"data/metadata/metadata_{timestamp}.csv", index=False)

        feature_names = [
            "mean",
            "std",
            "rms",
            "peak",
            "crest_factor",
            "kurtosis",
            "skewness",
            "shape_factor",
            "impulse_factor",
            "variance",
            "mean_abs",
            "energy",
        ]

        with open(f"data/processed/feature_names_{timestamp}.txt", "w") as f:
            for name in feature_names:
                f.write(f"{name}\n")

        log.log_info("Dataset saved to disk:")
        log.log_info(f"Raw signals: data/raw/X_raw_{timestamp}.npy")
        log.log_info(f"Features: data/processed/X_features_{timestamp}.npy")
        log.log_info(f"Labels: data/raw/y_{timestamp}.npy")
        log.log_info(f"Metadata: data/metadata/metadata_{timestamp}.csv")
        log.log_info(f"Feature names: data/processed/feature_names_{timestamp}.txt")

    def print_dataset_summarry(
        self, X_raw: np.ndarray, y: np.ndarray, metadata: list
    ) -> None:
        metadata_df = pd.DataFrame(metadata)

        log.log_info("{'='*60}")
        log.log_info("DATASET SUMMARY")
        log.log_info("{'='*60}")

        log.log_info("Basic Statistics:")
        log.log_info(f"Total samples: {len(X_raw)}")
        log.log_info(f"Normal samples: {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
        log.log_info(f"Cavitation samples: {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
        log.log_info(
            f"Signal length: {X_raw.shape[1]} samples ({X_raw.shape[1]/self.sample_rate:.1f}s)"
        )

        log.log_info("Cavitation Severity Distribution:")
        cavitation_meta = metadata_df[metadata_df["cavitation_severity"] != "none"]
        if len(cavitation_meta) > 0:
            severity_counts = cavitation_meta["cavitation_severity"].value_counts()
            for severity, count in severity_counts.items():
                percentage = count / len(cavitation_meta) * 100
                log.log_info(
                    f"{severity.capitalize()}: {count} samples ({percentage:.1f}%)"
                )

        log.log_info("Marine Conditions Distribution:")
        marine_counts = metadata_df["marine_condition"].value_counts()
        for condition, count in marine_counts.items():
            percentage = count / len(metadata_df) * 100
            log.log_info(
                f"{condition.capitalize()}: {count} samples ({percentage:.1f}%)"
            )

        log.log_info("RPM Distribution:")
        log.log_info(f"Min RPM: {metadata_df['rpm'].min()}")
        log.log_info(f"Max RPM: {metadata_df['rpm'].max()}")
        log.log_info(f"Mean RPM: {metadata_df['rpm'].mean():.0f}")
        log.log_info(f"Std RPM: {metadata_df['rpm'].std():.0f}")

        log.log_info("Signal Statistics:")
        log.log_info(f"Mean RMS: {metadata_df['signal_rms'].mean():.4f}")
        log.log_info(f"Mean Peak: {metadata_df['signal_peak'].mean():.4f}")
        log.log_info(f"Signal range: [{X_raw.min():.3f}, {X_raw.max():.3f}]")

        log.log_info("Dataset ready for ML training!")
