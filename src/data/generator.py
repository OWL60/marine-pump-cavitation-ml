'''
Marine pump vibration data generator
Generates synthetic vibration data for marine pumps
'''

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from matplotlib import pyplot as plt
from scipy import signal

class MarinePumpVibrationDataGenerator:
    def __init__(self, sample_rate: int = 1000):
        self.sample_rate = sample_rate
    
    def generate_vibration_signal(self, rpm: int = 1750, duration: float = 1.0) -> np.ndarray:
        """
        Generate synthetic vibration signal for a marine pump.  
        Args:
            rpm (int): Rotations per minute of the pump.
            duration (float): Duration of the signal in seconds.        
        Returns:
            np.ndarray: Generated vibration signal.

        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        shaft_frequency = rpm / 60  # Convert RPM to Hz
        signal = 0.5 * np.sin(2 * np.pi * shaft_frequency * t)  # Base vibration signal
        # Add harmonics
        signal += 0.2 * np.sin(2 * np.pi * 2 * shaft_frequency * t)  # 2nd harmonic
        signal += 0.1 * np.sin(2 * np.pi * 3 * shaft_frequency * t)  # 3rd harmonic

        # Add some noise
        noise = 0.05 * np.random.normal(size=signal.shape)
        signal += noise

        return signal
    

    def save_signal_to_csv(self, signal: np.ndarray, filename: str = 'normal_vibration.csv') -> None:
        """
            Save the generated vibration signal to a CSV file.  

            
            Args:
                signal (np.ndarray): Vibration signal to save.
                filename (str): Name of the CSV file.
        """

        df = pd.DataFrame({
                'time': np.arange(len(signal)) / self.sample_rate,
                'vibration': signal,
            })
        df.to_csv(filename, index=False)


    def plot_signal(self, signal: np.ndarray, title: str = 'Normal pump vibration') -> None:
        """
        Plot the generated vibration signal.  
        Args:
            signal (np.ndarray): Vibration signal to plot.
            title (str): Title of the plot.
        """

        plot_sample = min(1000, len(signal))
        plt.figure(figsize=(10, 4))
        plt.plot(signal[:plot_sample],  'b-', label='Vibration Signal', linewidth=1)
        plt.title(title)
        plt.xlabel('sample number')
        plt.ylabel('Vibration Amplitude')
        plt.grid(True, alpha=0.3)

        plt.figtext(0.5, -0.05, 
            f"Length: {len(signal)} samples | Sample rate: 10 kHz | RMS: {np.sqrt(np.mean(signal**2)):.3f}",
            ha='center', fontsize=9)
        
        plt.savefig("normal_pump_vibration.png", bbox_inches='tight')
        plt.show()

    def add_cavitation_effect(self, signal: np.ndarray, sample_rate, duration, severity: float = 'mild', cavitation_start: float = 0.5) -> np.ndarray:
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
        cavitation_signal = signal.copy()
        n_sample = len(signal)
        time_to_take_sample = np.linspace(0, n_sample / sample_rate, n_sample)
        start_index = int(cavitation_start * sample_rate)

        if start_index >= n_sample:
            print("Warning: cavitation_start exceeds signal duration. No cavitation effect added.")
            start_index = n_sample // 2  # Default to middle of the signal

        cav_freq = 8000  # Cavitation frequency component
        severity_levels = {
            'mild': 0.3,
            'moderate': 0.6,
            'severe': 1.0
        }
        hf_noise = np.zeros_like(n_sample)
        hf_noise[start_index:] = 0.3 * np.sin(2 * np.pi * cav_freq * time_to_take_sample[start_index:])
        hf_noise *= severity_levels.get(severity, 0.5)

        # Add Random spikes
        spike = np.zeros_like(signal)
        duration_after_cavitation = n_sample - start_index / sample_rate
        spikes_per_second = {
            'mild': 5,
            'moderate': 20,
            'severe': 30
        }
        num_spikes = int(spikes_per_second.get(severity, 10) * duration_after_cavitation)
        
        if num_spikes > 0:
            spike_indices = np.random.choice(np.range(start_index, n_sample), size=min(num_spikes, n_sample - start_index), replace=False)
            spike_amplitudes = {
                'mild': 0.5,
                'moderate': 1.0,
                'severe': 2.5
            }
            spike_magnitudes = spike_amplitudes.get(severity, 1.0) * np.random.randn(len(spike_indices))
            spike[spike_indices] = spike_magnitudes

        # Amplitude modulation
        modulation = np.zeros_like(signal)
        mod_freq = 5  # 5 Hz modulation frequency
        modulation[start_index:] = 0.5 * np.sin(2 * np.pi * mod_freq * time_to_take_sample[start_index:])
        modulation_effect = modulation + 1.0

        # Add sub-harmonics
        shaft_frequency = 1750 / 60  # Assuming rpm=1750 for sub-harmonic calculation
        sub_harmonic = np.zeros_like(signal)
        sub_harmonic_freq =  shaft_frequency / 2  # Half of shaft frequency
        sub_harmonic[start_index:] = 0.2 * np.sin(2 * np.pi * sub_harmonic_freq * time_to_take_sample[start_index:])

        # Combine all effects
        cavitation_signal[start_index:] = (cavitation_signal[start_index:] + hf_noise[start_index:] + spike[start_index:] + sub_harmonic[start_index:]) * modulation_effect[start_index:] 

        # Add some extra noise for severe cavitation
        if severity == 'severe':
            extra_noise = 0.1 * np.random.normal(size=cavitation_signal[start_index:].shape)
            cavitation_signal[start_index:] += extra_noise

        # rms before and after Cavitation
        rms_before = np.sqrt(np.mean(signal[:start_index]**2))
        rms_after = np.sqrt(np.mean(cavitation_signal[start_index:]**2))

        # Frequency analysis of cavitation portion
        
        fft_cav = np.abs(np.fft.fft(cavitation_signal[start_index: start_index + 1000]))
        freq = np.fft.fftfreq(1000, 1/self.sample_rate)

        hf_mask = np.abs(freq) >= 5000 & np.abs(freq) <= 10000
        hf_energy = np.sum(fft_cav[hf_mask])
        print(f"RMS before cavitation: {rms_before:.3f}, RMS after cavitation: {rms_after:.3f}, High-frequency energy: {hf_energy:.3f}")

        lf_mask = np.abs(freq) <= 100
        lf_energy = np.sum(fft_cav[lf_mask])
        print(f"Low-frequency energy: {lf_energy:.3f}")

        return cavitation_signal
    
    def compare_signals(self, normal_signal: np.ndarray, cavitation_signal: np.ndarray, cavitation_start: float = 0.5) -> dict:
        """
        Compare normal and cavitation signals using statistical metrics.  
        Args:
            normal_signal (np.ndarray): Normal vibration signal.
            cavitation_signal (np.ndarray): Vibration signal with cavitation effect.
        Returns:
            dict: Dictionary containing comparison metrics. 
        """
        start_index = int(cavitation_start * self.sample_rate)
        normal_rms = np.sqrt(np.mean(normal_signal[start_index:]**2))
        cavitation_rms = np.sqrt(np.mean(cavitation_signal[start_index:]**2))

        normal_peak = np.max(np.abs(normal_signal[start_index:]))
        cavitation_peak = np.max(np.abs(cavitation_signal[start_index:]))

        comparison_metrics = {
            'normal_rms': normal_rms,
            'cavitation_rms': cavitation_rms,
            'normal_peak': normal_peak,
            'cavitation_peak': cavitation_peak,
            'peak_increase': None,
            'rms_increase': None,
        }

        comparison_metrics['peak_increase'] = comparison_metrics['cavitation_peak'] / comparison_metrics['normal_peak']
        comparison_metrics['rms_increase'] = comparison_metrics['cavitation_rms'] / comparison_metrics['normal_rms']

        # Calculate kurtosis
        comparison_metrics['normal_kurtosis'] = np.mean(normal_signal - np.mean(normal_signal))**4 / (np.std(normal_signal)**4)
        comparison_metrics['cavitation_kurtosis'] = np.mean(cavitation_signal - np.mean(cavitation_signal))**4 / (np.std(cavitation_signal)**4)

        return comparison_metrics