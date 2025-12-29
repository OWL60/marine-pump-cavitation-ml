'''
Marine pump vibration data generator
Generates synthetic vibration data for marine pumps
'''

import numpy as np
import pandas as pd
from scipy import signal
from utils import log

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

    def plot_signal(self, axes: np.ndarray, signal: np.ndarray, pos: int, title: str = 'Normal pump vibration') -> None:
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
        axes[pos].plot(signal, 'b-', label='Vibration Signal', linewidth=1)  
        axes[pos].set_title(title)
        axes[pos].set_xlabel('sample number')
        axes[pos].set_ylabel('Vibration Amplitude')
        axes[pos].grid(True, alpha=0.3)

    def add_cavitation_effect(self, signal: np.ndarray, sample_rate, severity: float = 'mild', cavitation_start: float = 0.5) -> np.ndarray:
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
            log.log_warning("cavitation_start exceeds signal duration. No cavitation effect added.")
            start_index = n_sample // 2  # Default to middle of the signal

        cav_freq = 8000  # Cavitation frequency component
        severity_levels = {
            'mild': 0.3,
            'moderate': 0.6,
            'severe': 1.0
        }
        hf_noise = np.zeros(n_sample)
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
            spike_indices = np.random.choice(np.arange(start_index, n_sample), size=min(num_spikes, n_sample - start_index), replace=False)
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

        hf_mask = (np.abs(freq) >= 5000) & (np.abs(freq) <= 10000)
        hf_energy = np.sum(fft_cav[hf_mask])
        log.log_info(f"RMS before cavitation: {rms_before:.3f}, RMS after cavitation: {rms_after:.3f}, High-frequency energy: {hf_energy:.3f}")

        lf_mask = np.abs(freq) <= 100
        lf_energy = np.sum(fft_cav[lf_mask])
        log.log_info(f"Low-frequency energy: {lf_energy:.3f}")

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
    
    def add_ship_motion(self, signal: np.ndarray, marine_condition: str = 'moderate', include_engine_load: bool = True) -> np.ndarray:
        """
        Add ship motion effect to the vibration signal.  
        Args:
            signal (np.ndarray): Original vibration signal.
            marine_condition (str): Marine condition ('calm', 'moderate', 'rough').
            include_engine_load (bool): Whether to include engine load variations.
        Returns:
            np.ndarray: Vibration signal with ship motion effect.
        """
        log.log_warning(f"Adding ship motion effect to the signal (sea state: {marine_condition})")
        
        motion_signal = signal.copy()
        n_sample = len(signal)
        t = np.linspace(0,  n_sample / self.sample_rate, n_sample, endpoint=False)

        roll_params = {
            'calm': {'freq': 0.05, 'amplitude': 0.05, 'phase': 0.0},
            'moderate': {'freq': 0.12, 'amplitude': 0.15, 'phase': 0.0},
            'rough': {'freq': 0.18, 'amplitude': 0.25, 'phase': np.pi / 4},
            'storm': {'freq': 0.22, 'amplitude': 0.35, 'phase': np.pi / 2}, 
        }
        params = roll_params.get(marine_condition, roll_params['moderate'])
        roll_frequency = params['freq']
        roll_amplitude = params['amplitude'] 
        roll_phase = params['phase']

        ship_roll = roll_amplitude * np.sin(2 * np.pi * roll_frequency * t + roll_phase)

        # Pitch motion
        pitch_params = {
            'calm': {'freq': 0.07, 'amplitude': 0.03},
            'moderate': {'freq': 0.10, 'amplitude': 0.10},
            'rough': {'freq': 0.15, 'amplitude': 0.18},
            'storm': {'freq': 0.20, 'amplitude': 0.30}, 
        }
        params = pitch_params.get(marine_condition, pitch_params['moderate'])
        pitch_frequency = params['freq']
        pitch_amplitude = params['amplitude']

        ship_pitch = pitch_amplitude * np.sin(2 * np.pi * pitch_frequency * t + np.pi / 2)

        # heave motion
        heave_params = {
            'calm': {'freq': 0.03, 'amplitude': 0.02},
            'moderate': {'freq': 0.06, 'amplitude': 0.08},
            'rough': {'freq': 0.09, 'amplitude': 0.12},
            'storm': {'freq': 0.12, 'amplitude': 0.20}, 
        }
        params = heave_params.get(marine_condition, heave_params['moderate'])
        heave_frequency = params['freq']
        heave_amplitude = params['amplitude']

        ship_heave = heave_amplitude * np.sin(2 * np.pi * heave_frequency * t + np.pi / 4)

        # Engine load variations if enabled
        if include_engine_load:
            # Engine load based on ship operation, Manouvering, cruising, full speed
            load_profile = np.ones_like(signal)
            phase_duration = n_sample // 4
            load_profile[:phase_duration] *= np.uniform.random(0.6, 0.8) # Manoeuvering
            load_profile[phase_duration:2*phase_duration] *= np.uniform.random(0.8, 1.0) # Cruising
            load_profile[2*phase_duration:3*phase_duration] *= np.uniform.random(1.0, 1.2) # Full speed
            load_profile[3*phase_duration:] *= np.uniform.random(0.7, 0.9) # Slowing down

            #Add some random load fluctuations
            load_fluctuations = 0.05 * np.random.randn(n_sample)
            load_profile *= (1 + 0.1 * load_fluctuations) 

            load_profile = np.clip(load_profile, 0.3, 1.0) # Limit load profile
        else:
            load_profile = np.ones_like(signal) # No load variations


        # Seawater turbulence effect
        density_factor = 1.025  # Density of seawater

        # Sea temperature varies between 0 and 30 degrees Celsius
        temp_variation = 15 + 10 * np.sin(2 * np.pi * 0.01 * t)  # Slow variation over time
        temperature_factor = 1 + (temp_variation - 15) * 0.01  # Simplified effect

        # Bubble/Cavitation noise from seawater
        bubble_noise = 0.02 * np.random.normal(size=signal.shape) * density_factor * temperature_factor

        # Combine all effects
        motion_effect = 1 + ship_roll + ship_pitch + ship_heave
        motion_signal = motion_signal * motion_effect * load_profile + bubble_noise 


        log.log_info(" Marine Conditions Applied:")
        log.log_info(f" • Ship rolling: {roll_frequency:.2f} Hz, amplitude: {roll_amplitude:.2f}")
        log.log_info(f" • Ship pitching: {pitch_frequency:.2f} Hz, amplitude: {pitch_amplitude:.2f}")
        log.log_info(f" • Ship heaving: {heave_frequency:.2f} Hz, amplitude: {heave_amplitude:.2f}")
        
        if include_engine_load:
            avg_load = np.mean(load_profile)
            min_load = np.min(load_profile)
            max_load = np.max(load_profile)
            log.log_info(f" • Engine load: {min_load:.1f}-{max_load:.1f} (avg: {avg_load:.2f})")
        
        # Calculate modulation depth
        modulation_depth = np.max(motion_effect) - np.min(motion_effect)
        log.log_info(f" • Motion modulation: {modulation_depth:.2f}")
        
        # Calculate signal change
        rms_original = np.sqrt(np.mean(signal**2))
        rms_marine = np.sqrt(np.mean(motion_signal**2))
        log.log_info(f" • RMS change: {rms_marine/rms_original:.2f} x")
        return motion_signal
    
    def comapare_land_v_marine(self, land_signal: np.ndarray, marine_signal: np.ndarray) -> dict:
        """
        Compare land-based and marine-based vibration signals using statistical metrics.  
        Args:
            land_signal (np.ndarray): Land-based vibration signal.
            marine_signal (np.ndarray): Marine-based vibration signal.
        Returns:
            dict: Dictionary containing comparison metrics. 
        """
        stats = {
            'land_rms': np.sqrt(np.mean(land_signal**2)),
            'marine_rms': np.sqrt(np.mean(marine_signal**2)),
            'land_std': np.std(land_signal),
            'marine_std': np.std(marine_signal),
            'land_kurtosis': np.mean(land_signal - np.mean(land_signal))**4 / (np.std(land_signal)**4),
            'marine_kurtosis': np.mean(marine_signal - np.mean(marine_signal))**4 / (np.std(marine_signal)**4),
            'variation_coefficient': np.std(marine_signal) / (np.mean(marine_signal) + 1e-6),
        }

        # Calculate PSD (Power Spectral Density) using Welch's method
        f_land, pxx_land = signal.welch(land_signal, fs=self.sample_rate, nperseg=1024)
        f_marine, pxx_marine = signal.welch(marine_signal, fs=self.sample_rate, nperseg=1024)

        # mask for frequencies <= 2.0 Hz
        low_freq_mask = f_land <= 2.0
        stats['land_low_freq_energy'] = np.sum(pxx_land[low_freq_mask])
        stats['marine_low_freq_energy'] = np.sum(pxx_marine[low_freq_mask])
        stats['low_freq_energy_ratio'] = stats['marine_low_freq_energy'] / (stats['land_low_freq_energy'] + 1e-6)   

        # log results with calculated stats
        log.log_info(" Comparison between Land-based and Marine-based Vibration Signals:")
        log.log_info(f" • RMS: Land = {stats['land_rms']:.3f}, Marine = {stats['marine_rms']:.3f}, Ratio = {stats['marine_rms']/stats['land_rms']:.2f} x")
        log.log_info(f" • Std Dev: Land = {stats['land_std']:.3f}, Marine = {stats['marine_std']:.3f}, Ratio = {stats['marine_std']/stats['land_std']:.2f} x")
        log.log_info(f" • Kurtosis: Land = {stats['land_kurtosis']:.3f}, Marine = {stats['marine_kurtosis']:.3f}")
        log.log_info(f" • Low-Freq Energy (<=2.0 Hz): Land = {stats['land_low_freq_energy']:.3f}, Marine = {stats['marine_low_freq_energy']:.3f}, Ratio = {stats['low_freq_energy_ratio']:.2f} x")       
        
        return stats
    
    def generate_marine_scenarios(self, base_signal: np.ndarray, scenarios: list) -> dict:
        """
        Generate multiple marine scenarios from a base vibration signal.  
        Args:
            signal (np.ndarray): Base vibration signal.
            scenarios (list): List of scenario configurations.
        Returns:
            dict: Dictionary of generated scenario signals.
        """
        scenarios = ['calm', 'moderate', 'rough', 'storm']
        marine_signals = {}
        for scenario in scenarios:
            marine_signal = self.add_ship_motion(signal=base_signal, marine_condition=scenario, include_engine_load=True)
            marine_signals[scenario] = marine_signal
            log.log_info("Generated marine scenario: {scenario}")

        # calculate statistics for each scenario
        for scenario, signal in marine_signals.items():
            rms = np.sqrt(np.mean(signal**2))
            std = np.std(signal)
            log.log_info(f" • Scenario: {scenario} | RMS: {rms:.3f} | Std Dev: {std:.3f}")

        return marine_signals
    