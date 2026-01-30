"""
main
"""

from src.data import MarinePumpVibrationDataGenerator
from src.features import frequency_features

if __name__ == "__main__":
    generator = MarinePumpVibrationDataGenerator(sample_rate=10000)
    signal = generator.generate_vibration_signal(rpm=1750, duration=1.0)
    signal = generator.add_ship_motion(
        signal, marine_condition="rough", include_engine_load=True
    )
    signals = generator.add_cavitation_effect(
        signal, sample_rate=generator.sample_rate, severity="severe"
    )
    frequency_features.plot_frequency_spectrum_example(
        signal=signals, save_path="images/frequency_spectrum.png"
    )
