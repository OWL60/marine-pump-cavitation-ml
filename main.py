"""
main
"""

from src.data import MarinePumpVibrationDataGenerator
from src.features.frequency_features import FrequencyFeatureExtractor

if __name__ == "__main__":
    generator = MarinePumpVibrationDataGenerator(sample_rate=10000)
    signal = generator.generate_vibration_signal()
    signal = generator.add_ship_motion(
        signal, marine_condition="rough", include_engine_load=True
    )
    signals = generator.add_cavitation_effect(signal, severity="severe")

    extractor = FrequencyFeatureExtractor(generator)
    extractor.plot_frequency_spectrum_example(
        signal=signals, save_path="images/frequency_spectrum.png"
    )
