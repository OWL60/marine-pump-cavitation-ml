from src.data.generator import MarinePumpVibrationDataGenerator
from tests import tests 


if __name__ == "__main__":
    generator = MarinePumpVibrationDataGenerator(sample_rate=1000)
    vibration_signal = generator.generate_vibration_signal(rpm=1750, duration=2.0)
    tests.test_generate_vibration_signal()
    generator.plot_signal(vibration_signal)