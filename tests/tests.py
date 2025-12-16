import pandas as pd
import numpy as np
from utils import log
from src.data.generator import MarinePumpVibrationDataGenerator

def test_generate_vibration_signal() -> None:
    """
    Test the generate_vibration_signal method.      
    Returns:
        None
    """
    generator = MarinePumpVibrationDataGenerator(sample_rate=1000)  
    signal = generator.generate_vibration_signal(rpm=1750, duration=1.0)

    # find statics data, type and shape  
    stats = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'max': np.max(signal),
        'min': np.min(signal),
        'rms': np.sqrt(np.mean(signal**2)),
        'peak': np.max(np.abs(signal)),
    }

    assert len(signal) == int(generator.sample_rate * 1.0), "Signal length mismatch"
    assert abs(stats['mean']) < 0.1, "Mean value too high"
    assert stats['std'] > 0, "Standard deviation should be positive"
    assert stats['rms'] > 0, "RMS value should be positive"
    log.log_success("Vibration signal generation test passed!")

def test_cavitation_effect() -> None:
    """
    Test the add_cavitation_effect method of MarinePumpVibrationDataGenerator.
    Returns:
        None
    """
    generator = MarinePumpVibrationDataGenerator(sample_rate=1000)
    normal_signal = generator.generate_vibration_signal(rpm=1750, duration=2.0)

    severities = ('mild', 'moderate', 'severe')
    for severity in severities:
        cav_signal = generator.add_cavitation_effect(normal_signal, sample_rate=1000, duration=2.0, severity='moderate', cavitation_start=1.0)
        generator.compare_signals(normal_signal, cav_signal, cavitation_start=1.0)

        test_data = pd.DataFrame({
            'time': np.arange(len(normal_signal)) / generator.sample_rate,
            'normal_signal': normal_signal,
            f'cavitation_{severity}': cav_signal
        })
        test_data.to_csv(f'test_cavitation_{severity}.csv', index=False)
    log.log_success("Cavitation effect tests passed!")