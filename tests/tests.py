import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import log
from src.data.generator import MarinePumpVibrationDataGenerator
from src.features import time_features

def test_generate_vibration_signal() -> None:
    """
    Test the generate_vibration_signal method.      
    Returns:
        None
    """
    log.log_info("Test generate_vibration_signal")
    generator = MarinePumpVibrationDataGenerator(sample_rate=10000)  
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
    log.log_success("-"*60)

def test_cavitation_effect() -> None:
    """
    Test the add_cavitation_effect method of MarinePumpVibrationDataGenerator.
    The case for marine pump that's why we use the marine scenario.
    Similarly we can simulate for in-land pump if we ommit the marine scenario.
    Also, we can generate data if the engine is running or otherwise, by toggling "including_engine_load".
    """
    log.log_info("Test cavitation effect")

    generator = MarinePumpVibrationDataGenerator(sample_rate=1000)
    normal_signal = generator.generate_vibration_signal(rpm=1750, duration=2.0)

    normal_signal_len = len(normal_signal)
    plot_sample = min(500, normal_signal_len)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6))
    fig.suptitle("Different Views of Pump Vibration Signal", fontsize=14)
    fig.text(0.5, -0.05,
        f"Length: {normal_signal_len} samples | Sample rate: {generator.sample_rate/1000:.1f} kHz | RMS: {np.sqrt(np.mean(normal_signal**2)):.3f}",
        ha='center', fontsize=9)
    axes = axes.ravel()
    
    generator.plot_signal(axes, normal_signal[-plot_sample:], pos=0)

    marine_scenarios = generator.generate_marine_scenarios(normal_signal)
    severities = ('mild', 'moderate', 'severe')

    for i, (severity, scenario_signal) in enumerate(zip(severities, marine_scenarios.values()), start=1):
        cav_signal = generator.add_cavitation_effect(scenario_signal, sample_rate=1000, severity=severity, cavitation_start=1.0)
        generator.compare_signals(normal_signal, cav_signal, cavitation_start=1.0)

        test_data = pd.DataFrame({
            'time': np.arange(normal_signal_len) / generator.sample_rate,
            'normal_signal': normal_signal,
            f'cavitation_{severity}': cav_signal
        })
        test_data.to_csv(f'test_cavitation_{severity}.csv', index=False)
        generator.plot_signal(axes, cav_signal[-plot_sample:], pos=i, title=f"cavitation_{severity}")
        
    plt.tight_layout()
    fig.savefig("images/pump_vibration.png", bbox_inches='tight')
    plt.show()
    log.log_success("Cavitation effect tests passed!")
    log.log_success("-"*60)


def test_feature_extraction():
    """
    Test the feature extraction function
    """
    log.log_info("Test feature_extraction")
    np.random.seed(42) 
    t = np.linspace(0, 1, 1000)
    normal_signal = 0.05 + np.sin(2 * np.pi * 50 * t) + 0.02 * np.random.randn(1000)

    cavitation_signal = normal_signal.copy()
    spike_indices = np.random.choice(1000, 50, replace=False)
    cavitation_signal[spike_indices] += 2.0 * np.random.randn(50)

    log.log_info("Testing single signal feature extraction...")
    features1 = time_features.extract_time_features(normal_signal)
    log.log_info(f"Extracted {len(features1)} features")
    log.log_info("Example features:")
    for i, (key, value) in enumerate(list(features1.items())[:5]):
        log.log_info(f"{key}: {value:.4f}")
    
    # Test batch extraction
    log.log_info("Testing batch feature extraction...")
    signals = [normal_signal, cavitation_signal, normal_signal * 2]
    feature_matrix = time_features.batch_extract(signals, verbose=False)
    log.log_info(f"Feature matrix shape: {feature_matrix.shape}")
    log.log_info(f"Features per signal: {feature_matrix.shape[1]}")
    
    # Test normalization
    log.log_info("Testing feature normalization...")
    normalized = time_features.normalize_features(feature_matrix, method='standard')
    log.log_info(f"Normalized mean: {np.mean(normalized, axis=0)[:3]}...")
    log.log_info(f"Normalized std: {np.std(normalized, axis=0)[:3]}...")
    
    # Test feature names
    log.log_info("Testing feature names...")
    feature_names = time_features.get_feature_names()
    log.log_info(f"Number of features: {len(feature_names)}")
    log.log_info(f"First 5 features: {feature_names[:5]}")
    
    log.log_success("All tests passed!")
    log.log_success("-"*60)
    return feature_matrix, feature_names


def test_generate_dataset():
    """
    Test for generating dataset.
    """
    log.log_info('Test generate dataset')
    generator = MarinePumpVibrationDataGenerator(sample_rate=10000)

    log.log_info("Generating test dataset (50 samples)...")
    X_raw, X_features, y, metadata = generator.generate_dataset(n_samples=50, include_marine_conditions=True, save_to_disk=True)

    log.log_info("Dataset generated successfully!")
    log.log_info(f"Raw signals shape: {X_raw.shape}")
    log.log_info(f"Features shape: {X_features.shape}")
    log.log_info(f"Labels: {sum(y==0)} normal, {sum(y==1)} cavitation")

    # Quick ML test
    log.log_info("Quick ML test with Random Forest...")

    X_train, X_test, y_train, y_test = train_test_split(X_features, y, train_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    log.log_info(f"{y_pred}")
    accuracy = accuracy_score(y_test, y_pred)

    log.log_info(f"Model accuracy: {accuracy:.2%}")
    log.log_info("Feature importance top 10:")
    
    # Get feature names
    feature_names = [
        'mean', 'std', 'rms', 'peak', 'crest_factor',
        'kurtosis', 'skewness', 'shape_factor', 'impulse_factor',
        'variance', 'mean_abs', 'energy'
    ]
    size = len(feature_names)
    importances = model.feature_importances_[:size]
    indices = np.argsort(importances)[::-1]

    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        log.log_info(f"{i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
        
    log.log_info("-"*60)
    log.log_info("TEST COMPLETE!")
    log.log_info("Your dataset generator is working!")
    log.log_info("-"*60)
