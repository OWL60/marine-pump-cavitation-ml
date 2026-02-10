"""Quick benchmark script for features and models."""

from __future__ import annotations

from time import perf_counter
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.generator import MarinePumpVibrationDataGenerator
from src.features.frequency_features import FrequencyFeatureExtractor
from src.features.time_features import batch_extract
from src.features.time_frequency_features import extract_time_frequency_features
from src.models.cavitation_models import CavitationModelSuite


def run_benchmarks(n_samples: int = 200) -> dict:
    gen = MarinePumpVibrationDataGenerator(sample_rate=10_000, duration=1.0)
    extractor = FrequencyFeatureExtractor(gen)

    signals = []
    labels = []
    for i in range(n_samples):
        base = gen.generate_vibration_signal()
        if i % 2 == 0:
            signals.append(base)
            labels.append(0)
        else:
            signals.append(gen.add_cavitation_effect(base, severity="moderate"))
            labels.append(1)

    t0 = perf_counter()
    _ = batch_extract(signals, verbose=False)
    time_feat_s = perf_counter() - t0

    t0 = perf_counter()
    freq = extractor.batch_extract_frequency_features(signals, verbose=False)
    freq_feat_s = perf_counter() - t0

    t0 = perf_counter()
    tf = np.array(
        [
            list(extract_time_frequency_features(sig, gen.sample_rate).values())
            for sig in signals
        ]
    )
    tf_feat_s = perf_counter() - t0

    x = np.hstack([freq, tf])
    y = np.array(labels)

    model_results = CavitationModelSuite().train_and_benchmark(x, y)

    return {
        "time_features_s": time_feat_s,
        "frequency_features_s": freq_feat_s,
        "time_frequency_features_s": tf_feat_s,
        "model_results": {k: vars(v) for k, v in model_results.items()},
    }


if __name__ == "__main__":
    out = run_benchmarks(n_samples=100)
    print(out)
