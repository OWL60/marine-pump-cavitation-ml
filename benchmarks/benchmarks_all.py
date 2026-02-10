"""Benchmark suite with a simple user interface for comparing feature extraction pipelines."""

import argparse
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data.generator import MarinePumpVibrationDataGenerator
from src.features.frequency_features import FrequencyFeatureExtractor
from src.features.time_features import extract_time_features
from src.features.time_frequency_features import WaveletTimeFrequencyFeatureExtractor


@dataclass
class BenchmarkResult:
    name: str
    runtime_ms: float
    throughput_signals_per_sec: float


def _benchmark(name: str, fn: Callable[[], None], n_runs: int = 10) -> BenchmarkResult:
    start = time.perf_counter()
    for _ in range(n_runs):
        fn()
    elapsed = time.perf_counter() - start
    runtime_ms = (elapsed / n_runs) * 1000
    throughput = n_runs / elapsed if elapsed > 0 else 0.0
    return BenchmarkResult(name=name, runtime_ms=runtime_ms, throughput_signals_per_sec=throughput)


def run_benchmarks(num_signals: int, sample_rate: int, duration: float) -> List[BenchmarkResult]:
    generator = MarinePumpVibrationDataGenerator(sample_rate=sample_rate, duration=duration)
    signals = [generator.generate_vibration_signal() for _ in range(num_signals)]

    freq_extractor = FrequencyFeatureExtractor(generator)
    wavelet_extractor = WaveletTimeFrequencyFeatureExtractor(sample_rate=sample_rate)

    def _run_frequency_features() -> None:
        feature_matrix = freq_extractor.batch_extract_frequency_features(
            signals, verbose=False
        )
        if feature_matrix.ndim != 2 or feature_matrix.shape[1] == 0:
            raise ValueError(
                "Frequency feature extraction produced an empty feature matrix. "
                "Increase --sample-rate and/or --duration so valid frequency features can be computed."
            )

    # Validate frequency extraction succeeds before recording benchmark timings.
    _run_frequency_features()

    return [
        _benchmark("Time features", lambda: [extract_time_features(s) for s in signals]),
        _benchmark("Frequency features", _run_frequency_features),
        _benchmark("Wavelet time-frequency features", lambda: wavelet_extractor.batch_extract(signals)),
    ]


def display_results(results: List[BenchmarkResult]) -> None:
    print("\n=== Feature Extraction Benchmark (UI-style summary) ===")
    print(f"{'Pipeline':35} {'Avg Runtime (ms)':>18} {'Throughput (runs/s)':>22}")
    print("-" * 80)
    for result in results:
        print(f"{result.name:35} {result.runtime_ms:18.3f} {result.throughput_signals_per_sec:22.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark feature extraction pipelines")
    parser.add_argument("--num-signals", type=int, default=20, help="Number of signals per benchmark run")
    parser.add_argument("--sample-rate", type=int, default=10000, help="Signal sample rate")
    parser.add_argument("--duration", type=float, default=1.0, help="Signal duration in seconds (must be >= 1.0 for current generator)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_benchmarks(
        num_signals=args.num_signals,
        sample_rate=args.sample_rate,
        duration=args.duration,
    )
    display_results(results)


if __name__ == "__main__":
    main()
