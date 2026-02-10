"""End-to-end pipeline for marine pump cavitation prediction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from src.data import MarinePumpVibrationDataGenerator
from src.evaluation import calculate_all_metrics, cross_validate_model
from src.features.frequency_features import FrequencyFeatureExtractor
from src.features.time_features import batch_extract as batch_time_extract
from src.utils import Config
from src.visualization import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_roc_curve,
    plot_vibration_comparison,
)
from utils import log


def _ensure_dirs(config: Config) -> None:
    for key in ["data", "results", "models"]:
        Path(config["paths"][key]).mkdir(parents=True, exist_ok=True)


def build_feature_matrix(generator: MarinePumpVibrationDataGenerator, x_raw: np.ndarray) -> np.ndarray:
    """Combine time-domain and frequency-domain features into one matrix."""

    time_features = batch_time_extract(x_raw, verbose=False)
    freq_extractor = FrequencyFeatureExtractor(generator)
    freq_features = freq_extractor.batch_extract_frequency_features(x_raw, verbose=False)
    return np.hstack([time_features, freq_features])


def generate_data(config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data and return features and labels."""

    data_cfg = config["data"]
    generator = MarinePumpVibrationDataGenerator(
        sample_rate=data_cfg["sample_rate"],
        shaft_freq=data_cfg["shaft_freq"],
        duration=data_cfg["duration"],
    )

    x_raw, _, y, _ = generator.generate_dataset(
        n_samples=data_cfg["n_samples"],
        include_marine_conditions=data_cfg["include_marine_conditions"],
        save_to_disk=False,
    )

    x_features = build_feature_matrix(generator, x_raw)

    data_dir = Path(config["paths"]["data"])
    np.save(data_dir / "x_raw.npy", x_raw)
    np.save(data_dir / "x_features.npy", x_features)
    np.save(data_dir / "y.npy", y)

    log.log_success(f"Saved generated data to {data_dir}")
    return x_features, y


def train_models(config: Config, x_features: np.ndarray, y: np.ndarray) -> dict:
    """Train and evaluate a random forest model."""

    split = config["data"]["train_test_split"]
    x_train, x_test, y_train, y_test = train_test_split(
        x_features,
        y,
        test_size=split,
        random_state=config["ml"]["random_state"],
        stratify=y,
    )

    rf_cfg = config["ml"]["random_forest"]
    model = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_split=rf_cfg["min_samples_split"],
        random_state=config["ml"]["random_state"],
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    metrics = calculate_all_metrics(y_test, y_pred, y_prob)
    cv_mean, cv_std = cross_validate_model(model, x_features, y)
    metrics["cv_accuracy_mean"] = cv_mean
    metrics["cv_accuracy_std"] = cv_std

    model_path = Path(config["paths"]["models"]) / "random_forest.joblib"
    joblib.dump(model, model_path)

    result_payload = {
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }

    out_path = Path(config["paths"]["results"]) / "metrics.json"
    with out_path.open("w", encoding="utf-8") as file_obj:
        json.dump(metrics, file_obj, indent=2)

    log.log_success(f"Model trained and saved to {model_path}")
    return result_payload


def create_plots(config: Config, training_payload: dict) -> None:
    """Create and save pipeline visualizations."""

    results_dir = Path(config["paths"]["results"])
    plot_confusion_matrix(
        training_payload["y_test"],
        training_payload["y_pred"],
        str(results_dir / "confusion_matrix.png"),
    )
    plot_roc_curve(
        training_payload["y_test"],
        training_payload["y_prob"],
        str(results_dir / "roc_curve.png"),
    )
    plot_model_comparison(
        training_payload["metrics"],
        str(results_dir / "model_comparison.png"),
    )

    generator = MarinePumpVibrationDataGenerator(
        sample_rate=config["data"]["sample_rate"],
        shaft_freq=config["data"]["shaft_freq"],
        duration=config["data"]["duration"],
    )
    normal = generator.generate_vibration_signal()
    cav = generator.add_cavitation_effect(normal, severity="severe")
    plot_vibration_comparison(normal, cav, str(results_dir / "signal_comparison.png"))

    log.log_success(f"Saved plots to {results_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Marine pump cavitation complete app")
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--mode",
        choices=["generate_data", "train_models", "create_plots", "run_all"],
        default="run_all",
        help="Pipeline stage to execute",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.load_yaml(args.config)
    _ensure_dirs(cfg)

    if args.mode == "generate_data":
        generate_data(cfg)
    elif args.mode == "train_models":
        x_arr, y_arr = generate_data(cfg)
        train_models(cfg, x_arr, y_arr)
    elif args.mode == "create_plots":
        x_arr, y_arr = generate_data(cfg)
        payload = train_models(cfg, x_arr, y_arr)
        create_plots(cfg, payload)
    else:
        x_arr, y_arr = generate_data(cfg)
        payload = train_models(cfg, x_arr, y_arr)
        create_plots(cfg, payload)
