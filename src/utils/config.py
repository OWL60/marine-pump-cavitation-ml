"""Configuration utilities for loading and validating YAML settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


class Config:
    """Simple YAML-backed configuration object."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.data = data

    @classmethod
    def load_yaml(cls, path: str = "config.yaml") -> "Config":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        with config_path.open("r", encoding="utf-8") as file_obj:
            payload = yaml.safe_load(file_obj) or {}

        config = cls(payload)
        config.validate_config()
        return config

    def validate_config(self) -> None:
        required_top_level = ["data", "ml", "paths"]
        for section in required_top_level:
            if section not in self.data:
                raise ValueError(f"Missing required config section: {section}")

        data_keys = ["sample_rate", "shaft_freq", "duration", "n_samples", "train_test_split"]
        for key in data_keys:
            if key not in self.data["data"]:
                raise ValueError(f"Missing data config field: {key}")

        for path_key in ["data", "results", "models"]:
            if path_key not in self.data["paths"]:
                raise ValueError(f"Missing paths config field: {path_key}")

    def __getitem__(self, item: str) -> Dict[str, Any]:
        return self.data[item]
