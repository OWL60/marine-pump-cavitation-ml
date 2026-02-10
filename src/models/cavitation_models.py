"""Model training and benchmark helpers for cavitation classification."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass
class ModelBenchmark:
    name: str
    accuracy: float
    f1: float
    roc_auc: float
    train_time_s: float


class CavitationModelSuite:
    """Train and benchmark a small suite of classifiers."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            "logistic_regression": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
                ]
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            ),
            "svm_rbf": Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("clf", SVC(kernel="rbf", probability=True, random_state=random_state)),
                ]
            ),
        }

    def train_and_benchmark(
        self,
        x: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.25,
    ) -> Dict[str, ModelBenchmark]:
        """Split data, train each model, and return benchmark metrics."""
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            stratify=y,
            random_state=self.random_state,
        )

        results: Dict[str, ModelBenchmark] = {}
        for name, model in self.models.items():
            start = perf_counter()
            model.fit(x_train, y_train)
            train_time = perf_counter() - start

            pred = model.predict(x_test)
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(x_test)[:, 1]
            else:
                decision = model.decision_function(x_test)
                proba = 1.0 / (1.0 + np.exp(-decision))

            results[name] = ModelBenchmark(
                name=name,
                accuracy=float(accuracy_score(y_test, pred)),
                f1=float(f1_score(y_test, pred)),
                roc_auc=float(roc_auc_score(y_test, proba)),
                train_time_s=float(train_time),
            )
        return results
