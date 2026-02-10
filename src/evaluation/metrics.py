"""Evaluation metrics helpers for classification tasks."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score


def calculate_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute standard and advanced binary classification metrics."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_prob)),
        "auc_pr": float(average_precision_score(y_true, y_prob)),
    }


def cross_validate_model(model, x_data: np.ndarray, y_data: np.ndarray, cv: int = 5) -> Tuple[float, float]:
    """Return mean and std cross-validation accuracy."""

    scores = cross_val_score(model, x_data, y_data, cv=cv, scoring="accuracy")
    return float(scores.mean()), float(scores.std())


def statistical_significance_test(scores_a: np.ndarray, scores_b: np.ndarray) -> Dict[str, float]:
    """Simple paired comparison summary (mean difference and win rate)."""

    if len(scores_a) != len(scores_b):
        raise ValueError("Score arrays must have the same length")

    diff = scores_a - scores_b
    return {
        "mean_difference": float(np.mean(diff)),
        "std_difference": float(np.std(diff)),
        "win_rate": float(np.mean(diff > 0)),
    }
