"""Performance plot utilities."""

from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    """Save a confusion matrix plot."""

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm).plot(ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, save_path: str) -> None:
    """Save ROC curve plot."""

    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(metrics: Dict[str, float], save_path: str) -> None:
    """Save bar chart for selected metrics."""

    keys = ["accuracy", "precision", "recall", "f1", "auc_roc", "auc_pr"]
    values = [metrics[key] for key in keys]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(keys, values, color="tab:green")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Summary")
    for idx, value in enumerate(values):
        ax.text(idx, value + 0.02, f"{value:.2f}", ha="center", fontsize=9)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
