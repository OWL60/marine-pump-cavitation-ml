"""Signal visualization utilities."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_vibration_comparison(normal_signal: np.ndarray, cav_signal: np.ndarray, save_path: str) -> None:
    """Plot normal vs cavitation vibration in time-domain."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(normal_signal, color="tab:blue", linewidth=1)
    axes[0].set_title("Normal Pump Vibration")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)

    axes[1].plot(cav_signal, color="tab:red", linewidth=1)
    axes[1].set_title("Cavitation Pump Vibration")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
