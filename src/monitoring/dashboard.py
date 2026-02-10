"""Minimal monitoring dashboard helpers for cavitation risk."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List

import numpy as np


@dataclass
class RiskMonitor:
    """Rolling risk monitor for online predictions."""

    window: int = 100
    history: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def update(self, risk_score: float) -> Dict[str, float]:
        self.history.append(float(risk_score))
        arr = np.array(self.history, dtype=float)
        return {
            "risk_mean": float(np.mean(arr)),
            "risk_std": float(np.std(arr)),
            "risk_latest": float(arr[-1]),
            "risk_p95": float(np.percentile(arr, 95)),
            "count": float(len(arr)),
        }

    def status(self, threshold: float = 0.7) -> str:
        if not self.history:
            return "no_data"
        return "alert" if self.history[-1] >= threshold else "normal"


def build_dashboard_payload(
    timestamps: List[str],
    risk_scores: List[float],
    threshold: float = 0.7,
) -> Dict[str, object]:
    """Build serializable payload for streamlit/monitoring layer."""
    if len(timestamps) != len(risk_scores):
        raise ValueError("timestamps and risk_scores lengths must match")

    alerts = [score >= threshold for score in risk_scores]
    return {
        "series": [{"timestamp": t, "risk": float(r), "alert": bool(a)} for t, r, a in zip(timestamps, risk_scores, alerts)],
        "threshold": float(threshold),
        "n_alerts": int(np.sum(alerts)),
    }
