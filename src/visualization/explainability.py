"""Simple explainability helpers for cavitation risk predictions."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ExplainabilityResult:
    """Container for prediction and human-readable explanation."""

    cavitation_predicted: bool
    risk_score: float
    reasons: List[str]


def _threshold_checks(time_features: Dict[str, float], freq_features: Dict[str, float]) -> List[str]:
    """Evaluate domain-inspired thresholds and return triggered reasons."""
    reasons: List[str] = []

    cav_indicator = float(freq_features.get("cavitation_indicator", 0.0))
    if cav_indicator > 0.08:
        reasons.append(
            f"High cavitation indicator ({cav_indicator:.3f}) suggests broadband high-frequency bubble-collapse energy."
        )

    hf_ratio = float(freq_features.get("energy_ratio_high", 0.0)) + float(
        freq_features.get("energy_ratio_ultra_high", 0.0)
    )
    if hf_ratio > 0.45:
        reasons.append(
            f"Elevated high-frequency energy ratio ({hf_ratio:.3f}) is consistent with cavitation vibration signatures."
        )

    crest_factor = float(time_features.get("crest_factor", 0.0))
    if crest_factor > 2.5:
        reasons.append(
            f"High crest factor ({crest_factor:.3f}) indicates impulsive spikes typical of vapor bubble collapse."
        )

    kurtosis = float(time_features.get("kurtosis", 0.0))
    if kurtosis > 4.0:
        reasons.append(
            f"High kurtosis ({kurtosis:.3f}) indicates a heavy-tailed, shock-like vibration distribution."
        )

    spectral_bandwidth = float(freq_features.get("spectral_bandwidth_hz", 0.0))
    if spectral_bandwidth > 1200:
        reasons.append(
            f"Wide spectral bandwidth ({spectral_bandwidth:.1f} Hz) suggests turbulent broadband excitation."
        )

    return reasons


def explain_cavitation_prediction(
    time_features: Dict[str, float], freq_features: Dict[str, float]
) -> ExplainabilityResult:
    """Predict cavitation with a lightweight heuristic and return reasons.

    The risk score is normalized by the number of activated rules.
    """
    reasons = _threshold_checks(time_features, freq_features)
    max_rules = 5
    risk_score = min(len(reasons) / max_rules, 1.0)
    cavitation_predicted = len(reasons) >= 2

    if cavitation_predicted and not reasons:
        reasons = ["Model confidence exceeded decision threshold."]

    return ExplainabilityResult(
        cavitation_predicted=cavitation_predicted,
        risk_score=risk_score,
        reasons=reasons,
    )
