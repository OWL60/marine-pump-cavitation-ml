"""Tests for cavitation explainability helpers."""

from src.visualization.explainability import explain_cavitation_prediction


def test_predicts_cavitation_with_explanations():
    time_features = {"crest_factor": 3.3, "kurtosis": 5.1}
    freq_features = {
        "cavitation_indicator": 0.2,
        "energy_ratio_high": 0.35,
        "energy_ratio_ultra_high": 0.2,
        "spectral_bandwidth_hz": 1800,
    }

    result = explain_cavitation_prediction(time_features, freq_features)

    assert result.cavitation_predicted is True
    assert result.risk_score > 0
    assert len(result.reasons) >= 2


def test_no_cavitation_when_rules_not_triggered():
    time_features = {"crest_factor": 1.2, "kurtosis": 2.5}
    freq_features = {
        "cavitation_indicator": 0.01,
        "energy_ratio_high": 0.1,
        "energy_ratio_ultra_high": 0.05,
        "spectral_bandwidth_hz": 400,
    }

    result = explain_cavitation_prediction(time_features, freq_features)

    assert result.cavitation_predicted is False
    assert result.risk_score == 0
    assert result.reasons == []
