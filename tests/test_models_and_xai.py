import numpy as np

from src.explainability.xai import ExplainabilityToolkit
from src.models.cavitation_models import CavitationModelSuite


def test_model_suite_benchmark_output(cavitation_vibration, frequency_feature_extractor):
    cav_signal, base_signal, _, _ = cavitation_vibration
    extractor = frequency_feature_extractor

    x = extractor.batch_extract_frequency_features(
        [base_signal, cav_signal, base_signal * 1.1, cav_signal * 0.9], verbose=False
    )
    y = np.array([0, 1, 0, 1])
    results = CavitationModelSuite(random_state=0).train_and_benchmark(x, y, test_size=0.5)

    assert "random_forest" in results
    assert 0 <= results["random_forest"].accuracy <= 1


def test_xai_fallback_pipeline(cavitation_vibration, frequency_feature_extractor):
    cav_signal, base_signal, _, _ = cavitation_vibration
    extractor = frequency_feature_extractor

    x = extractor.batch_extract_frequency_features(
        [base_signal, cav_signal, base_signal * 1.2, cav_signal * 0.8, base_signal * 0.95, cav_signal * 1.05],
        verbose=False,
    )
    y = np.array([0, 1, 0, 1, 0, 1])
    suite = CavitationModelSuite(random_state=42)
    model = suite.models["logistic_regression"]
    model.fit(x, y)

    toolkit = ExplainabilityToolkit(model)
    shap_out = toolkit.shap_summary(x, x[:2])
    lime_out = toolkit.lime_explain_instance(x, x[0])

    assert "importance" in shap_out
    assert len(lime_out) > 0
