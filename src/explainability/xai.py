"""SHAP/LIME helpers with graceful fallback when optional deps are missing."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.inspection import permutation_importance


class ExplainabilityToolkit:
    """Compute feature attributions via SHAP/LIME when available."""

    def __init__(self, model, feature_names: Optional[list[str]] = None):
        self.model = model
        self.feature_names = feature_names

    def shap_summary(self, x_background: np.ndarray, x_eval: np.ndarray) -> Dict[str, np.ndarray]:
        """Return mean absolute SHAP values; fallback to permutation importance."""
        try:
            import shap

            explainer = shap.Explainer(self.model, x_background)
            values = explainer(x_eval)
            shap_values = np.asarray(values.values)
            mean_abs = np.mean(np.abs(shap_values), axis=0)
            return {"method": "shap", "importance": mean_abs}
        except Exception:
            dummy_y = self.model.predict(x_background)
            perm = permutation_importance(
                self.model,
                x_background,
                dummy_y,
                n_repeats=5,
                random_state=42,
            )
            return {"method": "permutation_fallback", "importance": perm.importances_mean}

    def lime_explain_instance(self, x_train: np.ndarray, x_instance: np.ndarray) -> Dict[str, float]:
        """Explain a single row with LIME when installed, fallback to local gradient proxy."""
        try:
            from lime.lime_tabular import LimeTabularExplainer

            names = self.feature_names or [f"feature_{i}" for i in range(x_train.shape[1])]
            explainer = LimeTabularExplainer(
                x_train,
                feature_names=names,
                class_names=["normal", "cavitation"],
                mode="classification",
            )
            explanation = explainer.explain_instance(
                x_instance,
                self.model.predict_proba,
                num_features=min(10, x_train.shape[1]),
            )
            return {k: float(v) for k, v in explanation.as_list()}
        except Exception:
            eps = 1e-4
            base = self.model.predict_proba(x_instance.reshape(1, -1))[0, 1]
            grads = {}
            names = self.feature_names or [f"feature_{i}" for i in range(x_train.shape[1])]
            for idx, name in enumerate(names):
                perturbed = x_instance.copy()
                perturbed[idx] += eps
                new_pred = self.model.predict_proba(perturbed.reshape(1, -1))[0, 1]
                grads[name] = float((new_pred - base) / eps)
            return grads
