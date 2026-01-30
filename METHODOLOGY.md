## Research Methodology

This study proposes a **Physics-Informed Explainable Machine Learning (PI-XAI) framework** for early detection/prediction of cavitation in marine centrifugal pumps. The methodology combines vibration signal analysis, physics-based modeling, feature engineering, and interpretable machine learning techniques. The workflow is divided into the following key steps:

### 1. Data Collection and Signal Generation

* **Synthetic Signal Generation:**
  Instead of real-world pump data, this study uses synthetically generated vibration signals to simulate normal and cavitation scenarios. The synthetic signals replicate the key characteristics of marine pump operation:

  * Baseline motor vibration at fundamental shaft frequencies (50 Hz)
  * Harmonics of the primary frequency
  * Random noise to replicate measurement imperfections
  * Marine scenarios (including engine load)
  * Cavitation effects: high-frequency bursts, amplitude modulation
  * Low-frequency modulation to simulate ship motion and engine load variations

This step ensures the dataset covers **both normal and cavitation scenarios**, crucial for machine learning training.

---

### 2. Feature Extraction

* **Time-Domain Features:**
  Captures statistical and shape characteristics of the vibration signal:

  * Statistical: mean, standard deviation, variance
  * Shape-based: RMS, peak, peak-to-peak, crest factor, impulse factor, kurtosis, skewness

* **Frequency-Domain Features:**
  Captures spectral properties of the signal through **FFT and power spectrum analysis**:

  * Peak frequency (`peak_frequency_hz`)
  * Spectral centroid (`spectral_centroid_hz`)
  * RMS frequency (`rms_frequency_hz`)
  * Total power (`total_power`)
  * Energy ratios in defined frequency bands (e.g., low/high frequency energy ratios)

* **Physics-Informed Features:**
  Integrates pump physics and fluid dynamics into the feature set to improve model generalization and interpretability:

  * NPSH margin
  * Cavitation number
  * Reynolds number
  * Energy ratio relative to expected physics-based values

---

### 3. Machine Learning Modeling

* **Traditional ML Models:** Random Forest, Support Vector Machine (SVM), XGBoost, Logistic Regression (baseline)

* **Deep Learning Models:**

  * CNN for automatic feature extraction from raw vibration signals
  * LSTM for capturing temporal dependencies
  * Hybrid CNN-LSTM for combined spatial-temporal feature learning

* **Training and Evaluation:**

  * Dataset split into training and testing sets (50% normal, 50% cavitation)
  * Models evaluated using accuracy, precision, recall, F1-score, and AUC-ROC metrics
  * Cross-validation applied to ensure model robustness

---

### 4. Explainable AI

* **SHAP (SHapley Additive exPlanations):** Quantifies the contribution of each feature to the model’s prediction
* **LIME (Local Interpretable Model-Agnostic Explanations):** Provides local interpretability for individual predictions
* **Outcome:** Engineers can understand which vibration frequencies or time-domain features indicate cavitation, increasing trust in model predictions

---

### 5. Visualization and Reporting

* **Time-Domain Plots:** Highlight transient spikes, impulses, and anomalies
* **Frequency-Domain Plots:** Show power spectrum, dominant frequencies, and energy distribution
* **Feature Distributions:** Compare normal vs cavitation cases
* **Dashboard Integration:** Streamlit dashboard for interactive visualization and real-time predictions

---

### 6. Evaluation & Validation

* **Performance Metrics:** Accuracy, precision, recall, F1-score, AUC-ROC
* **Early Detection Capability:** Measured as lead-time for predicting cavitation events
* **Robustness Testing:** Evaluated under variable operating conditions and noise levels
* **Statistical Significance:** Confirmed with hypothesis tests to ensure reliability of predictions

---

**Summary Workflow:**

1. Generate or acquire vibration data →
2. Extract time-domain, frequency-domain, and physics-informed features →
3. Train ML models (traditional + deep learning) →
4. Apply XAI techniques for interpretability →
5. Visualize results and evaluate early detection performance

This methodology ensures a **data-driven, physically meaningful, and interpretable framework** for early cavitation prediction in marine pumps.

---
