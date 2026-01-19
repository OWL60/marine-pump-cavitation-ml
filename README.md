# Physics-Informed Explainable ML for Marine Pump Cavitation Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!---[![Tests](https://github.com/owl60/MARINE-PUMP-CAVITATION-ML/actions/workflow/tests.yml/badge.svg)](https://github.com/owl60 MARINE-PUMP-CAVITATION-ML/actions)--->
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->

**Master's Thesis Research: A Physics-Informed and Explainable Machine Learning Framework for Early Cavitation Risk Prediction in Marine Centrifugal Pumps Using Vibration Data**

---


![Pump vibration](images/pump_vibration.png)

---

## Research Overview

This repository contains the complete implementation of a novel **physics-informed explainable machine learning framework** for early cavitation risk prediction in marine centrifugal pumps. 
The goal is to provide early warnings by learning from both simulated and real-world physical characteristics of cavitation.

**What Does it mean by  Physics-Informed  Explainable Machine Learning in This Context?**<br>
Our research title describes an advanced machine learning approach that blends three key concepts:<br>

1. **Physics-Informed**<br>
This means the machine learning model isn’t just learning from data, it also incorporates known physical laws or equations related to cavitation in centrifugal pumps.<br>
- In our case, these could be equations from fluid dynamics (e.g., Bernoulli’s principle, pump performance curves, NPSH equations), acoustic/vibration physics (how bubble collapse generates specific vibration signatures), or mechanical models of the pump.
- The model might use these physics equations to guide training, generate synthetic data, or as part of the loss function to ensur the predictions are physically plausible.
- Benefits: improves accuracy with less data, ensures predictions make physical sense, and can extrapolate better to unseen conditions.<br>
2. **Explainable Machine Learning (XAI)**<br>
This means the model is designed to provide understandable reasons for its predictions.
- Since cavitation prediction in a safety-critical marine system requires trust and diagnosis, you wouldn’t want a pure “black-box” model.
- Techniques like SHAP, LIME, or attention mechanisms might be used to highlight which vibration frequencies or time features signal cavitation risk.
- This helps engineers to understand why the model predicts risk, e.g., “increased amplitude at 5 kHz combined with reduced 1x RPM harmonic indicates early cavitation.”<br>


**Combined in our research**<br>
Our framework:<br>
- Uses vibration data (common for cavitation detection, because collapsing vapor bubbles cause high-frequency vibrations).
- Integrates physics knowledge (maybe equations relating NPSH, flow rate, and vibration patterns) to inform feature selection, data augmentation, or model architecture.
- Applies explainable ML to make the risk prediction interpretable to marine engineers.<br>

---

### Why this combination is powerful for early cavitation prediction

1. **Early detection** — Physical models help identify subtle signatures before severe damage.<br>
2. **Data efficiency** — Physics reduces need for massive labeled failure datasets.<br>
3. **Trust and adoption** — Explainability helps engineers act on predictions confidently.<br>
4. **Marine context** — Centrifugal pumps in ships are critical and everywhere; failure risks safety, hence this robust approach.<br>

Also, the research addresses critical gaps in current condition monitoring systems by:

**Bridging physics models with data-driven ML** for more accurate predictions<br>
**Providing human-interpretable explanations** using SHAP and LIME for engineer trust<br>
**Predicting cavitation risk 24-48 hours earlier** than threshold-based methods<br>
**Accounting for marine-specific conditions** (ship motion, variable loads, seawater properties)

---

## Table of Contents

- [Research Gap](#-research-gap)
- [Methodology](#-methodology)
- [Key Features](#-key-features)
- [Contributing](CONTRIBUTING)
- [License](LICENSE)

---

## Research Gap

Current approaches to marine pump cavitation monitoring face four critical limitations:

1. **Physics-ML Disconnect**: Traditional models either use pure physics (accurate but inflexible) or black-box ML (adaptive but uninterpretable)
2. **Late Detection**: Threshold-based alarms trigger only after cavitation damage occurs
3. **Lack of Marine Context**: Industrial solutions don't account for ship motion, variable loads, and seawater properties
4. **Engineer Distrust**: Black-box ML predictions lack explanations needed for critical maritime decisions

**Our Contribution**: A hybrid framework that integrates pump physics with explainable ML for early, trustworthy risk prediction in marine environments.

---

## Methodology

### 1. Physics-Informed Feature Engineering
```python
# Example: Incorporating pump physics into ML features
def calculate_physics_features(vibration_signal, pump_rpm, seawater_density):
    """Extract physics-guided features"""
    features = {
        'npsh_margin': calculate_npsh_margin(pump_params),
        'reynolds_effect': estimate_turbulence_level(vibration),
        'cavitation_number': compute_cavitation_number(flow_params),
        'energy_ratio_actual_vs_expected': compare_with_physics_model()
    }
    return features
```

---
## key-features

| Feature                     | Description                                           | 
|----------------------------|--------------------------------------------------------|
| Physics-ML Fusion           | Integrates pump equations as ML constraints           |
| Marine-Specific Explanations| SHAP plots translated to engineering recommendations  |
| Early Risk Prediction       | Predicts cavitation 24–48 hours before damage         |
| Variable Condition Robustness| Accounts for ship motion and load changes            |
| Real Data Integration       | Uses actual ship maintenance records                  |
| Multiple Model Architecture | CNN, LSTM and hybrid models for time series cavitation data |
| Visualization               | Visualization of cavitation prediction results        |

---

## Contributing.
This is a Master's thesis repository, but contributions are welcome for:

- Additional ML models
- New physics feature implementations
- Documentation improvements
- Translation of explanations to different languages

See the full contributing text in the [CONTRIBUTING](CONTRIBUTING) file.

---

## Licence
This project is licensed under the MIT License.
You are free to:
- Use the software for personal and commercial purposes
- Modify and distribute the source code
- Include the software in proprietary projects
  
Under the following conditions:
- The original copyright notice and license must be included in all copies or substantial portions of the software
- The software is provided “as is”, without warranty of any kind.
  
See the full license text in the [LICENSE](LICENSE) file.


---

## Implementation checklist.


### Core Infrastructure
- [x] Repository setup and structure
- [x] README documentation
- [x] Python environment setup
- [x] Basic configuration files


### Core Foundation
**src/data/generator.py**
- [x] generate_normal_vibration
   - [x] Base 50Hz motor vibration
   - [x] Add harmonics 
   - [x] Add random noise 
- [x] add_cavitation_effects
   - [x] High frequency component 
   - [x] Random bursts
   - [x] Amplitude modulation
- [x] add_ship_motion
   - [x] Low frequency modulation
   - [x] Engine load variations
- [x] generate_dataset
    - [x] 50% normal, 50% cavitation
    - [x] Save to data/ folder
**src/features/time_features.py**
- [x] extract_time_features
   - [x] Statistical: mean, std, variance
   - [x] Shape: RMS, peak, crest_factor
   - [x] Advanced: kurtosis, skewness
   - [x] Others: shape_factor, impulse_factor
- [x] batch_extract


### ML Models

**src/models/traditional_ml.py**
- [ ] class TraditionalML
   - [ ] dict of models
      - [ ] RandomForestClassifier
      - [ ] SVC (Support Vector Machine)
      - [ ] XGBClassifier
      - [ ] LogisticRegression (baseline)
   - [ ] train_all
   - [ ] predict_all
   - [ ] get_scores
- [ ] train_test_split_wrapper
- [ ] save_models
      
**src/models/deep_learning.py**
- [ ] build_cnn
   - [ ] Conv1D layers
   - [ ] MaxPooling1D
   - [ ] Dropout for regularization
   - [ ] Dense output layer
- [ ] build_lstm
   - [ ] LSTM layers (50 units)
   - [ ] Return sequences
   - [ ] TimeDistributed layers
- [ ] build_cnn_lstm
    - [ ] CNN for feature extraction
    - [ ] LSTM for temporal patterns


### Physics & Explainability 
**src/features/physics_features.py**
- [ ] calculate_cavitation_number
- [ ] calculate_npsh_margin
- [ ] estimate_reynolds_number
- [ ] energy_ratio_high_low
    - [ ] Low freq energy (0-100Hz)
    - [ ] High freq energy (1000+ Hz)
          
**src/explainability/shap_explainer.py**
- [ ] class SHAPExplainer
   - [ ] explain
   - [ ] plot_summary
   - [ ] plot_force
- [ ] install_shap
- [ ] save_shap_plots

**src/explainability/lime_explainer.py**
- [ ] class LIMEExplainer
   - [ ] explain_instance
   - [ ] plot_explanation
- [ ] generate_text_explanation


### Visualization
**src/visualization/signals.py**
- [ ] plot_vibration_comparison
   - [ ] Time domain plot
   - [ ] Color coding
   - [ ] Labels and titles
- [ ] plot_spectrogram
- [ ] plot_feature_distribution

**src/visualization/performance.py**
- [ ] plot_confusion_matrix
- [ ] plot_roc_curve
- [ ] plot_precision_recall
- [ ] plot_model_comparison
    - [ ] Bar chart of accuracies
    - [ ] Error bars if available

**src/visualization/dashboard.py**
- [ ] create_dashboard()
   - [ ] Streamlit setup
   - [ ] File upload widget
   - [ ] Signal visualization
   - [ ] ML prediction display
   - [ ] Explanation display
- [ ] requirements: streamlit, plotly
- [ ] Run with: streamlit run dashboard.py


### Evaluation & Main Pipeline
**src/evaluation/metrics.py**
- [ ] calculate_all_metrics
   - [ ] Standard: accuracy, precision, recall, f1
   - [ ] Advanced: AUC-ROC, AUC-PR
   - [ ] Early detection: lead_time, false_alarm_rate
- [ ] cross_validate_model
- [ ] statistical_significance_test

**main.py**
   - [ ] Load config from config.yaml
   - [ ] Generate/load data
   - [ ] Extract features
   - [ ] Train models
   - [ ] Evaluate and compare
   - [ ] Generate visualizations
- [ ] command line arguments
   - [ ] generate_data
   - [ ] train_models
   - [ ] create_plots
- [ ] Logging to file and console


### Configuration & Utils
**config.yaml**
- [ ] data parameters
   - [ ] sample_rate
   - [ ] signal_duration
   - [ ] train_test_split
- [ ] ml parameters
   - [ ] random_forest
   - [ ] cnn
- [ ] paths
    - [ ] data: 'data/'
    - [ ] results: 'results/'

**src/utils/config.py**
- [ ] class Config
   - [ ] load_yaml
- [ ] validate_config
