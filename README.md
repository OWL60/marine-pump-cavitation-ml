# Physics-Informed Explainable ML for Marine Pump Cavitation Prediction

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<!-- [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX) -->
<!-- [![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX) -->

**Master's Thesis Research: A Physics-Informed and Explainable Machine Learning Framework for Early Cavitation Risk Prediction in Marine Centrifugal Pumps Using Vibration Data**

---

## Research Overview

This repository contains the complete implementation of a novel **physics-informed explainable machine learning framework** for early cavitation risk prediction in marine centrifugal pumps. The research addresses critical gaps in current condition monitoring systems by:

**Bridging physics models with data-driven ML** for more accurate predictions<br>
**Providing human-interpretable explanations** using SHAP and LIME for engineer trust<br>
**Predicting cavitation risk 24-48 hours earlier** than threshold-based methods<br>
**Accounting for marine-specific conditions** (ship motion, variable loads, seawater properties)

<!-- <p align="center">
  <img src="results/figures/framework_overview.png" alt="Framework Overview" width="700">
  <br>
  <em>Physics-Informed Explainable ML Framework Architecture</em>
</p> -->

---

## Table of Contents

- [Research Gap](#-research-gap)
- [Methodology](#-methodology)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Repository Structure](#-repository-structure)
- [Results](#-results)
- [Publication](#-publication)
- [PhD Pathway](#-phd-pathway)
- [Citation](#-citation)
- [License](#-license)

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
