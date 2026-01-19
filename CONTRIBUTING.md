# Contributing to marine-centrifugal-pump-cavitation-ML

Thank you for your interest in contributing to this research-driven project.
This repository focuses on ML-based vibration analysis and cavitation prediction in marine centrifugal pumps.

We welcome contributions from:
- Marine engineers
- AI/ML engineers
- Signal processing specialists
- Control engineers
- Students & academics
---

## How You Can Contribute

You can contribute in several ways:

### 1. Improve Code
- Optimize vibration processing
- Improve features extraction
- Add physics-informed methods
- Improve model performance
- Fixing bugs

### 2. Add New Models
- Deep Learning (CNN, LSTM)
- Anomaly detection
- Hybrid physics + AI
- Explainable AI (SHAP / LIME)

### 3. Improve Documentation
- Fix typos
- Add clearer explanations
- Add diagrams, figures and drawings
- Improve reproducibility

### 4. Research Contributions
- Add references
- Suggest improvements
- Improve dataset handling
- Add validation techniques
---


## Setup Instructions 

1. Fork the repository
2. Clone your fork:

```bash
git clone https://github.com/your-username/MARINE-PUMP-CAVITATION-ML.git
```

3. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate # (linux/Mac)
venv\Scripts\activate # (Windows)
```

4. Install dependencies:
``` bash
cd MARINE-PUMP-CAVITATION-ML
pip install -r requirements.txt
```

## Running Tests
``` bash
cd MARINE-PUMP-CAVITATION-ML\tests 
python main.py # Run all tests.
```
By default the commands run all tests, excluding the slow tests, the following arguments can be used to select a particular behaviour during running of the tests<br>
- Example:
```bash
python main.py # This will run all tests
python main.py --file tests_vibration.py --slow  # This will run all tests that are slow inside tests_vibration.py
python main.py --file tests_vibration.py --function test_generate_dataset_land_only  # This will run test for particular function i.e test_generate_dataset_land_only function
```

All tests must pass!

---

## Code Style
- Follow PEP8
- Use miningful variable names
- Clear conventions (branch naming, commit meanigful messages)
- Add docstring to all functions and files
- Keep code modular
- Comment complex math/logic clearly
---

## ML Guidelines
When adding models
- Include training script
- Save models inside `/models`
- Document dataset used
- Report evaluation metrics
- Ensure reproducibility
---

## Development Workflow
1. **Make your changes**
   - Write code
   - Add tests for new functionality
   - Update documentation if necessary
   - Ensure all tests pass


2. **Pull Request Process:**
    - Create a new branch:
    ```bash
    git checkout -b feature-name
    ```
    - Commit changes clearly:
    ```bash
    git commit -m "feat: Add cavitation frequency feature extraction"
    ```
    - Push branch:
    ```bash
    git push origin feature-name
    ```
    - Pull request

- ***Quick reference:***
     - Fork -> Clone -> Create Branch -> Develop -> Test -> Pull request.


## Follow [Conventional Commits](https://conventionalcommits.org):
- feat: New feature
- fix: Bug fix
- docs: Documentation changes
- test: Adding or updating tests
- refactor: Code changes that neither fix bugs nor add features
- chore: Maintanance task
- style: Formatting, missing semicolons.
---

## Important Notes
- Do not upload sensitive/private data
- Keep dataset small or provide download link
- Cite source when adding research material
- Respect academic integrity
---

## Code of Conduct
Be respectful and constructive.<br>
This project support open scientific collaboration.

---

## Contact
For research collaboration or major contributions, open an issue first.

Thank you for supporting marine AI research.