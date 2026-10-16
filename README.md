# Heart Disease Risk Prediction (Healthcare / Medical) — Internal Project 1

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-F7931E.svg)](https://scikit-learn.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready, clinical decision support system that predicts patient risk of coronary heart disease using the UCI Cleveland Heart Disease dataset. This project implements an end-to-end machine learning pipeline: automated data ingestion, exploratory data analysis (EDA), rigorous cross-validation comparison across classical and gradient boosted models, Bayesian hyperparameter optimization with Optuna, comprehensive model evaluation on a stratified hold-out test set, and an interactive Streamlit diagnostic web application.

---

## 🏥 Application Domain

**Cardiovascular Disease Risk Stratification & Clinical Decision Support (CDSS)**

Coronary artery disease (CAD) is the leading cause of mortality worldwide. Early detection of clinically significant coronary stenosis (>50% diameter narrowing) allows physicians to initiate preventive pharmacotherapy, recommend lifestyle modifications, or schedule confirmatory stress angiography. In clinical risk screening, minimizing **False Negatives (high Recall / Sensitivity)** is paramount to avoid missing high-risk patients.

---

## 📊 Dataset & Attribution

- **Dataset Name:** [UCI Heart Disease Dataset (Cleveland Clinic Foundation)](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Primary Source:** [UCI Machine Learning Repository - Cleveland Data](https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data)
- **Principal Investigators:** 
  - Robert Detrano, M.D., Ph.D. (V.A. Medical Center, Long Beach and Cleveland Clinic Foundation)
- **Target Formulation:** Binarized angiographic disease status:
  - `0`: Normal / No significant narrowing (<50% stenosis)
  - `1`: Heart disease present (>50% diameter narrowing in at least one major vessel)

### Clinical Features:
| Feature | Type | Description |
|---|---|---|
| `age` | Continuous | Age in years |
| `sex` | Categorical | Sex (1 = male, 0 = female) |
| `cp` | Categorical | Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic) |
| `trestbps` | Continuous | Resting blood pressure (mm Hg on admission) |
| `chol` | Continuous | Serum cholesterol in mg/dl |
| `fbs` | Categorical | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| `restecg` | Categorical | Resting ECG (0: normal, 1: ST-T wave abnormality, 2: LV hypertrophy) |
| `thalach` | Continuous | Maximum heart rate achieved during exercise |
| `exang` | Categorical | Exercise-induced angina (1 = yes, 0 = no) |
| `oldpeak` | Continuous | ST depression induced by exercise relative to rest |
| `slope` | Categorical | Slope of peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping) |
| `ca` | Categorical | Number of major vessels (0–3) colored by fluoroscopy (missing imputed) |
| `thal` | Categorical | Thalassemia status (3: normal, 6: fixed defect, 7: reversible defect) |

---

## 📈 Benchmark & Results

### 1. 5-Fold Stratified Cross-Validation Comparison

During baseline model selection, three distinct architectures were evaluated across 5 stratified folds on the training partition (80% of data, N=242):

| Model | Mean ROC-AUC | Mean F1-Score | Mean Accuracy | Mean Precision | Mean Recall |
|---|:---:|:---:|:---:|:---:|:---:|
| **Logistic Regression (Baseline)** | **0.9025 ± 0.014** | **0.8245 ± 0.007** | **84.71% ± 1.00%** | **87.68% ± 5.42%** | **78.34% ± 4.59%** |
| Random Forest (100 trees) | 0.8968 ± 0.029 | 0.7948 ± 0.047 | 81.81% ± 3.07% | 81.73% ± 4.05% | 78.26% ± 9.81% |
| XGBoost (Gradient Boosted Trees) | 0.8671 ± 0.023 | 0.7594 ± 0.026 | 78.10% ± 2.15% | 76.78% ± 4.47% | 75.61% ± 5.64% |

*Logistic Regression outperformed tree-based ensembles on this tabular dataset, providing superior calibration, generalization without overfitting, and clinical interpretability.*

### 2. Final Tuned Hyperparameters (Optuna — 40 Trials)

Using Tree-structured Parzen Estimator (TPE) Bayesian optimization with 5-fold CV:

| Parameter | Final Tuned Value | Search Range / Options | Description |
|---|:---:|:---:|---|
| **Classifier** | `LogisticRegression` | — | Linear classifier with LASSO regularization |
| `l1_ratio` | `1.0` | `[0.0, 0.5, 1.0]` | 1.0 enforces pure L1 LASSO feature sparsity |
| `C` | `0.381607` | `[1e-3, 50.0]` (log-uniform) | Inverse regularization strength |
| `solver` | `saga` | Fixed | Stochastic Average Gradient with L1 support |
| `max_iter` | `2000` | Fixed | Convergence limit |
| `random_state` | `42` | Fixed | Deterministic reproducibility |

### 3. Held-Out Test Set Performance (N=61, 20% Stratified)

| Metric | Score | Clinical Relevance |
|---|:---:|---|
| **ROC-AUC** | **0.9643** | Exceptional class separability across all discrimination thresholds |
| **Recall (Sensitivity)** | **92.86%** | **26 out of 28** heart disease cases correctly caught (only 2 false negatives) |
| **Accuracy** | **88.52%** | High overall diagnostic concordance |
| **Precision** | **83.87%** | Low false alarm rate in outpatient referral |
| **F1-Score** | **0.8814** | Harmonic mean of precision and recall |

#### Confusion Matrix:
```
                 Predicted Negative (0)    Predicted Positive (1)
Actual Negative:          28                         5
Actual Positive:           2                        26
```

---

## 🔬 Diagnostic Visualizations

Generated automatically and saved in the [`reports/`](reports/) directory:

- **Target Class Balance:** [`reports/class_balance.png`](reports/class_balance.png)
- **Correlation Matrix Heatmap:** [`reports/correlation_matrix.png`](reports/correlation_matrix.png)
- **Continuous Feature Distributions:** [`reports/feature_distributions.png`](reports/feature_distributions.png)
- **ROC Curve:** [`reports/roc_curve.png`](reports/roc_curve.png)
- **Confusion Matrix:** [`reports/confusion_matrix.png`](reports/confusion_matrix.png)
- **Feature Importance:** [`reports/feature_importance.png`](reports/feature_importance.png)

---

## 📁 Project Structure

```text
heart-disease-risk-prediction/
├── .gitignore                      # Git ignore rules (venv, cache, checkpoints)
├── README.md                       # Complete technical & clinical documentation
├── requirements.txt                # Pinned production dependencies
├── app.py                          # Interactive Streamlit clinical web application
├── data/
│   ├── raw_cleveland.data          # Downloaded raw UCI dataset
│   └── heart_disease_cleaned.csv   # Cleaned, standardized, binarized dataset
├── models/
│   └── heart_disease_model.joblib  # Serialized end-to-end inference pipeline
├── reports/
│   ├── metrics.json                # JSON record of CV, Optuna params, and test metrics
│   ├── class_balance.png           # Target balance visualization
│   ├── correlation_matrix.png      # Feature correlation heatmap
│   ├── feature_distributions.png   # Stratified clinical distributions
│   ├── categorical_features.png    # Breakdown of categorical predictors
│   ├── roc_curve.png               # ROC curve on held-out test partition
│   ├── confusion_matrix.png        # Confusion matrix on test partition
│   └── feature_importance.png      # Key clinical predictor weights
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Ingestion, cleaning, target binarization
│   ├── eda.py                      # Publication-quality exploratory visualization generator
│   └── train.py                    # Preprocessing pipeline, CV comparison, Optuna tuning, evaluation
└── tests/
    ├── __init__.py
    └── test_pipeline.py            # Unit and integration pytest suite
```

---

## 🚀 Installation & Usage

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/krutharyan22/heart-disease-risk-prediction.git
cd heart-disease-risk-prediction

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Run Data Download & EDA

```bash
# Download dataset from UCI and clean
python src/data_loader.py

# Generate all exploratory plots in reports/
python src/eda.py
```

### 3. Run Training, Cross-Validation & Optuna Tuning

```bash
# Executes 5-fold CV comparison, Optuna 40-trial tuning, test evaluation, and model serialization
python src/train.py
```

### 4. Run Automated Tests

```bash
PYTHONPATH=. pytest tests/
```

### 5. Launch the Streamlit Web Application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`. Use the sidebar presets to immediately evaluate test cases or enter custom patient vitals to receive instantaneous risk probabilities.

---

## ⚖️ License & Disclaimer

- **License:** MIT License.
- **Disclaimer:** This software is intended for research, educational, and decision-support demonstration purposes only. It is not approved as a medical device for standalone clinical diagnosis.
