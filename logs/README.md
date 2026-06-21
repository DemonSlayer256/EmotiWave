# Experiment Logs

This directory contains the complete chronological history of all major experiments performed during the development of the DREAMER EEG Emotion Recognition project.

The objective of these logs is to provide a reproducible record of:

- Feature engineering decisions
- Model architecture changes
- Hyperparameter optimization attempts
- Evaluation protocol corrections
- Dataset processing modifications
- Accuracy improvements and regressions
- Research observations and conclusions

---

# Log Structure

Each run is documented independently and includes:

- Motivation
- Pipeline configuration
- Features used
- Models used
- Evaluation methodology
- Accuracy metrics
- Observations
- Conclusions
- Impact on subsequent experiments

---

# Experiment Timeline

| Run | Description | Status |
|-------|-------------|---------|
| Run 01 | Basic Differential Entropy Features | Completed |
| Run 02 | Windowed DE + Differential Asymmetry | Completed |
| Run 03 | Baseline Corrected Differential Entropy | Completed |
| Run 04 | Tuned XGBoost | Completed |
| Run 05 | Initial Classical ML Pipeline (ANOVA + RBF SVM) | Completed |
| Run 06 | PCA-Based Feature Space | Completed |
| Run 07 | Standardized PCA + Dynamic Thresholding | Completed |
| Run 08 | Corrected SVM Pipeline with MI Feature Selection | Completed |
| Run 09 | Fixed Classical ML Pipeline (935 Features + CORAL + RBF SVM) | Current Best Classical Baseline |

---

# Current Best Pipeline

Feature Extraction:

```text
935 Features
```

Includes:

- Differential Entropy (DE)
- PSD
- Higuchi Fractal Dimension
- Hjorth Parameters
- Statistical Moments
- Relative Energy
- Band Ratios
- Spectral Entropy
- Differential Asymmetry
- Rational Asymmetry
- Functional Connectivity

Evaluation:

```text
Subject-Dependent:
    Stratified 5-Fold CV

Subject-Independent:
    LOSO + CORAL
```

Classifier:

```text
Mutual Information Feature Selection
      ↓
SelectKBest(k=60)
      ↓
StandardScaler
      ↓
RBF SVM
```

---

# Best Results So Far

## Binary Valence Classification

### Subject-Dependent

```text
Accuracy            : 58.51%
Balanced Accuracy   : 48.70%
F1 Score            : 17.87%
ROC-AUC             : 48.61%
MCC                 : -0.026
```

### Subject-Independent

```text
Accuracy            : 51.45%
Balanced Accuracy   : 53.82%
F1 Score            : 49.84%
ROC-AUC             : 55.87%
MCC                 : 0.074
```

---

## Four-Class Emotion Classification

### Subject-Dependent

```text
Accuracy            : 35.04%
Balanced Accuracy   : 24.01%
F1 Macro            : 15.83%
MCC                 : -0.045
```

### Subject-Independent

```text
Accuracy            : 26.81%
Balanced Accuracy   : 31.25%
F1 Macro            : 23.24%
MCC                 : 0.074
```

---

# Key Research Findings

## 1. Baseline Correction Produced the Largest Gain

The most significant improvement across all experiments came from:

```text
Stimulus Feature − Baseline Feature
```

rather than model tuning.

Observed improvement:

```text
54.2% → 61.5%
```

during early Differential Entropy experiments.

---

## 2. More Features ≠ Better Performance

Several runs demonstrated that increasing feature dimensionality without proper selection caused severe overfitting.

Feature selection became essential due to:

```text
414 samples
vs
935 features
```

---

## 3. Subject-Dependent Learning Is Data Limited

Each subject provides only:

```text
18 trials
```

which leads to:

```text
≈14 training samples
```

per fold.

This severely limits model capacity and increases variance.

---

## 4. Subject-Independent Learning Remains Challenging

Even after:

- CORAL Alignment
- PCA
- Mutual Information Selection
- SVM Optimization

cross-subject generalization remained difficult because EEG patterns differ substantially between individuals.

---

## 5. Classical ML Ceiling Reached

Current evidence suggests classical machine learning on DREAMER typically saturates around:

```text
60–68% Accuracy
```

for binary emotion classification.

Further improvements likely require:

- EEGNet
- CNN-LSTM
- Attention Models
- Transformer Architectures
- Graph Neural Networks

---

# Files

```text
logs/
│
├── README.md               ← This file
│
├── Run_01.md
├── Run_02.md
├── Run_03.md
├── Run_04.md
├── Run_05.md
├── Run_06.md
├── Run_07.md
├── Run_08.md
├── Run_09.md
```

---

# Reproducibility

All reported results were generated using:

```text
Python 3.x
NumPy
SciPy
Pandas
Scikit-Learn
XGBoost
LightGBM
Optuna
```

Dataset:

```text
DREAMER
```

Evaluation:

```text
Subject-Dependent:
    Stratified K-Fold

Subject-Independent:
    Leave-One-Subject-Out (LOSO)

Domain Adaptation:
    CORAL
```

---

**Last Updated:** Run 09  
**Project Status:** Classical ML Baseline Complete