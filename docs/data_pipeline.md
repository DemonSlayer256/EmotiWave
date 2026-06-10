# EEG Emotion Recognition using DREAMER Dataset

## Project Overview

This project implements an EEG-based emotion recognition pipeline using the DREAMER dataset. The objective is to classify emotional states from EEG recordings using handcrafted Differential Entropy (DE) features and machine learning models.

The project was developed incrementally with a research-oriented approach where each design decision, feature engineering modification, and model improvement was evaluated experimentally before being adopted.

The pipeline follows:

```text
DREAMER.mat
    ↓
Dataset Exploration
    ↓
process_data.py
    ↓
data/interim/
    ↓
feature_extraction.py
    ↓
data/processed/
    ↓
XGBoost
    ↓
Model Evaluation
    ↓
Feature Importance Analysis
```

---

# Directory Structure

```text
data/
├── raw/
│   ├── DREAMER.mat
│   └── metadata.json
│
├── interim/
│   ├── metadata.json
│   ├── baseline EEG files
│   ├── stimulus EEG files
│   └── ...
│
├── processed/
│   ├── features_valence.csv
│   ├── features_arousal.csv
│   └── features_dominance.csv
│
backend/
frontend/
models/
logs/
outputs/

scripts/
├── explore_data.py
├── process_data.py
├── feature_extraction.py
└── train.py
```

---

# Dataset

Dataset Used:

DREAMER Dataset

Dataset Statistics:

```text
Subjects              : 23
Trials per Subject    : 18
Total Trials          : 414
EEG Channels          : 14
Sampling Frequency    : 128 Hz
```

EEG Channels:

```text
AF3 F7 F3 FC5
T7 P7 O1 O2
P8 T8 FC6 F4
F8 AF4
```

Emotional Dimensions:

```text
Valence
Arousal
Dominance
```

Ratings are provided on a scale:

```text
1 → 5
```

---

# Dataset Exploration

## explore_data.py

Purpose:

* Inspect DREAMER MATLAB structure
* Verify field names
* Validate EEG storage format
* Examine emotional ratings
* Understand baseline and stimulus recordings

Discovered Structure:

```text
DREAMER
│
├── Dataset Metadata
├── EEG Information
└── Data
     └── Subject
          ├── Age
          ├── Gender
          ├── EEG
          │    ├── baseline
          │    └── stimuli
          ├── ScoreValence
          ├── ScoreArousal
          └── ScoreDominance
```

This exploration stage prevented implementation errors and established the foundation for all subsequent processing.

---

# Data Processing

## process_data.py

Purpose:

Convert DREAMER.mat into a structured dataset suitable for machine learning.

Tasks Performed:

1. Extract subject metadata
2. Extract baseline EEG
3. Extract stimulus EEG
4. Extract emotional ratings
5. Generate binary labels
6. Save EEG recordings as NumPy arrays
7. Generate metadata index

Output:

```text
data/interim/
```

---

# Label Generation Strategy

The DREAMER dataset provides ratings between:

```text
1 → 5
```

## Initial Approach

```python
label = int(score >= 3)
```

Observed Distribution:

```text
Valence   : {1: 253, 0: 161}
Arousal   : {1: 300, 0: 114}
Dominance : {1: 319, 0: 95}
```

Problem:

Significant class imbalance.

---

## Final Approach

```python
label = int(score > 3)
```

Interpretation:

```text
1,2,3 → Low
4,5   → High
```

Final Distribution:

```text
Valence   : {0: 251, 1: 163}
Arousal   : {0: 233, 1: 181}
Dominance : {0: 215, 1: 199}
```

Reasons:

* Rating 3 represents neutral emotion
* Better class balance
* Reduced classifier bias
* Consistent with emotion-recognition literature

---

# Metadata Structure

Example:

```json
{
    "subject_id": 1,
    "trial_id": 1,

    "baseline_file": "S01_T01_BASE.npy",
    "stimulus_file": "S01_T01_STIM.npy",

    "age": 22,
    "gender": "male",

    "valence": 4,
    "arousal": 3,
    "dominance": 2,

    "valence_class": 1,
    "arousal_class": 0,
    "dominance_class": 0
}
```

---

# Feature Engineering Evolution

The feature engineering pipeline underwent several iterations.

Each experiment was evaluated independently.

---

## Run 1 — Basic Differential Entropy

Features:

```text
14 Channels × 5 Frequency Bands
```

Bands:

```text
Delta  : 1–4 Hz
Theta  : 4–8 Hz
Alpha  : 8–13 Hz
Beta   : 13–30 Hz
Gamma  : 30–45 Hz
```

Feature:

```text
Differential Entropy (DE)
```

Formula:

```text
DE = 0.5 log(2πeσ²)
```

Feature Count:

```text
70
```

Result:

```text
Accuracy = 54.2%
```

Observation:

Only slightly above random chance.

---

## Run 2 — Windowed DE + Differential Asymmetry

Added:

### Windowed DE

Instead of computing DE across the entire trial:

```text
60-second EEG
      ↓
2-second windows
      ↓
DE per window
      ↓
Statistical aggregation
```

Added:

```text
Mean
Std
```

---

### Differential Asymmetry (DA)

Pairs:

```text
AF3 ↔ AF4
F7  ↔ F8
F3  ↔ F4
FC5 ↔ FC6
T7  ↔ T8
P7  ↔ P8
```

Formula:

```text
DA = Left − Right
```

Result:

```text
Accuracy = 55.4%
```

Improvement:

```text
+1.2%
```

Observation:

Marginal gain.

---

## Run 3 — Baseline Corrected DE

Major design change.

DREAMER provides baseline EEG before every stimulus.

Instead of:

```text
Feature = Stimulus DE
```

we used:

```text
Feature = Stimulus DE − Baseline DE
```

Applied to every channel and band.

Result:

```text
Accuracy = 61.5%
```

Improvement:

```text
+7.3%
```

Largest improvement observed.

Key Finding:

Baseline correction contributes more than model tuning.

---

## Run 4 — Tuned XGBoost

Added:

```text
RandomizedSearchCV
```

Parameters tuned:

```text
n_estimators
max_depth
learning_rate
subsample
colsample_bytree
```

Best Parameters:

```python
{
    'subsample': 0.9,
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.01,
    'colsample_bytree': 0.8
}
```

Result:

```text
Accuracy = 60.2%
```

Observation:

Model optimization alone did not significantly improve performance.

---

# Current Feature Extraction Pipeline

The current feature extraction stage incorporates multiple feature families.

---

## 1. Baseline-Corrected Windowed DE

For every:

```text
Channel × Band
```

Computed:

```text
Mean
Std
Max
Min
Median
```

using:

```text
Stimulus DE − Baseline DE
```

---

## 2. Relative Band Energy

Formula:

```text
Band DE
──────────────
Total Band DE
```

Purpose:

Capture spectral energy distribution.

---

## 3. Frequency Band Ratios

Computed per channel:

```text
Alpha / Beta
Theta / Beta
Alpha / Gamma
```

Purpose:

Capture cognitive-state relationships between frequency bands.

---

## 4. Differential Asymmetry (DA)

Formula:

```text
DA = Left − Right
```

Applied to all hemispheric pairs and bands.

---

## 5. Rational Asymmetry (RASM)

Formula:

```text
RASM = Left / Right
```

Purpose:

Capture proportional hemispheric differences.

Widely used in:

```text
DREAMER
DEAP
SEED
```

literature.

---

## 6. Subject-wise Normalization

Applied after feature extraction.

Formula:

```text
z = (x − μ) / σ
```

Performed independently for each subject.

Purpose:

* Remove subject-specific offsets
* Reduce inter-subject variability
* Improve subject-dependent learning

---

# Current Feature Statistics

Current Dataset:

```text
Samples  : 414
Features : 522
```

Generated from:

```text
Statistical DE Features
Relative Energy
Band Ratios
DA
RASM
```

---

# Training Pipeline

## Classifier

```text
XGBoost Classifier
```

Reasons:

* Strong performance on tabular data
* Handles nonlinear relationships
* Robust to irrelevant features
* Widely used in EEG emotion-recognition literature

---

## Class Balancing

Applied:

```python
compute_sample_weight(
    class_weight="balanced"
)
```

Purpose:

Handle remaining class imbalance.

---

## Feature Selection

Applied:

```python
SelectKBest
```

Method:

```python
mutual_info_classif
```

Purpose:

Reduce noisy features.

---

# Evaluation Strategy

---

## Subject-Dependent Evaluation

Method:

```text
5-Fold Stratified Cross Validation
```

Metrics:

* Accuracy
* Balanced Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Matthews Correlation Coefficient (MCC)

Latest Results:

```text
Accuracy            0.5098 ± 0.0352
Balanced Accuracy   0.5320 ± 0.0279
Precision           0.4203 ± 0.0235
Recall              0.6377 ± 0.0608
F1                  0.5056 ± 0.0259
ROC-AUC             0.5745 ± 0.0556
MCC                 0.0636 ± 0.0547
```

---

## Subject-Independent Evaluation

Method:

```text
Leave-One-Subject-Out (LOSO)
```

Procedure:

```text
Train on 22 Subjects
Test on Remaining Subject
Repeat for all Subjects
```

Latest Results:

```text
Accuracy            0.5072 ± 0.1037
Balanced Accuracy   0.5423 ± 0.1074
Precision           0.4264 ± 0.1384
Recall              0.6840 ± 0.1852
F1                  0.5103 ± 0.1246
ROC-AUC             0.5671 ± 0.1272
MCC                 0.0886 ± 0.2146
```

---

# Most Important Features

Top Features Identified by XGBoost:

```text
F8_delta_relative
F3_gamma_mean
O1_theta_max
F4_alpha_mean
P8_gamma_std
F4_alpha_std
P8_gamma_median
O1_delta_std
O1_beta_std
F8_alpha_std
T8_gamma_std
P7_beta_median
O1_theta_relative
O2_beta_median
F3_beta_min
F4_delta_mean
T8_gamma_mean
T8_alpha_std
F7_gamma_relative
RASM_F7_F8_alpha
```

These features currently contribute most to classification performance.

---

# Key Research Findings

## Finding 1

Baseline correction produced the largest gain.

```text
54.2% → 61.5%
```

---

## Finding 2

Feature engineering contributes more than model tuning.

---

## Finding 3

Current handcrafted features are still insufficient for state-of-the-art performance.

Modern DREAMER research achieving:

```text
85%–95%
```

typically employs:

* CNNs
* CNN-LSTM
* EEGNet
* Graph Neural Networks
* Transformers
* Attention Mechanisms
* Connectivity Features

---

# Planned Future Work

## Phase 4

Feature Selection Refinement

```text
SHAP
Boruta
Mutual Information Ranking
```

---

## Phase 5

Additional EEG Features

```text
PSD
Band Power
Log Band Power
Hjorth Parameters
```

---

## Phase 6

Connectivity Features

```text
Pearson Correlation
Coherence
PLV
```

---

## Phase 7

Deep Learning Benchmarking

Models:

```text
EEGNet
CNN
CNN-LSTM
TCN
Transformer
```

Target:

```text
Subject-Dependent Accuracy > 85%
Subject-Independent Accuracy > 75%
```

---

# Current Status

Completed:

* Dataset Exploration
* Data Extraction
* Metadata Generation
* Label Engineering
* Baseline Processing
* Differential Entropy Features
* Asymmetry Features
* Subject-wise Normalization
* XGBoost Training
* Subject-Dependent Evaluation
* Subject-Independent Evaluation
* Feature Importance Analysis

Next Priority:

```text
Feature Selection
Connectivity Features
EEGNet Benchmark
```
