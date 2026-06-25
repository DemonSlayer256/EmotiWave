# Run 11 — Valence and Arousal Evaluation with XGBoost

## Objective

Extend the leak-proof XGBoost evaluation pipeline to train and evaluate both **Valence** and **Arousal** binary classifiers using the same windowed DREAMER feature representation.

The evaluation retains all safeguards introduced in Run 9:

* StratifiedGroupKFold for subject-dependent evaluation
* Leave-One-Subject-Out (LOSO) for subject-independent evaluation
* CORAL domain adaptation during LOSO testing
* Mutual Information feature selection (Top-60 features)
* StandardScaler normalization
* Removal of `window_id` from the feature space
* XGBoost classifier with tuned hyperparameters

---

## Dataset

### Valence

* Samples: 414
* Features: 938
* Class Distribution:

  * Class 0: 251
  * Class 1: 163

### Arousal

* Samples: 414
* Features: 938
* Class Distribution:

  * Class 0: 233
  * Class 1: 181

---

## Model Configuration

### Feature Selection

```python
SelectKBest(mutual_info_classif, k=60)
```

### XGBoost Parameters

```python
n_estimators = 300
max_depth = 4
learning_rate = 0.01
subsample = 0.9
colsample_bytree = 0.8
```

---

# Results

## Arousal Classification

### Subject-Dependent Evaluation

| Metric            | Mean   | Std    |
| ----------------- | ------ | ------ |
| Accuracy          | 0.6014 | 0.1383 |
| Balanced Accuracy | 0.5225 | 0.1216 |
| F1 Score          | 0.3149 | 0.2830 |
| AUC               | 0.5297 | 0.1999 |
| MCC               | 0.0375 | 0.2501 |

### Subject-Independent Evaluation (LOSO + CORAL)

| Metric            | Mean   | Std    |
| ----------------- | ------ | ------ |
| Accuracy          | 0.5738 | 0.1145 |
| Balanced Accuracy | 0.5083 | 0.1037 |
| F1 Score          | 0.3067 | 0.1775 |
| AUC               | 0.5417 | 0.1450 |
| MCC               | 0.0257 | 0.2278 |

---

## Valence Classification

### Subject-Dependent Evaluation

| Metric            | Mean   | Std    |
| ----------------- | ------ | ------ |
| Accuracy          | 0.5877 | 0.1450 |
| Balanced Accuracy | 0.5167 | 0.1331 |
| F1 Score          | 0.2625 | 0.2401 |
| AUC               | 0.5152 | 0.2377 |
| MCC               | 0.0274 | 0.2690 |

### Subject-Independent Evaluation (LOSO + CORAL)

| Metric            | Mean   | Std    |
| ----------------- | ------ | ------ |
| Accuracy          | 0.5821 | 0.1307 |
| Balanced Accuracy | 0.5486 | 0.1149 |
| F1 Score          | 0.3737 | 0.1732 |
| AUC               | 0.5858 | 0.1339 |
| MCC               | 0.0979 | 0.2561 |

---

# Comparison

| Task    | Subject-Dependent Accuracy | Subject-Independent Accuracy |
| ------- | -------------------------- | ---------------------------- |
| Arousal | 60.14%                     | 57.38%                       |
| Valence | 58.77%                     | 58.21%                       |

### Best Result

**Valence Subject-Independent**

* Accuracy: **58.21%**
* Balanced Accuracy: **54.86%**
* AUC: **58.58%**

### Observation

Unexpectedly, the LOSO valence model performs nearly as well as the subject-dependent model. This suggests that the extracted features capture patterns that generalize across subjects better for valence than for arousal.

---

# Analysis

Several observations emerged from this experiment:

1. Performance remains only marginally above chance level for both tasks.
2. Balanced accuracy values remain close to 0.50, indicating difficulty learning minority-class patterns.
3. CORAL domain alignment provides only limited gains under the current feature representation.
4. Mutual-information feature selection combined with XGBoost does not significantly outperform previous classical pipelines.
5. Valence appears easier to generalize across subjects than arousal.
6. High variance between subjects indicates strong inter-subject EEG variability.

---

# Conclusion

Run 11 evaluated the complete leak-proof XGBoost pipeline on both valence and arousal classification tasks.

The results indicate that:

* Subject-dependent performance reaches approximately 60% accuracy.
* Subject-independent performance remains between 57–58%.
* Domain adaptation and feature selection alone are insufficient to overcome inter-subject variability.
* Further improvements will likely require richer feature engineering, temporal modeling, subject adaptation techniques, or deep learning architectures.

This run establishes a reliable benchmark for future experiments involving advanced feature extraction and neural approaches.
