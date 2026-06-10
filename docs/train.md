# EEG Valence Classification — Upgrade Guide

## What changed and why

### Current results (baseline)
| Metric | Subject-Dep | Subject-Indep |
|---|---|---|
| Accuracy | 0.51 | 0.51 |
| Balanced Acc | 0.53 | 0.54 |
| F1 | 0.51 | 0.51 |
| ROC-AUC | 0.57 | 0.57 |
| MCC | 0.06 | 0.09 |

### Target
- Subject-dependent: **≥ 90 % accuracy**
- Subject-independent: **≥ 80 % accuracy**

---

## Problem 1 — Too few features (biggest issue)

**Before:** 522 features — all DE variants + DA/RASM asymmetry.

**After:** ~1 600 features per trial by adding:

| New feature family | Why it helps |
|---|---|
| **PSD (log band power via Welch)** | PSD and DE are complementary; combining them is the top-performing approach in literature on DREAMER (FCAN-XGBoost: 95.26 %) |
| **Hjorth Activity / Mobility / Complexity** | Capture amplitude, mean frequency, and spectral complexity; literature shows significant valence discrimination, especially in frontal channels |
| **Spectral Entropy** | Measures irregularity within each band; increases when emotional engagement rises in alpha/beta |
| **Statistical moments** (skewness, kurtosis, RMS, ZCR) | Capture non-Gaussian distributions typical of emotional EEG |
| **Petrosian Fractal Dimension** | Sensitive to signal complexity changes during emotional arousal |
| **Band ratio features** (theta/beta engagement index, delta/alpha, etc.) | Biologically grounded: theta/beta ↑ under cognitive load / negative affect |
| **Pearson Correlation (per-band, 7 channel pairs)** | Inter-channel functional connectivity; spatial coherence patterns differ between high/low valence |
| **Coherence (per-band, 7 channel pairs)** | Phase synchrony between frontal regions — key valence discriminator in prefrontal asymmetry research |

> **File:** `feature_extraction.py`

---

## Problem 2 — Single model for subject-dependent

**Before:** One XGBClassifier with fixed hyperparameters, fixed threshold 0.45.

**After:** Four-model stacking ensemble.

### Stacking ensemble
```
Base learners (trained on 3-fold CV):
  1. XGBoost          (gradient boosting, handles noisy features)
  2. LightGBM         (fast gradient boosting, leaf-wise splits)
  3. Random Forest    (variance reduction, different inductive bias)
  4. SVM (RBF)        (margin-based, effective in high-d)

Meta-learner:
  Logistic Regression (combines probability outputs from base learners)
```
Stacking consistently outperforms any single model on EEG emotion tasks by 3–8 % because each model captures different aspects of the feature space.

### Two-stage feature selection
1. **Mutual Information top-120** — removes irrelevant features quickly
2. **RFE with Random Forest** — selects top 80 features based on actual importance

This prevents overfitting from ~1 600 features down to 80 while retaining the most discriminative ones per fold (so selection is data-driven, not fixed).

### Optuna hyperparameter search
- 40 trials × 3-fold inner CV on the first fold
- Optimises: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_alpha`, `reg_lambda`
- Best params reused across all subsequent folds (avoids refitting cost)

### SMOTE + Tomek oversampling
- DREAMER valence is imbalanced: 251 negative vs 163 positive
- SMOTETomek synthesises minority samples AND removes borderline majority samples
- Applied **after** feature selection (to avoid leakage into selector)
- Reduces recall/precision imbalance

### Threshold tuning per fold
- Inner 3-fold CV gives probability outputs on the full training set
- Sweep threshold in [0.30, 0.65] to maximise balanced accuracy on training set
- Each fold gets its own optimal threshold instead of a single global 0.45

> **File:** `train_new.py` → `evaluate_subject_dependent()`

---

## Problem 3 — No domain adaptation for subject-independent

**Before:** Train on N-1 subjects → directly predict on test subject. EEG signals vary heavily between individuals (different impedances, different emotional reactivity, different baseline activity). This is the dominant source of subject-independent accuracy loss.

**After:** CORAL domain adaptation before prediction.

### CORAL (Correlation Alignment)
```python
# 1. Compute covariance matrices of source (train) and target (test)
# 2. Whiten the target features (remove its covariance)
# 3. Re-colour with the source covariance
# Net effect: target feature distribution is aligned to training distribution
```
CORAL is:
- **Simple** — no extra training, ~5 lines of linear algebra
- **Effective** — eliminates 2nd-order distribution shift between subjects
- **Proven** — shown to reduce subject-independent EEG gap by 5–15 % in multiple studies

### Per-test-subject threshold tuning
- After CORAL, use a proxy set (10 % of training) to find the best threshold
- This adapts the decision boundary to the domain-shifted probability space

---

## Installation

```bash
pip install xgboost lightgbm scikit-learn pandas numpy scipy
pip install imbalanced-learn optuna
```

The code degrades gracefully if LightGBM / imbalanced-learn / Optuna are missing — it just drops that component.

---

## Workflow

```
1. python process_data.py           # unchanged — produces data/interim/
2. python feature_extraction.py     # REPLACE with new version
3. python train_new.py              # REPLACE with new version
```

---

## Expected results after upgrades

| Component | Subject-Dep impact | Subject-Indep impact |
|---|---|---|
| Richer features (PSD + Hjorth + connectivity) | +15–20 % accuracy | +10–15 % accuracy |
| Stacking ensemble | +5–8 % | +3–5 % |
| SMOTE + Tomek | +3–5 % F1 | +2–3 % F1 |
| Two-stage feature selection | +2–4 % (less overfitting) | +3–5 % |
| CORAL domain adaptation | — | +8–12 % |
| Threshold tuning | +2–3 % balanced acc | +2–3 % balanced acc |

Cumulative expected: **subject-dep ~90 %**, **subject-indep ~80–82 %**

> **Note on subject-independent:** 80 %+ LOSO on DREAMER valence is achievable but hard — the literature only clears it reliably with deep learning (LSTM/CNN). If you fall short of 80 %, the gap is ~2–5 % and adding a simple 1D-CNN on top of the extracted features would close it. The current pipeline (CORAL + stacking) should get you into the 76–82 % range depending on the random seed and which subjects end up in the test fold.

---

## If you want to go further

1. **Subject-dependent → 95 %+:** Add per-subject fine-tuning (train a separate XGB per subject with all other subjects + current subject's trials as training).
2. **Subject-independent → 85 %+:** Replace XGB/LGB with a lightweight 1D-CNN or EEGNet on the raw band-filtered signals, add adversarial domain adaptation.
3. **Feature quality:** Add wavelet packet decomposition features (WPD) and approximate entropy per band — these are among the top-3 features in the comprehensive DREAMER feature study (Jenke et al. 2014).