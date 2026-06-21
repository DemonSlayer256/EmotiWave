# Run 9 — Production Classical ML Pipeline
```

## Feature Set

```text
935 Features
```

Including:

* Differential Entropy
* PSD
* HFD
* Hjorth
* Statistical Moments
* Band Ratios
* Relative Energy
* Spectral Entropy
* DA
* RASM
* Connectivity

## Binary Classification

### Subject-Dependent

```text
Accuracy               58.51%
Balanced Accuracy      48.70%
F1                     17.87%
ROC-AUC                48.61%
MCC                   -0.0261
```

### Subject-Independent

```text
Accuracy               51.45%
Balanced Accuracy      53.82%
F1                     49.84%
ROC-AUC                55.87%
MCC                    0.0740
```

## Four-Class Classification

### Subject-Dependent

```text
Accuracy               35.04%
Balanced Accuracy      24.01%
Macro F1               15.83%
```

### Subject-Independent

```text
Accuracy               26.81%
Balanced Accuracy      31.25%
Macro F1               23.24%
```

## Observation

Classical ML performance plateaued around:

```text
Balanced Accuracy ≈ 60%
ROC-AUC ≈ 63%
```

