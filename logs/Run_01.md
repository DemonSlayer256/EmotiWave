# Run 1 — Basic Differential Entropy Baseline

## Objective

Establish a baseline EEG emotion recognition system using only Differential Entropy (DE) features.

## Features

### Channels

```text
14 EEG Channels
````

### Frequency Bands

```text
Delta  (1–4 Hz)
Theta  (4–8 Hz)
Alpha  (8–13 Hz)
Beta   (13–30 Hz)
Gamma  (30–45 Hz)
```

### Feature

```text
Differential Entropy (DE)
```

Formula:

```text
DE = 0.5 log(2πeσ²)
```

## Feature Count

```text
70
```

## Model

```text
XGBoost (Default Parameters)
```

## Result

```text
Accuracy = 54.2%
```

## Observation

Performance was only slightly above random guessing.

The extracted features captured coarse spectral information but ignored temporal dynamics and hemispheric asymmetry.

## Conclusion

Need richer feature representations.
