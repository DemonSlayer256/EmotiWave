# Run 2 — Windowed DE + Differential Asymmetry
```

## Motivation

Single-trial DE averages discard temporal variations.

## Added

### Windowed DE

```text
60-second EEG
      ↓
2-second windows
      ↓
DE per window
      ↓
Mean + Std aggregation
```

### Differential Asymmetry (DA)

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

## Result

```text
Accuracy = 55.4%
```

Improvement:

```text
+1.2%
```

## Observation

Temporal statistics improved robustness slightly.

DA contributed useful information but gains remained marginal.

## Conclusion

Feature engineering alone was insufficient.
