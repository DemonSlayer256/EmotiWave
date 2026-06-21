# Run 4 — Tuned XGBoost
```

## Motivation

Improve classification through hyperparameter optimization.

## Added

RandomizedSearchCV

### Best Parameters

```python
{
    'subsample': 0.9,
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.01,
    'colsample_bytree': 0.8
}
```

## Result

```text
Accuracy = 60.2%
```

## Observation

Performance slightly decreased.

## Conclusion

Model optimization cannot compensate for feature limitations.
