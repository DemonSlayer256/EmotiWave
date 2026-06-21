# Run 3 — Baseline Corrected Differential Entropy
```

## Motivation

DREAMER provides baseline EEG before every stimulus.

## Change

Previous:

```text
Feature = Stimulus DE
```

New:

```text
Feature = Stimulus DE − Baseline DE
```

Applied to all channels and bands.

## Result

```text
Accuracy = 61.5%
```

Improvement:

```text
+7.3%
```

## Observation

Largest improvement observed throughout feature-engineering phase.

## Key Finding

Baseline correction contributes more than model tuning.
