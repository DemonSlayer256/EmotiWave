# DREAMER Dataset Structure

## Dataset Overview

The DREAMER dataset is an emotion recognition dataset containing EEG and ECG recordings collected from **23 subjects** while watching **18 emotional video clips**.

### Metadata

| Property          | Value  |
| ----------------- | ------ |
| Subjects          | 23     |
| Video Sequences   | 18     |
| EEG Sampling Rate | 128 Hz |
| ECG Sampling Rate | 256 Hz |
| EEG Channels      | 14     |

Total trial recordings:

```text
23 Subjects × 18 Videos = 414 Samples
```

---

## Top-Level Structure

```text
DREAMER
├── Data
├── EEG_SamplingRate
├── ECG_SamplingRate
├── EEG_Electrodes
├── noOfSubjects
├── noOfVideoSequences
├── Disclaimer
├── Provider
├── Version
└── Acknowledgement
```

The `Data` field contains all subject recordings and emotion ratings.

---

## Subject Structure

Each subject contains the following fields:

```text
Subject
├── Age
├── Gender
├── EEG
├── ECG
├── ScoreValence
├── ScoreArousal
└── ScoreDominance
```

---

## EEG Structure

The EEG field contains baseline and stimulus recordings.

```text
EEG
├── baseline
└── stimuli
```

Each category contains recordings for all 18 video trials.

```text
EEG
├── baseline
│   └── (18,1)
│       └── Trial EEG Signal
│
└── stimuli
    └── (18,1)
        └── Trial EEG Signal
```

### Signal Dimensions

#### Baseline EEG

```text
Shape: (7808, 14)
```

* 7808 time samples
* 14 EEG channels
* Approximately 61 seconds

#### Stimulus EEG

```text
Shape: (25472, 14)
```

* 25472 time samples
* 14 EEG channels
* Approximately 199 seconds (~3.3 minutes)

---

## Emotion Labels

Each subject contains ratings for every video trial.

```text
ScoreValence    → (18,1)
ScoreArousal    → (18,1)
ScoreDominance  → (18,1)
```

Each score corresponds to one of the 18 video sequences.

---

## EEG Channel Layout

The dataset uses the 14-channel Emotiv EPOC headset.

```text
AF3, F7, F3, FC5,
T7, P7, O1, O2,
P8, T8, FC6, F4,
F8, AF4
```

### Brain Regions

* Frontal: AF3, F7, F3, F4, F8, AF4
* Fronto-Central: FC5, FC6
* Temporal: T7, T8
* Parietal: P7, P8
* Occipital: O1, O2

---

## Structure Summary

```text
DREAMER
└── Data (1,23)
    └── Subject
        ├── Age
        ├── Gender
        ├── EEG
        │   ├── baseline
        │   │   └── Trial -> (7808,14)
        │   └── stimuli
        │       └── Trial -> (25472,14)
        ├── ECG
        ├── ScoreValence
        ├── ScoreArousal
        └── ScoreDominance
```

---

## Usage in Emotion Recognition Pipeline

For each subject and video trial:

```text
Stimulus EEG
      ↓
Preprocessing
      ↓
Differential Entropy Feature Extraction
      ↓
Feature Vector
      ↓
Valence / Arousal / Dominance Labels
      ↓
XGBoost Classification
      ↓
SHAP Explainability
```

This produces a total of 414 EEG samples for model training and evaluation.

## Sources and references

### Official Source

DREAMER Dataset (Zenodo)

https://zenodo.org/records/546113

### Mirror Source

Kaggle Mirror

https://www.kaggle.com/datasets/phhasian0710/dreamer

## Local Storage

The dataset is expected to be placed in:

```text
data/raw/
├── dreamer.mat
├── DREAMER.zip
└── metadata.json
```

Since your repository may eventually be cloned by others, I would also add a short note:

### Dataset Download

Due to dataset licensing and repository size constraints, the dataset files may be not included in this repository. Download the dataset from one of the sources above and place the files in `data/raw/`.