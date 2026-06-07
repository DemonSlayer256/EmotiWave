"""
Convert DREAMER.mat into a structured dataset.

Output:
data/interim/
├── metadata.json
├── S01_T01.npy
├── S01_T02.npy
└── ...

Author: Derek
"""

from pathlib import Path
import json

import numpy as np
from scipy.io import loadmat


RAW_FILE = Path("data/raw/DREAMER.mat")
OUTPUT_DIR = Path("data/interim")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_binary_label(score: float) -> int:
    """
    DREAMER ratings are on a 1-5 scale.

    High emotion = >= 3
    Low emotion  = < 3
    """
    return int(score > 3.0)


def load_dataset():
    mat = loadmat(RAW_FILE)

    dreamer = mat["DREAMER"]

    return dreamer


def extract_subject_metadata(subject):
    age = int(subject["Age"][0, 0][0])

    gender_raw = subject["Gender"][0, 0]

    try:
        gender = str(gender_raw[0])
    except Exception:
        gender = str(gender_raw)

    return age, gender


def save_trial_eeg(eeg_data, subject_id, trial_id):
    filename = f"S{subject_id:02d}_T{trial_id:02d}.npy"

    filepath = OUTPUT_DIR / filename

    np.save(filepath, eeg_data.astype(np.float32))

    return filename


def process_subject(subject, subject_id):
    records = []

    age, gender = extract_subject_metadata(subject)

    eeg_struct = subject["EEG"][0, 0]

    stimuli_trials = eeg_struct["stimuli"][0, 0]

    valence_scores = subject["ScoreValence"][0, 0].flatten()
    arousal_scores = subject["ScoreArousal"][0, 0].flatten()
    dominance_scores = subject["ScoreDominance"][0, 0].flatten()

    num_trials = len(valence_scores)

    for trial_idx in range(num_trials):

        eeg_trial = stimuli_trials[trial_idx, 0]

        filename = save_trial_eeg(
            eeg_trial,
            subject_id,
            trial_idx + 1
        )

        valence = float(valence_scores[trial_idx])
        arousal = float(arousal_scores[trial_idx])
        dominance = float(dominance_scores[trial_idx])

        records.append(
            {
                "subject_id": subject_id,
                "trial_id": trial_idx + 1,

                "eeg_file": filename,

                "shape": list(eeg_trial.shape),

                "age": age,
                "gender": gender,

                "valence": valence,
                "arousal": arousal,
                "dominance": dominance,

                "valence_class": create_binary_label(valence),
                "arousal_class": create_binary_label(arousal),
                "dominance_class": create_binary_label(dominance)
            }
        )

    return records


def save_metadata(metadata):
    output_file = OUTPUT_DIR / "metadata.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            metadata,
            f,
            indent=4
        )


def main():
    print("=" * 60)
    print("Loading DREAMER Dataset")
    print("=" * 60)

    dreamer = load_dataset()

    data = dreamer["Data"][0, 0]

    total_subjects = data.shape[1]

    all_metadata = []

    for subject_idx in range(total_subjects):

        print(
            f"Processing Subject "
            f"{subject_idx + 1}/{total_subjects}"
        )

        subject = data[0, subject_idx]

        records = process_subject(
            subject,
            subject_idx + 1
        )

        all_metadata.extend(records)

    save_metadata(all_metadata)

    print("\nProcessing Complete")
    print(f"Total Trials: {len(all_metadata)}")
    print(f"Output Dir : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()