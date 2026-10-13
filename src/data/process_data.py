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

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat


RAW_FILE = Path("data/raw/DREAMER.mat")
OUTPUT_DIR = Path("data/interim")
QUADRANT_PLOT_PATH = Path("logs/plots/valence_intensity_quadrants.png")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def create_binary_label(score: float) -> int:
    """
    DREAMER ratings are on a 1-5 scale.

    High emotion = > 3
    Low emotion  = <= 3
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


def save_trial_eeg(
    stimulus_eeg,
    baseline_eeg,
    subject_id,
    trial_id
):
    stimulus_file = (
        f"S{subject_id:02d}_T{trial_id:02d}_stimulus.npy"
    )

    baseline_file = (
        f"S{subject_id:02d}_T{trial_id:02d}_baseline.npy"
    )

    np.save(
        OUTPUT_DIR / stimulus_file,
        stimulus_eeg.astype(np.float32)
    )

    np.save(
        OUTPUT_DIR / baseline_file,
        baseline_eeg.astype(np.float32)
    )

    return stimulus_file, baseline_file

def process_subject(subject, subject_id):
    records = []

    age, gender = extract_subject_metadata(subject)

    eeg_struct = subject["EEG"][0, 0]

    stimuli_trials = eeg_struct["stimuli"][0, 0]
    baseline_trials = eeg_struct['baseline'][0, 0]

    valence_scores = subject["ScoreValence"][0, 0].flatten()
    arousal_scores = subject["ScoreArousal"][0, 0].flatten()
    dominance_scores = subject["ScoreDominance"][0, 0].flatten()

    num_trials = len(valence_scores)

    for trial_idx in range(num_trials):

        stimulus_eeg = stimuli_trials[trial_idx, 0]
        baseline_eeg = baseline_trials[trial_idx, 0]

        stimulus_file, baseline_file = (
            save_trial_eeg(
                stimulus_eeg,
                baseline_eeg,
                subject_id,
                trial_idx + 1
            )
        )
        valence = float(valence_scores[trial_idx])
        arousal = float(arousal_scores[trial_idx])
        dominance = float(dominance_scores[trial_idx])

        records.append(
            {
                "subject_id": subject_id,
                "trial_id": trial_idx + 1,
                "stimulus_file": stimulus_file,
                "baseline_file": baseline_file,

                "stimulus_shape": list(
                    stimulus_eeg.shape
                ),

                "baseline_shape": list(
                    baseline_eeg.shape
                ),

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


def plot_valence_intensity_quadrants(metadata):
    counts = [[0, 0], [0, 0]]

    for record in metadata:
        valence_class = record["valence_class"]
        intensity_class = record["arousal_class"]
        counts[intensity_class][valence_class] += 1

    total_trials = sum(sum(row) for row in counts)
    if total_trials == 0:
        raise ValueError("Cannot plot valence and intensity without trials.")

    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(counts, cmap="Blues", origin="lower")
    max_count = max(max(row) for row in counts)

    for intensity_class in range(2):
        for valence_class in range(2):
            count = counts[intensity_class][valence_class]
            valence_label = "Positive" if valence_class else "Negative"
            intensity_label = "High" if intensity_class else "Low"
            percentage = count / total_trials * 100
            color = (
                "white"
                if max_count and count >= max_count * 0.55
                else "#1f2937"
            )
            axis.text(
                valence_class,
                intensity_class,
                f"{valence_label} valence\n"
                f"{intensity_label} intensity\n"
                f"{count} trials ({percentage:.1f}%)",
                ha="center",
                va="center",
                color=color,
            )

    axis.set_xticks([0, 1], labels=["Negative", "Positive"])
    axis.set_yticks([0, 1], labels=["Low", "High"])
    axis.set_xlabel("Valence")
    axis.set_ylabel("Intensity (arousal)")
    axis.set_title("DREAMER Trial Distribution by Valence and Intensity")
    figure.colorbar(image, ax=axis, label="Number of trials")
    figure.text(
        0.5,
        0.01,
        "Ratings <= 3 are negative/low; ratings > 3 are positive/high.",
        ha="center",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))

    QUADRANT_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(QUADRANT_PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)
    print(f"Valence/intensity graph saved to: {QUADRANT_PLOT_PATH}")


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
    plot_valence_intensity_quadrants(all_metadata)

    print("\nProcessing Complete")
    print(f"Total Trials: {len(all_metadata)}")
    print(f"Output Dir : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()