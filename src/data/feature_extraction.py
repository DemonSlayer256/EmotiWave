"""
Extract Differential Entropy (DE) features from DREAMER EEG trials.

Input:
    data/interim/
        metadata.json
        SXX_TXX.npy

Output:
    data/processed/
        features_valence.csv
        features_arousal.csv
        features_dominance.csv
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


INTERIM_DIR = Path("data/interim")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


FS = 128

CHANNELS = [
    "AF3", "F7", "F3", "FC5",
    "T7", "P7", "O1", "O2",
    "P8", "T8", "FC6", "F4",
    "F8", "AF4"
]

BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45)
}


def bandpass_filter(signal, low, high, fs=FS, order=4):
    nyquist = 0.5 * fs

    low /= nyquist
    high /= nyquist

    b, a = butter(
        order,
        [low, high],
        btype="band"
    )

    return filtfilt(b, a, signal)


def differential_entropy(signal):
    variance = np.var(signal)

    variance = max(
        variance,
        1e-10
    )

    return 0.5 * np.log(
        2 * np.pi * np.e * variance
    )


def extract_trial_features(eeg):
    features = {}

    for ch_idx, channel_name in enumerate(CHANNELS):

        channel_signal = eeg[:, ch_idx]

        for band_name, (low, high) in BANDS.items():

            filtered = bandpass_filter(
                channel_signal,
                low,
                high
            )

            de = differential_entropy(
                filtered
            )

            feature_name = (
                f"{channel_name}_{band_name}"
            )

            features[feature_name] = de

    return features


def load_metadata():
    with open(
        INTERIM_DIR / "metadata.json",
        "r",
        encoding="utf-8"
    ) as f:
        return json.load(f)


def build_dataset(metadata, label_key):
    rows = []

    total = len(metadata)

    for idx, record in enumerate(metadata, start=1):

        print(
            f"[{idx}/{total}] "
            f"{record['eeg_file']}"
        )

        eeg = np.load(
            INTERIM_DIR / record["eeg_file"]
        )

        features = extract_trial_features(
            eeg
        )

        row = {
            "subject_id": record["subject_id"],
            "trial_id": record["trial_id"],
            **features,
            "label": record[label_key]
        }

        rows.append(row)

    return pd.DataFrame(rows)


def save_dataset(df, filename):
    output_file = PROCESSED_DIR / filename

    df.to_csv(
        output_file,
        index=False
    )

    print(
        f"Saved: {output_file}"
    )


def main():
    print(
        "Loading metadata..."
    )

    metadata = load_metadata()

    datasets = {
        "features_valence.csv":
            "valence_class",

        "features_arousal.csv":
            "arousal_class",

        "features_dominance.csv":
            "dominance_class"
    }

    for filename, label_key in datasets.items():

        print(
            f"\nCreating {filename}"
        )

        df = build_dataset(
            metadata,
            label_key
        )

        save_dataset(
            df,
            filename
        )

    print(
        "\nFeature extraction complete."
    )


if __name__ == "__main__":
    main()