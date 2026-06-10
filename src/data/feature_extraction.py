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

WINDOW_SIZE = 2 * FS  # 2 seconds


ASYMMETRY_PAIRS = [
    ("AF3", "AF4"),
    ("F7", "F8"),
    ("F3", "F4"),
    ("FC5", "FC6"),
    ("T7", "T8"),
    ("P7", "P8")
]


def compute_window_de(signal):
    de_values = []

    for start in range(
        0,
        len(signal) - WINDOW_SIZE + 1,
        WINDOW_SIZE
    ):
        window = signal[
            start:start + WINDOW_SIZE
        ]

        de_values.append(
            differential_entropy(window)
        )

    return np.array(de_values)

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

def safe_divide(a, b, eps=1e-10):
    return a / (b + eps)

def normalize_per_subject(df):

    feature_cols = [
        col
        for col in df.columns
        if col not in [
            "subject_id",
            "trial_id",
            "label"
        ]
    ]

    df[feature_cols] = (
        df.groupby("subject_id")[feature_cols]
        .transform(
            lambda x:
            (x - x.mean())
            /
            (x.std() + 1e-8)
        )
    )

    return df

def extract_trial_features(
    stimulus_eeg,
    baseline_eeg
):
    features = {}

    stim_cache = {}
    base_cache = {}
    de_summary = {}

    for ch_idx, channel_name in enumerate(CHANNELS):

        de_summary[channel_name] = {}
        stim_signal = (
            stimulus_eeg[:, ch_idx]
            - np.mean(
                stimulus_eeg[:, ch_idx]
            )
        )

        base_signal = (
            baseline_eeg[:, ch_idx]
            - np.mean(
                baseline_eeg[:, ch_idx]
            )
        )

        stim_cache[channel_name] = {}
        base_cache[channel_name] = {}

        for band_name, (low, high) in BANDS.items():

            stim_filtered = bandpass_filter(
                stim_signal,
                low,
                high
            )

            base_filtered = bandpass_filter(
                base_signal,
                low,
                high
            )

            stim_cache[channel_name][
                band_name
            ] = stim_filtered

            base_cache[channel_name][
                band_name
            ] = base_filtered

            stim_de = compute_window_de(
                stim_filtered
            )

            base_de = compute_window_de(
                base_filtered
            )

            de_summary[channel_name][band_name] = (
            np.mean(stim_de)
            - np.mean(base_de)
            )
            
            band_feature = {
                "mean":
                    np.mean(stim_de) - np.mean(base_de),

                "std":
                    np.std(stim_de) - np.std(base_de),

                "max":
                    np.max(stim_de) - np.max(base_de),

                "min":
                    np.min(stim_de) - np.min(base_de),

                "median":
                    np.median(stim_de) - np.median(base_de)
            }

            for stat_name, value in band_feature.items():

                features[
                    f"{channel_name}_{band_name}_{stat_name}"
                ] = value
        total_energy = sum(
            abs(
                de_summary[channel_name][band]
            )
            for band in BANDS
        )

        for band_name in BANDS:

            relative_energy = safe_divide(
                de_summary[channel_name][band_name],
                total_energy
            )

            features[
                f"{channel_name}_{band_name}_relative"
            ] = relative_energy

        alpha = de_summary[channel_name]["alpha"]
        beta = de_summary[channel_name]["beta"]
        theta = de_summary[channel_name]["theta"]
        gamma = de_summary[channel_name]["gamma"]

        features[
            f"{channel_name}_alpha_beta_ratio"
        ] = safe_divide(alpha, beta)

        features[
            f"{channel_name}_theta_beta_ratio"
        ] = safe_divide(theta, beta)

        features[
            f"{channel_name}_alpha_gamma_ratio"
        ] = safe_divide(alpha, gamma)

    for left, right in ASYMMETRY_PAIRS:
        for band_name in BANDS:

            stim_left = np.mean(
                compute_window_de(
                    stim_cache[left][band_name]
                )
            )

            stim_right = np.mean(
                compute_window_de(
                    stim_cache[right][band_name]
                )
            )

            base_left = np.mean(
                compute_window_de(
                    base_cache[left][band_name]
                )
            )

            base_right = np.mean(
                compute_window_de(
                    base_cache[right][band_name]
                )
            )

            da_feature = (
                (stim_left - stim_right)
                -
                (base_left - base_right)
            )

            features[
                f"DA_{left}_{right}_{band_name}"
            ] = da_feature

            rasm_feature = (
            safe_divide(
                stim_left,
                stim_right
            )
            -
            safe_divide(
                base_left,
                base_right
            )
        )

            features[
                f"RASM_{left}_{right}_{band_name}"
            ] = rasm_feature

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
            f"{record['stimulus_file']}"
        )

        stimulus = np.load(
            INTERIM_DIR / record["stimulus_file"]
        )

        baseline = np.load(
            INTERIM_DIR / record['baseline_file']
        )

        features = extract_trial_features(
            stimulus, 
            baseline
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
            "valence_class"

        # "features_arousal.csv":
        #     "arousal_class",

        # "features_dominance.csv":
        #     "dominance_class"
    }

    for filename, label_key in datasets.items():

        print(
            f"\nCreating {filename}"
        )
        df = build_dataset(
            metadata,
            label_key
        )

        print(
            "Applying subject-wise normalization..."
        )

        df = normalize_per_subject(df)

        save_dataset(
            df,
            filename
        )

    print(
        "\nFeature extraction complete."
    )


if __name__ == "__main__":
    main()