import json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt


DATA_DIR = Path("data/interim")
PLOT_PATH = Path("logs/plots/eeg_trial_subject_1.png")
files = sorted(DATA_DIR.glob("*.npy"))

with open(DATA_DIR / "metadata.json", "r") as f:
    metadata = json.load(f)

def inspect_processed():
    print("\nPROCESSED DATA")
    print(
        f"Total Records : {len(metadata)}\n"
        f"First Record  : {metadata[0]}\n"
        f"Last Record   : {metadata[-1]}"
    )

def inspect_eeg():
    print("\nEEG FILES")
    sample = np.load(files[0])
    print(
        f"EEG Files : {len(files)}\n"
        f"Shape     : {sample.shape}\n"
        f"Type      : {sample.dtype}\n"
        f"Min       : {sample.min():.4f}\n"
        f"Max       : {sample.max():.4f}\n"
        f"Mean      : {sample.mean():.4f}"
    )

def inspect_channels():
    print("\nCHANNEL CHECK")
    channels = {np.load(f).shape[1] for f in files}
    print(f"Channel Counts Found: {channels}")

def inspect_balance():
    print("\nCLASS BALANCE")
    valence = Counter(row["valence_class"] for row in metadata)
    arousal = Counter(row["arousal_class"] for row in metadata)
    dominance = Counter(row["dominance_class"] for row in metadata)
    print(
        f"Valence   : {dict(valence)}\n"
        f"Arousal   : {dict(arousal)}\n"
        f"Dominance : {dict(dominance)}"
    )

def inspect_trial():
    print("\nEEG VISUALIZATION")
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    eeg = np.load(DATA_DIR / "S01_T01.npy")
    plt.figure(figsize=(12, 5))
    plt.plot(eeg[:, 0])
    plt.title("Subject 1 Trial 1 - Channel 1")
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {PLOT_PATH}")

def main():
    print("Inspecting processed DREAMER dataset.")
    inspect_processed()
    inspect_eeg()
    inspect_channels()
    inspect_balance()
    inspect_trial()
    print("Inspection completed successfully.")

if __name__ == "__main__":
    main()