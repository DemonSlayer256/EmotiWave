"""
Extract rich multi-domain features from DREAMER EEG trials.

Feature set (per channel x band unless noted):
  - Differential Entropy (DE)          - your original
  - Power Spectral Density (PSD)        - log band power via Welch
  - Hjorth Activity / Mobility / Complexity
  - Statistical moments: mean, std, skewness, kurtosis, RMS
  - Spectral Entropy (per band)
  - Petrosian Fractal Dimension
  - Zero-Crossing Rate
Asymmetry features (per pair x band):
  - DA  (Differential Asymmetry)
  - RASM (Rational Asymmetry)
Cross-channel connectivity (selected frontal pairs):
  - Pearson Correlation (per band)
  - Coherence (per band)

Output:
    data/processed/
        features_valence.csv
        features_arousal.csv
        features_dominance.csv
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch, coherence as scipy_coherence
from scipy.stats import skew, kurtosis as scipy_kurtosis

warnings.filterwarnings("ignore")

# ─── paths ────────────────────────────────────────────────────────────────────
INTERIM_DIR   = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─── EEG constants ────────────────────────────────────────────────────────────
FS = 128

CHANNELS = [
    "AF3", "F7", "F3", "FC5",
    "T7",  "P7", "O1", "O2",
    "P8",  "T8", "FC6","F4",
    "F8",  "AF4"
]

CH_IDX = {ch: i for i, ch in enumerate(CHANNELS)}

BANDS = {
    "delta": (1,  4),
    "theta": (4,  8),
    "alpha": (8,  13),
    "beta":  (13, 30),
    "gamma": (30, 45),
}

WINDOW_SIZE = 2 * FS   # 2-second non-overlapping windows

ASYMMETRY_PAIRS = [
    ("AF3", "AF4"),
    ("F7",  "F8"),
    ("F3",  "F4"),
    ("FC5", "FC6"),
    ("T7",  "T8"),
    ("P7",  "P8"),
]

# frontal / temporal pairs for connectivity
CONNECTIVITY_PAIRS = [
    ("AF3", "AF4"),
    ("F3",  "F4"),
    ("F7",  "F8"),
    ("T7",  "T8"),
    ("FC5", "FC6"),
    ("F3",  "T7"),
    ("F4",  "T8"),
]


# ─── signal utilities ─────────────────────────────────────────────────────────
def bandpass_filter(signal, low, high, fs=FS, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def safe_divide(a, b, eps=1e-10):
    return a / (b + eps)


# ─── per-window DE ────────────────────────────────────────────────────────────
def differential_entropy(segment):
    var = max(np.var(segment), 1e-10)
    return 0.5 * np.log(2 * np.pi * np.e * var)


def windowed_de(signal):
    vals = []
    for s in range(0, len(signal) - WINDOW_SIZE + 1, WINDOW_SIZE):
        vals.append(differential_entropy(signal[s: s + WINDOW_SIZE]))
    return np.array(vals) if vals else np.array([differential_entropy(signal)])


# ─── Hjorth parameters ────────────────────────────────────────────────────────
def hjorth(signal):
    """Returns (activity, mobility, complexity)."""
    activity   = np.var(signal)
    d1         = np.diff(signal)
    mobility   = np.sqrt(safe_divide(np.var(d1), activity))
    d2         = np.diff(d1)
    mob_d1     = np.sqrt(safe_divide(np.var(d2), np.var(d1)))
    complexity = safe_divide(mob_d1, mobility)
    return float(activity), float(mobility), float(complexity)


# ─── PSD via Welch ────────────────────────────────────────────────────────────
def band_psd(signal, low, high, fs=FS):
    """Log-power in a frequency band using Welch."""
    nperseg = min(len(signal), FS * 2)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= low) & (freqs <= high)
    bp = np.mean(psd[mask]) if mask.any() else 1e-10
    return float(np.log(max(bp, 1e-10)))


# ─── spectral entropy ─────────────────────────────────────────────────────────
def spectral_entropy(signal, low, high, fs=FS):
    nperseg = min(len(signal), FS * 2)
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)
    mask = (freqs >= low) & (freqs <= high)
    p = psd[mask]
    if p.sum() < 1e-10:
        return 0.0
    p_norm = p / p.sum()
    return float(-np.sum(p_norm * np.log(p_norm + 1e-10)))


# ─── Petrosian fractal dimension ─────────────────────────────────────────────
def petrosian_fd(signal):
    n    = len(signal)
    diff = np.diff(signal)
    nzc  = np.sum(diff[:-1] * diff[1:] < 0)
    if nzc == 0:
        return 0.0
    return float(np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * nzc))))


# ─── zero crossing rate ───────────────────────────────────────────────────────
def zero_crossing_rate(signal):
    return float(((signal[:-1] * signal[1:]) < 0).mean())


# ─── band coherence (mean over freqs in band) ────────────────────────────────
def band_coherence(sig_a, sig_b, low, high, fs=FS):
    nperseg = min(len(sig_a), FS * 2)
    freqs, coh = scipy_coherence(sig_a, sig_b, fs=fs, nperseg=nperseg)
    mask = (freqs >= low) & (freqs <= high)
    return float(np.mean(coh[mask])) if mask.any() else 0.0


# ─── main feature extractor ───────────────────────────────────────────────────
def extract_trial_features(stimulus_eeg, baseline_eeg):
    """
    Returns a flat dict of scalar features for one trial.
    stimulus_eeg / baseline_eeg: (T, 14) float arrays.
    """
    features = {}

    # per-channel signals (mean-centred, stimulus – baseline referenced)
    stim_sigs = {}   # {ch: {band: filtered_signal}}
    base_sigs = {}

    # ── per-channel, per-band ──────────────────────────────────────────────
    for ch in CHANNELS:
        ci = CH_IDX[ch]
        stim_raw = stimulus_eeg[:, ci]  - stimulus_eeg[:, ci].mean()
        base_raw = baseline_eeg[:, ci]  - baseline_eeg[:, ci].mean()

        stim_sigs[ch] = {}
        base_sigs[ch] = {}

        # Hjorth on raw band-limited signal (broadband 1–45 Hz)
        stim_broad = bandpass_filter(stim_raw, 1, 45)
        base_broad = bandpass_filter(base_raw, 1, 45)
        hj_s = hjorth(stim_broad)
        hj_b = hjorth(base_broad)
        for hname, hs, hb in zip(
            ["activity", "mobility", "complexity"], hj_s, hj_b
        ):
            features[f"{ch}_hjorth_{hname}"] = hs - hb

        # PFD and ZCR on broadband
        features[f"{ch}_pfd"]  = petrosian_fd(stim_broad)  - petrosian_fd(base_broad)
        features[f"{ch}_zcr"]  = zero_crossing_rate(stim_broad) - zero_crossing_rate(base_broad)

        # statistical moments on broadband
        for tag, arr_s, arr_b in [
            ("stim", stim_broad, None),
        ]:
            diff_sig = stim_broad - np.mean(stim_broad[:len(base_broad)])
            features[f"{ch}_stat_mean"]     = float(np.mean(diff_sig))
            features[f"{ch}_stat_std"]      = float(np.std(diff_sig))
            features[f"{ch}_stat_skew"]     = float(skew(diff_sig))
            features[f"{ch}_stat_kurt"]     = float(scipy_kurtosis(diff_sig))
            features[f"{ch}_stat_rms"]      = float(np.sqrt(np.mean(diff_sig ** 2)))

        # per-band features
        band_de_stim = {}
        band_de_base = {}
        band_energy  = {}

        for band, (lo, hi) in BANDS.items():
            sf = bandpass_filter(stim_raw, lo, hi)
            bf = bandpass_filter(base_raw, lo, hi)
            stim_sigs[ch][band] = sf
            base_sigs[ch][band] = bf

            de_s = windowed_de(sf)
            de_b = windowed_de(bf)
            band_de_stim[band] = np.mean(de_s)
            band_de_base[band] = np.mean(de_b)
            diff_de = np.mean(de_s) - np.mean(de_b)
            band_energy[band]  = abs(diff_de)

            # DE statistics
            features[f"{ch}_{band}_de_mean"]   = diff_de
            features[f"{ch}_{band}_de_std"]    = np.std(de_s) - np.std(de_b)
            features[f"{ch}_{band}_de_max"]    = np.max(de_s) - np.max(de_b)
            features[f"{ch}_{band}_de_min"]    = np.min(de_s) - np.min(de_b)
            features[f"{ch}_{band}_de_median"] = np.median(de_s) - np.median(de_b)

            # PSD
            features[f"{ch}_{band}_psd"] = band_psd(sf, lo, hi) - band_psd(bf, lo, hi)

            # Spectral entropy
            features[f"{ch}_{band}_spec_ent"] = (
                spectral_entropy(sf, lo, hi) - spectral_entropy(bf, lo, hi)
            )

        # relative band energy
        total_e = sum(band_energy.values()) + 1e-10
        for band in BANDS:
            features[f"{ch}_{band}_de_relative"] = band_energy[band] / total_e

        # band ratio features (neurophysiologically meaningful)
        α = band_de_stim["alpha"] - band_de_base["alpha"]
        β = band_de_stim["beta"]  - band_de_base["beta"]
        θ = band_de_stim["theta"] - band_de_base["theta"]
        γ = band_de_stim["gamma"] - band_de_base["gamma"]
        δ = band_de_stim["delta"] - band_de_base["delta"]

        features[f"{ch}_alpha_beta_ratio"]  = safe_divide(α, β)
        features[f"{ch}_theta_beta_ratio"]  = safe_divide(θ, β)
        features[f"{ch}_alpha_gamma_ratio"] = safe_divide(α, γ)
        features[f"{ch}_theta_alpha_ratio"] = safe_divide(θ, α)
        features[f"{ch}_beta_alpha_ratio"]  = safe_divide(β, α)
        features[f"{ch}_delta_alpha_ratio"] = safe_divide(δ, α)
        # engagement index: β / (α + θ)
        features[f"{ch}_engagement"]        = safe_divide(β, abs(α) + abs(θ))

    # ── asymmetry features ────────────────────────────────────────────────
    for left, right in ASYMMETRY_PAIRS:
        for band in BANDS:
            sl = np.mean(windowed_de(stim_sigs[left][band]))
            sr = np.mean(windowed_de(stim_sigs[right][band]))
            bl = np.mean(windowed_de(base_sigs[left][band]))
            br = np.mean(windowed_de(base_sigs[right][band]))

            features[f"DA_{left}_{right}_{band}"]   = (sl - sr) - (bl - br)
            features[f"RASM_{left}_{right}_{band}"] = (
                safe_divide(sl, sr) - safe_divide(bl, br)
            )

    # ── inter-channel connectivity ─────────────────────────────────────────
    for ch_a, ch_b in CONNECTIVITY_PAIRS:
        for band in BANDS:
            sa = stim_sigs[ch_a][band]
            sb = stim_sigs[ch_b][band]
            ba = base_sigs[ch_a][band]
            bb = base_sigs[ch_b][band]

            lo, hi = BANDS[band]

            # Pearson correlation (stimulus – baseline)
            corr_s = float(np.corrcoef(sa, sb)[0, 1]) if len(sa) > 1 else 0.0
            corr_b = float(np.corrcoef(ba, bb)[0, 1]) if len(ba) > 1 else 0.0
            features[f"CORR_{ch_a}_{ch_b}_{band}"] = corr_s - corr_b

            # Coherence
            coh_s = band_coherence(sa, sb, lo, hi)
            coh_b = band_coherence(ba, bb, lo, hi)
            features[f"COH_{ch_a}_{ch_b}_{band}"] = coh_s - coh_b

    return features


# ─── normalisation ────────────────────────────────────────────────────────────
def normalize_per_subject(df):
    feature_cols = [
        c for c in df.columns if c not in {"subject_id", "trial_id", "label"}
    ]
    df[feature_cols] = (
        df.groupby("subject_id")[feature_cols]
        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    )
    return df


# ─── I/O helpers ─────────────────────────────────────────────────────────────
def load_metadata():
    with open(INTERIM_DIR / "metadata.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset(metadata, label_key):
    rows  = []
    total = len(metadata)

    for idx, rec in enumerate(metadata, 1):
        print(f"  [{idx:3d}/{total}] {rec['stimulus_file']}")

        stim = np.load(INTERIM_DIR / rec["stimulus_file"])
        base = np.load(INTERIM_DIR / rec["baseline_file"])

        feats = extract_trial_features(stim, base)
        rows.append({
            "subject_id": rec["subject_id"],
            "trial_id":   rec["trial_id"],
            **feats,
            "label": rec[label_key],
        })

    return pd.DataFrame(rows)


def save_dataset(df, filename):
    out = PROCESSED_DIR / filename
    df.to_csv(out, index=False)
    print(f"  Saved: {out}  shape={df.shape}")


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    print("Loading metadata …")
    metadata = load_metadata()

    datasets = {
        "features_valence.csv":   "valence_class",
        # "features_arousal.csv":   "arousal_class",
        # "features_dominance.csv": "dominance_class",
    }

    for filename, label_key in datasets.items():
        print(f"\nCreating {filename}  ({label_key})")
        df = build_dataset(metadata, label_key)

        print("  Applying per-subject z-score normalisation …")
        df = normalize_per_subject(df)

        save_dataset(df, filename)

    print("\nFeature extraction complete.")


if __name__ == "__main__":
    main()