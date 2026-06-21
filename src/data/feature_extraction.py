from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, welch
from scipy.stats import skew, kurtosis as sp_kurtosis

# --- HYPER-PARALLELIZATION CORE ---
from joblib import Parallel, delayed
import os

INTERIM_DIR   = Path("data/interim")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FS          = 128
WINDOW_SEC  = 4                   
OVERLAP_PCT = 0.5                 
WINDOW_SIZE = WINDOW_SEC * FS
STRIDE_SIZE = int(WINDOW_SIZE * (1 - OVERLAP_PCT))

CHANNELS = ["AF3","F7","F3","FC5","T7","P7","O1","O2","P8","T8","FC6","F4","F8","AF4"]
CH_IDX   = {ch: i for i, ch in enumerate(CHANNELS)}
BANDS = {"delta":(1,4),"theta":(4,8),"alpha":(8,13),"beta":(13,30),"gamma":(30,45)}

ASYMMETRY_PAIRS   = [("AF3","AF4"),("F7","F8"),("F3","F4"),("FC5","FC6"),("T7","T8"),("P7","P8")]
CONNECTIVITY_PAIRS = [("AF3","AF4"),("F3","F4"),("F7","F8"),("T7","T8"),("FC5","FC6"),("F3","T7"),("F4","T8")]

def bandpass(sig,lo,hi,fs=FS,order=4):
    nyq=0.5*fs; b,a=butter(order,[lo/nyq,hi/nyq],btype="band"); return filtfilt(b,a,sig)

def safe_div(a,b,eps=1e-10): return a/(b+eps)

def de(seg): return 0.5*np.log(2*np.pi*np.e*max(np.var(seg),1e-10))

def higuchi_fd(sig,kmax=8):
    N=len(sig); L=[]
    for k in range(1,kmax+1):
        Lk=[]
        for m in range(1,k+1):
            nm=int((N-m)/k)
            if nm<1: continue
            idx=np.array([m-1+i*k for i in range(1,nm+1)]); idx=idx[idx<N]
            if len(idx)<2: continue
            Lmk=np.sum(np.abs(np.diff(sig[idx])))*(N-1)/(k*nm); Lk.append(Lmk)
        if Lk: L.append((k,np.mean(Lk)))
    if len(L)<2: return 1.0
    ks=np.log([x[0] for x in L]); Ls=np.log([x[1] for x in L])
    return float(np.polyfit(ks,Ls,1)[0])

def hjorth(sig):
    act=np.var(sig); d1=np.diff(sig); mob=np.sqrt(safe_div(np.var(d1),act))
    comp=safe_div(np.sqrt(safe_div(np.var(np.diff(d1)),np.var(d1))),mob)
    return float(act),float(mob),float(comp)

def band_psd(sig,lo,hi,fs=FS):
    f,p=welch(sig,fs=fs,nperseg=min(len(sig),fs*2)); m=(f>=lo)&(f<=hi)
    return float(np.log(max(np.mean(p[m]) if m.any() else 1e-10,1e-10)))

def spec_ent(sig,lo,hi,fs=FS):
    f,p=welch(sig,fs=fs,nperseg=min(len(sig),fs*2)); m=(f>=lo)&(f<=hi); p=p[m]
    if p.sum()<1e-10: return 0.0
    pn=p/p.sum(); return float(-np.sum(pn*np.log(pn+1e-10)))

def precompute_baseline_averages(base_eeg):
    base_feats = {}
    bc = {}
    for ch in CHANNELS:
        ci = CH_IDX[ch]
        br = base_eeg[:, ci] - base_eeg[:, ci].mean()
        bc[ch] = {}
        bb = bandpass(br, 1, 45)
        
        base_feats[f"{ch}_hfd"] = higuchi_fd(bb)
        hb = hjorth(bb)
        for n, vb in zip(["hjorth_act", "hjorth_mob", "hjorth_comp"], hb):
            base_feats[f"{ch}_{n}"] = vb
        base_feats[f"{ch}_zcr"] = float(((bb[:-1] * bb[1:]) < 0).mean())
        
        for band, (lo, hi) in BANDS.items():
            bf = bandpass(br, lo, hi)
            bc[ch][band] = bf
            base_feats[f"{ch}_{band}_de_mean"] = de(bf)
            base_feats[f"{ch}_{band}_psd"] = band_psd(bf, lo, hi)
            base_feats[f"{ch}_{band}_spec_ent"] = spec_ent(bf, lo, hi)
            base_feats[f"{ch}_{band}_hfd"] = higuchi_fd(bf)
            
        base_feats[f"{ch}_alpha_de_base"] = de(bc[ch]["alpha"])
        base_feats[f"{ch}_beta_de_base"] = de(bc[ch]["beta"])
        base_feats[f"{ch}_theta_de_base"] = de(bc[ch]["theta"])
        base_feats[f"{ch}_gamma_de_base"] = de(bc[ch]["gamma"])
        base_feats[f"{ch}_delta_de_base"] = de(bc[ch]["delta"])

    for left, right in ASYMMETRY_PAIRS:
        for band in BANDS:
            bl = de(bc[left][band])
            br_ = de(bc[right][band])
            base_feats[f"DA_{left}_{right}_{band}_base"] = bl - br_
            base_feats[f"RASM_{left}_{right}_{band}_base"] = safe_div(bl, br_)

    for ca, cb in CONNECTIVITY_PAIRS:
        for band in BANDS:
            ba, bb2 = bc[ca][band], bc[cb][band]
            base_feats[f"CORR_{ca}_{cb}_{band}_base"] = float(np.corrcoef(ba, bb2)[0,1]) if len(ba) > 1 else 0.0
            
    return base_feats

def extract_window_features(stim_win_eeg, base_averages):
    feats = {}
    sc = {}
    for ch in CHANNELS:
        ci = CH_IDX[ch]
        sr = stim_win_eeg[:, ci] - stim_win_eeg[:, ci].mean()
        sc[ch] = {}
        sb = bandpass(sr, 1, 45)
        
        feats[f"{ch}_hfd"] = higuchi_fd(sb) - base_averages[f"{ch}_hfd"]
        hs = hjorth(sb)
        for n, vs in zip(["hjorth_act", "hjorth_mob", "hjorth_comp"], hs):
            feats[f"{ch}_{n}"] = vs - base_averages[f"{ch}_{n}"]
            
        feats[f"{ch}_mean"] = float(np.mean(sb))
        feats[f"{ch}_std"] = float(np.std(sb))
        feats[f"{ch}_skew"] = float(skew(sb))
        feats[f"{ch}_kurt"] = float(sp_kurtosis(sb))
        feats[f"{ch}_rms"] = float(np.sqrt(np.mean(sb**2)))
        feats[f"{ch}_zcr"] = float(((sb[:-1] * sb[1:]) < 0).mean()) - base_averages[f"{ch}_zcr"]
        
        be = {}
        for band, (lo, hi) in BANDS.items():
            sf = bandpass(sr, lo, hi)
            sc[ch][band] = sf
            win_de = de(sf)
            feats[f"{ch}_{band}_de_mean"] = win_de - base_averages[f"{ch}_{band}_de_mean"]
            feats[f"{ch}_{band}_psd"] = band_psd(sf, lo, hi) - base_averages[f"{ch}_{band}_psd"]
            feats[f"{ch}_{band}_spec_ent"] = spec_ent(sf, lo, hi) - base_averages[f"{ch}_{band}_spec_ent"]
            feats[f"{ch}_{band}_hfd"] = higuchi_fd(sf) - base_averages[f"{ch}_{band}_hfd"]
            be[band] = abs(win_de - base_averages[f"{ch}_{band}_de_mean"])
            
        tot = sum(be.values()) + 1e-10
        for band in BANDS:
            feats[f"{ch}_{band}_de_rel"] = be[band] / tot
            
        a = de(sc[ch]["alpha"]) - base_averages[f"{ch}_alpha_de_base"]
        b_ = de(sc[ch]["beta"]) - base_averages[f"{ch}_beta_de_base"]
        t = de(sc[ch]["theta"]) - base_averages[f"{ch}_theta_de_base"]
        g = de(sc[ch]["gamma"]) - base_averages[f"{ch}_gamma_de_base"]
        d2 = de(sc[ch]["delta"]) - base_averages[f"{ch}_delta_de_base"]
        
        feats[f"{ch}_alpha_beta_ratio"] = safe_div(a, b_)
        feats[f"{ch}_theta_beta_ratio"] = safe_div(t, b_)
        feats[f"{ch}_alpha_gamma_ratio"] = safe_div(a, g)
        feats[f"{ch}_engagement"] = safe_div(b_, abs(a) + abs(t))
        feats[f"{ch}_delta_theta_ratio"] = safe_div(d2, t)

    for left, right in ASYMMETRY_PAIRS:
        for band in BANDS:
            sl = de(sc[left][band])
            sr_ = de(sc[right][band])
            feats[f"DA_{left}_{right}_{band}"] = (sl - sr_) - base_averages[f"DA_{left}_{right}_{band}_base"]
            feats[f"RASM_{left}_{right}_{band}"] = safe_div(sl, sr_) - base_averages[f"RASM_{left}_{right}_{band}_base"]

    for ca, cb in CONNECTIVITY_PAIRS:
        for band in BANDS:
            sa, sb2 = sc[ca][band], sc[cb][band]
            cs = float(np.corrcoef(sa, sb2)[0,1]) if len(sa) > 1 else 0.0
            feats[f"CORR_{ca}_{cb}_{band}"] = cs - base_averages[f"CORR_{ca}_{cb}_{band}_base"]
            
    return feats

def make_quadrant_label(valence_score, arousal_score):
    hv = valence_score > 3.0
    ha = arousal_score > 3.0
    if hv and ha:  return 0
    if not hv and ha: return 1
    if hv and not ha: return 2
    return 3

# --- PARALLEL UNIT TASK ---
def process_single_trial(rec, label_key, quad):
    """
    Worker function executed in parallel across distinct processes.
    Processes one trial completely and returns a list of row dictionaries.
    """
    stim = np.load(INTERIM_DIR / rec["stimulus_file"])
    base = np.load(INTERIM_DIR / rec["baseline_file"])
    
    base_averages = precompute_baseline_averages(base)
    
    trial_rows = []
    total_samples = len(stim)
    start = 0
    win_idx = 0
    
    while start + WINDOW_SIZE <= total_samples:
        stim_win = stim[start:start+WINDOW_SIZE, :]
        f = extract_window_features(stim_win, base_averages)
        
        lbl = make_quadrant_label(rec["valence"], rec["arousal"]) if quad else rec[label_key]
        
        trial_rows.append({
            "subject_id": rec["subject_id"],
            "trial_id": rec["trial_id"],
            "window_id": win_idx,
            **f,
            "label": lbl
        })
        win_idx += 1
        start += STRIDE_SIZE
        
    return trial_rows

def build_dataset_parallel(metadata, label_key, quad=False):
    """
    Dispatches trial records concurrently across all available CPU cores.
    """
    # n_jobs=-1 automatically scales to use 100% of your CPU cores
    num_cores = os.cpu_count()
    print(f"  --> Launching Parallel Execution Pool across {num_cores} CPU cores...")
    
    results = Parallel(n_jobs=-1, verbose=10)(
        delayed(process_single_trial)(rec, label_key, quad) for rec in metadata
    )
    
    # Flatten the list of lists into a singular continuous array
    flattened_rows = [row for trial in results for row in trial]
    return pd.DataFrame(flattened_rows)

def normalize_per_subject(df):
    fc = [c for c in df.columns if c not in {"subject_id", "trial_id", "window_id", "label"}]
    df[fc] = df.groupby("subject_id")[fc].transform(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    return df

def load_metadata():
    with open(INTERIM_DIR / "metadata.json", "r", encoding="utf-8") as f: return json.load(f)

def main():
    print("Loading metadata …"); meta = load_metadata()
    datasets = {
        "features_valence.csv": ("valence_class", False),
        "features_arousal.csv": ("arousal_class", False),
        "features_quadrant.csv": ("", True),
    }
    for fname, (lkey, quad) in datasets.items():
        print(f"\nCreating {fname}")
        df = build_dataset_parallel(meta, lkey, quad)
        df = normalize_per_subject(df)
        out = PROCESSED_DIR / fname; df.to_csv(out, index=False)
        print(f"  Saved: {out}  shape={df.shape}")
        print(f"  Class dist: {dict(zip(*np.unique(df['label'].values, return_counts=True)))}")
    print("\nDone.")

if __name__ == "__main__": 
    main()