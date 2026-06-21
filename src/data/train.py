"""
train.py — High-Performance XGBoost Pipeline for Windowed DREAMER Data

Fixes implemented for Windowed Layout:
  1. Replaced slow RBF SVM with highly efficient, regularized XGBoost models.
  2. Integrated StratifiedGroupKFold for Subject-Dependent CV to stop intra-trial window leakage.
  3. Integrated LeaveOneGroupOut for Subject-Independent CV (LOSO) paired with window-level CORAL.
  4. Explicitly dropped 'window_id' from feature spaces to prevent index learning.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score, roc_auc_score, matthews_corrcoef)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline

BINARY_DATASET   = Path("data/processed/features_valence.csv")
QUAD_DATASET     = Path("data/processed/features_quadrant.csv")
TOP_K            = 60   # Keep top 60 features via MI selection


def load_windowed(path):
    """Loads windowed feature CSVs and builds strict separation groupings."""
    df = pd.read_csv(path)
    
    sub_groups = df["subject_id"].values
    # Prevent cross-window leakage from the same trial clip
    trial_groups = (df["subject_id"].astype(str) + "_" + df["trial_id"].astype(str)).values
    
    y = df["label"].values
    # Strip administrative descriptors out of the model training matrix
    X = df.drop(columns=["subject_id", "trial_id", "window_id", "label"], errors="ignore")
    
    print(f"  {path.name}: shape={df.shape} classes={dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, sub_groups, trial_groups


def coral(Xs, Xt):
    """Covariance Domain Alignment across source and target feature distributions."""
    reg = 1e-4 * np.eye(Xs.shape[1])
    cs = np.cov(Xs, rowvar=False) + reg
    ct = np.cov(Xt, rowvar=False) + reg
    try:
        Ut, St, _ = np.linalg.svd(ct)
        Wt = Ut @ np.diag(1 / np.sqrt(np.clip(St, 1e-10, None))) @ Ut.T
        Us, Ss, _ = np.linalg.svd(cs)
        Ws = Us @ np.diag(np.sqrt(np.clip(Ss, 1e-10, None))) @ Us.T
        return Xt @ Wt @ Ws
    except Exception:
        return Xt


def select(Xtr, ytr, Xte):
    """Mutual Information feature sub-selection layer."""
    k = min(TOP_K, Xtr.shape[1])
    sel = SelectKBest(mutual_info_classif, k=k)
    return sel.fit_transform(Xtr, ytr), sel.transform(Xte)

def build_xgb(n_class=2):
    """Returns an optimized XGBoost pipeline wrapper with your best search parameters."""
    objective = "binary:logistic" if n_class == 2 else "multi:softprob"
    
    return XGBClassifier(
        n_estimators=300,        # Scaled up for stable learning steps
        max_depth=4,             # Kept shallow to prevent overfitting
        learning_rate=0.01,      # Dropped lower for precise gradient descents
        subsample=0.9,           # Your optimized row sampling rate
        colsample_bytree=0.8,    # Your optimized feature sampling rate
        objective=objective,
        eval_metric="logloss" if n_class == 2 else "mlogloss",
        random_state=42,
        n_jobs=-1                # Still utilizes all 4 cores on Codespaces
    )

def metrics_binary(yt, yp, yprob):
    return {
        "acc": accuracy_score(yt, yp),
        "bal": balanced_accuracy_score(yt, yp),
        "f1": f1_score(yt, yp, zero_division=0),
        "auc": roc_auc_score(yt, yprob) if len(np.unique(yt)) > 1 else 0.5,
        "mcc": matthews_corrcoef(yt, yp)
    }


def metrics_multi(yt, yp):
    return {
        "acc": accuracy_score(yt, yp),
        "bal": balanced_accuracy_score(yt, yp),
        "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(yt, yp)
    }


def summarize(rows, title, cols):
    df = pd.DataFrame(rows)
    print(f"\n{'='*50}\n {title}\n{'='*50}")
    for c in cols:
        if c in df:
            print(f"  {c:<16} {df[c].mean():.4f} ± {df[c].std():.4f}")


# ═══════════════════════════════════════════
# LEAK-PROOF SUBJECT-DEPENDENT INTERFACE
# ═══════════════════════════════════════════
def eval_subject_dependent(X, y, sub_groups, trial_groups, n_class=2):
    tag = "binary" if n_class == 2 else "4-class"
    print(f"\n{'='*55}\nSUBJECT-DEPENDENT ({tag}, Window-Level 5-Fold SGKF)\n{'='*55}")
    
    Xn = X.values
    subs = np.unique(sub_groups)
    all_m = []
    
    # Ensure label ranges match exactly [0, n_classes - 1] for XGBoost requirements
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    for s in subs:
        mask = sub_groups == s
        Xs, ys, ts = Xn[mask], y_encoded[mask], trial_groups[mask]
        
        # Enforce keeping windows from identical clips entirely grouped together
        sgkf = StratifiedGroupKFold(n_splits=5)
        fm = []
        
        for tr, te in sgkf.split(Xs, ys, groups=ts):
            Xtr, Xte = Xs[tr], Xs[te]
            ytr, yte = ys[tr], ys[te]
            
            if len(np.unique(ytr)) < 2: 
                continue
                
            Xtr_s, Xte_s = select(Xtr, ytr, Xte)
            
            sc = StandardScaler()
            Xtr_sc = sc.fit_transform(Xtr_s)
            Xte_sc = sc.transform(Xte_s)
            
            model = build_xgb(n_class=n_class)
            model.fit(Xtr_sc, ytr)
            yp = model.predict(Xte_sc)
            
            if n_class == 2:
                yprob = model.predict_proba(Xte_sc)[:, 1]
                fm.append(metrics_binary(yte, yp, yprob))
            else:
                fm.append(metrics_multi(yte, yp))
                
        if not fm: 
            continue
        df_fold = pd.DataFrame(fm)
        print(f"  S{s:02d}  acc={df_fold['acc'].mean():.3f}  bal={df_fold['bal'].mean():.3f} (5-fold group locked)")
        all_m.append(df_fold.mean().to_dict())
        
    cols = ["acc", "bal", "f1", "auc", "mcc"] if n_class == 2 else ["acc", "bal", "f1_macro", "mcc"]
    summarize(all_m, f"SUBJECT-DEPENDENT SUMMARY ({tag})", cols)


# ═══════════════════════════════════════════
# REVISED SUBJECT-INDEPENDENT (LOSO) INTERFACE
# ═══════════════════════════════════════════
def eval_subject_independent(X, y, sub_groups, n_class=2):
    tag = "binary" if n_class == 2 else "4-class"
    print(f"\n{'='*55}\nSUBJECT-INDEPENDENT ({tag}, LOSO + CORAL + XGBoost)\n{'='*55}")
    
    Xn = X.values
    logo = LeaveOneGroupOut()
    results = []
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    for tr_i, te_i in logo.split(Xn, y_encoded, sub_groups):
        s = sub_groups[te_i[0]]
        Xtr, Xte = Xn[tr_i], Xn[te_i]
        ytr, yte = y_encoded[tr_i], y_encoded[te_i]
        
        Xtr_s, Xte_s = select(Xtr, ytr, Xte)
        
        sc = StandardScaler()
        Xtr_sc = sc.fit_transform(Xtr_s)
        Xte_sc = sc.transform(Xte_s)
        
        # Align target subject features with source subject distribution metrics
        Xte_sc = coral(Xtr_sc, Xte_sc)
        
        model = build_xgb(n_class=n_class)
        model.fit(Xtr_sc, ytr)
        yp = model.predict(Xte_sc)
        
        if n_class == 2:
            yprob = model.predict_proba(Xte_sc)[:, 1]
            mv = metrics_binary(yte, yp, yprob)
            results.append(mv)
            print(f"  S{s:02d}  acc={mv['acc']:.3f}  bal={mv['bal']:.3f}  auc={mv['auc']:.3f}")
        else:
            mv = metrics_multi(yte, yp)
            results.append(mv)
            print(f"  S{s:02d}  acc={mv['acc']:.3f}  bal={mv['bal']:.3f}")
            
    cols = ["acc", "bal", "f1", "auc", "mcc"] if n_class == 2 else ["acc", "bal", "f1_macro", "mcc"]
    summarize(results, f"SUBJECT-INDEPENDENT SUMMARY ({tag})", cols)


def main():
    print("\n" + "="*55)
    print("DREAMER High-Density Pipeline — Multi-Core XGBoost Engine")
    print("="*55)

    print("\n── Binary (valence) ──")
    Xv, yv, g_sub, g_trial = load_windowed(BINARY_DATASET)
    eval_subject_dependent(Xv, yv, g_sub, g_trial, n_class=2)
    eval_subject_independent(Xv, yv, g_sub, n_class=2)

    if QUAD_DATASET.exists():
        print("\n── 4-quadrant (valence × arousal) ──")
        Xq, yq, g_sub, g_trial = load_windowed(QUAD_DATASET)
        eval_subject_dependent(Xq, yq, g_sub, g_trial, n_class=4)
        eval_subject_independent(Xq, yq, g_sub, n_class=4)
    else:
        print(f"\n[SKIP] {QUAD_DATASET} not found — run parallel feature extraction first.")

    print("\nDone.")


if __name__ == "__main__":
    main()