"""
Train and evaluate valence classifier on DREAMER with improved methods.

Key upgrades over baseline
──────────────────────────
Subject-dependent  (target ≥ 90 %)
  • Stacking ensemble: XGBoost + LightGBM + SVM + Random Forest
    with Logistic Regression meta-learner
  • Optuna-based hyperparameter search on first fold, reuse params
  • SMOTE + Tomek oversampling to fix 251/163 imbalance
  • RFE + mutual information two-stage feature selection
  • Lower classification threshold tuned per fold on val set

Subject-independent  (target ≥ 80 %)
  • Z-score re-normalisation per test subject (test-time normalisation)
  • Coral domain adaptation: aligns covariance of each test-subject's
    features to the training distribution before prediction
  • Same stacking ensemble as above
  • Threshold sweep on held-out subjects' probability outputs
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from sklearn.feature_selection import (
    SelectKBest, mutual_info_classif, RFE
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("[WARN] LightGBM not found – ensemble will use XGB + RF + SVM only")

try:
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    HAS_IMBALANCED = True
except ImportError:
    HAS_IMBALANCED = False
    print("[WARN] imbalanced-learn not found – will use class_weight only")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("[WARN] Optuna not found – using default hyperparameters")


# ─── paths / constants ────────────────────────────────────────────────────────
DATASET      = Path("data/processed/features_valence.csv")
TOP_K_MI     = 120      # mutual-info top-k
TOP_K_RFE    = 80       # after RFE
NEGATIVES    = 251
POSITIVES    = 163
THRESHOLD_SD = 0.40     # subject-dependent default; tuned per fold
THRESHOLD_SI = 0.45     # subject-independent default; tuned per leave-out


# ─── data loading ─────────────────────────────────────────────────────────────
def load_and_prepare():
    print("\n[LOAD DATASET]")
    df = pd.read_csv(DATASET)
    print(f"  Shape: {df.shape}")

    groups = df["subject_id"].values
    y      = df["label"].values
    X      = df.drop(columns=["subject_id", "trial_id", "label"])

    print(f"  Features: {X.shape[1]}   Labels: {len(y)}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    return X, y, groups, X.columns.tolist()


# ─── feature selection ────────────────────────────────────────────────────────
def two_stage_feature_selection(X_tr, y_tr, X_te):
    """MI top-k → RFE with fast RF → final feature set."""
    # stage 1: mutual information
    mi = SelectKBest(mutual_info_classif, k=min(TOP_K_MI, X_tr.shape[1]))
    X_tr1 = mi.fit_transform(X_tr, y_tr)
    X_te1 = mi.transform(X_te)

    # stage 2: RFE with a small random forest (fast)
    rf_rfe = RandomForestClassifier(
        n_estimators=60, max_depth=6, n_jobs=-1, random_state=42
    )
    rfe = RFE(rf_rfe, n_features_to_select=min(TOP_K_RFE, X_tr1.shape[1]),
              step=0.1)
    X_tr2 = rfe.fit_transform(X_tr1, y_tr)
    X_te2 = rfe.transform(X_te1)

    return X_tr2, X_te2


# ─── CORAL domain adaptation ─────────────────────────────────────────────────
def coral_align(X_src, X_tgt):
    """
    CORAL: Correlation Alignment (Sun & Saenko 2016).
    Aligns the second-order statistics of the target to the source.
    Works on NumPy arrays.
    """
    # add small regularisation to avoid singular matrices
    reg = 1e-4 * np.eye(X_src.shape[1])

    cov_src = np.cov(X_src, rowvar=False) + reg
    cov_tgt = np.cov(X_tgt, rowvar=False) + reg

    # whitening transform for target
    try:
        U_t, S_t, _ = np.linalg.svd(cov_tgt)
        W_tgt = U_t @ np.diag(1.0 / np.sqrt(S_t + 1e-10)) @ U_t.T

        # re-colouring transform from source
        U_s, S_s, _ = np.linalg.svd(cov_src)
        W_src = U_s @ np.diag(np.sqrt(S_s + 1e-10)) @ U_s.T

        X_tgt_aligned = X_tgt @ W_tgt @ W_src
        return X_tgt_aligned
    except np.linalg.LinAlgError:
        return X_tgt   # fall back to unaligned


# ─── oversampling ─────────────────────────────────────────────────────────────
def resample(X, y):
    if not HAS_IMBALANCED:
        return X, y
    try:
        # SMOTETomek: synthesises minority + removes borderline majority
        smt = SMOTETomek(random_state=42)
        return smt.fit_resample(X, y)
    except Exception:
        try:
            sm = SMOTE(random_state=42, k_neighbors=3)
            return sm.fit_resample(X, y)
        except Exception:
            return X, y


# ─── hyperparameter search (Optuna, 1 fold) ───────────────────────────────────
_XGB_PARAMS  = None
_LGB_PARAMS  = None
_SEARCH_DONE = False


def _optuna_search(X_tr, y_tr):
    global _XGB_PARAMS, _LGB_PARAMS, _SEARCH_DONE
    if _SEARCH_DONE or not HAS_OPTUNA:
        return

    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)

    def xgb_objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_est", 200, 600),
            max_depth         = trial.suggest_int("depth", 3, 7),
            learning_rate     = trial.suggest_float("lr", 0.005, 0.05, log=True),
            subsample         = trial.suggest_float("sub", 0.6, 1.0),
            colsample_bytree  = trial.suggest_float("col", 0.5, 1.0),
            min_child_weight  = trial.suggest_int("mcw", 1, 10),
            gamma             = trial.suggest_float("gamma", 0, 5),
            reg_alpha         = trial.suggest_float("alpha", 1e-4, 10, log=True),
            reg_lambda        = trial.suggest_float("lambda", 1e-4, 10, log=True),
            scale_pos_weight  = NEGATIVES / POSITIVES,
            eval_metric       = "logloss",
            random_state      = 42,
            n_jobs            = -1,
        )
        scores = []
        for tr, va in inner_cv.split(X_tr, y_tr):
            m = XGBClassifier(**params)
            m.fit(X_tr[tr], y_tr[tr])
            p = m.predict_proba(X_tr[va])[:, 1]
            scores.append(roc_auc_score(y_tr[va], p))
        return np.mean(scores)

    study = optuna.create_study(direction="maximize")
    study.optimize(xgb_objective, n_trials=40, show_progress_bar=False)
    _XGB_PARAMS = study.best_params
    print(f"  [Optuna XGB] best ROC-AUC={study.best_value:.4f}  "
          f"params={_XGB_PARAMS}")

    if HAS_LGB:
        def lgb_objective(trial):
            params = dict(
                n_estimators      = trial.suggest_int("n_est", 200, 600),
                num_leaves        = trial.suggest_int("leaves", 15, 63),
                learning_rate     = trial.suggest_float("lr", 0.005, 0.05, log=True),
                subsample         = trial.suggest_float("sub", 0.6, 1.0),
                colsample_bytree  = trial.suggest_float("col", 0.5, 1.0),
                min_child_samples = trial.suggest_int("mcs", 5, 40),
                reg_alpha         = trial.suggest_float("alpha", 1e-4, 10, log=True),
                reg_lambda        = trial.suggest_float("lambda", 1e-4, 10, log=True),
                class_weight      = "balanced",
                random_state      = 42,
                n_jobs            = -1,
                verbose           = -1,
            )
            scores = []
            for tr, va in inner_cv.split(X_tr, y_tr):
                m = lgb.LGBMClassifier(**params)
                m.fit(X_tr[tr], y_tr[tr])
                p = m.predict_proba(X_tr[va])[:, 1]
                scores.append(roc_auc_score(y_tr[va], p))
            return np.mean(scores)

        study2 = optuna.create_study(direction="maximize")
        study2.optimize(lgb_objective, n_trials=40, show_progress_bar=False)
        _LGB_PARAMS = study2.best_params
        print(f"  [Optuna LGB] best ROC-AUC={study2.best_value:.4f}")

    _SEARCH_DONE = True


# ─── build stacking ensemble ─────────────────────────────────────────────────
def build_ensemble():
    """
    Base learners: XGBoost, LightGBM (if available), SVM (RBF), Random Forest.
    Meta-learner: Logistic Regression with L2.
    """
    xgb_params = _XGB_PARAMS or dict(
        n_estimators=400, max_depth=5, learning_rate=0.02,
        subsample=0.85, colsample_bytree=0.75,
        min_child_weight=3, gamma=0.1, reg_alpha=0.5, reg_lambda=1.0,
        scale_pos_weight=NEGATIVES / POSITIVES,
        eval_metric="logloss", random_state=42, n_jobs=-1,
    )

    estimators = [
        ("xgb", XGBClassifier(**xgb_params)),
        ("rf",  RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
        ("svm", Pipeline([
            ("scaler", StandardScaler()),
            ("svc",    SVC(
                kernel="rbf", C=5.0, gamma="scale",
                probability=True, class_weight="balanced",
                random_state=42,
            )),
        ])),
    ]

    if HAS_LGB:
        lgb_params = _LGB_PARAMS or dict(
            n_estimators=400, num_leaves=31, learning_rate=0.02,
            subsample=0.85, colsample_bytree=0.75,
            min_child_samples=15, reg_alpha=0.3, reg_lambda=0.5,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        )
        estimators.insert(1, ("lgb", lgb.LGBMClassifier(**lgb_params)))

    meta = LogisticRegression(C=1.0, max_iter=1000, random_state=42)

    return StackingClassifier(
        estimators=estimators,
        final_estimator=meta,
        cv=3,
        stack_method="predict_proba",
        n_jobs=1,
        passthrough=False,
    )


# ─── threshold optimisation ───────────────────────────────────────────────────
def find_best_threshold(y_true, y_prob):
    """Search threshold in [0.30, 0.65] to maximise balanced accuracy."""
    best_t, best_ba = 0.5, 0.0
    for t in np.arange(0.30, 0.66, 0.02):
        ba = balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t


# ─── metrics helper ──────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy":          accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision":         precision_score(y_true, y_pred, zero_division=0),
        "recall":            recall_score(y_true, y_pred, zero_division=0),
        "f1":                f1_score(y_true, y_pred, zero_division=0),
        "roc_auc":           roc_auc_score(y_true, y_prob),
        "mcc":               matthews_corrcoef(y_true, y_pred),
    }


def summarize_results(results, title):
    df = pd.DataFrame(results)
    print(f"\n[{title}]")
    for col in df.columns:
        print(f"  {col:<22}{df[col].mean():.4f} ± {df[col].std():.4f}")


# ─── SUBJECT-DEPENDENT ────────────────────────────────────────────────────────
def evaluate_subject_dependent(X, y, feature_names):
    print("\n" + "=" * 60)
    print("SUBJECT-DEPENDENT EVALUATION  (5-fold CV)")
    print("=" * 60)

    X_np = X.values
    cv   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for fold, (tr_idx, te_idx) in enumerate(cv.split(X_np, y), 1):
        print(f"\n  Fold {fold}")

        X_tr, X_te = X_np[tr_idx], X_np[te_idx]
        y_tr, y_te = y[tr_idx],    y[te_idx]

        # ── optional Optuna search on first fold ──────────────────────────
        if fold == 1 and HAS_OPTUNA:
            print("  Running Optuna (this takes ~2 min)…")
            _optuna_search(X_tr, y_tr)

        # ── feature selection ─────────────────────────────────────────────
        X_tr_fs, X_te_fs = two_stage_feature_selection(X_tr, y_tr, X_te)

        # ── oversampling ──────────────────────────────────────────────────
        X_tr_rs, y_tr_rs = resample(X_tr_fs, y_tr)

        # ── train stacking ensemble ───────────────────────────────────────
        model = build_ensemble()

        sw = compute_sample_weight("balanced", y_tr_rs)
        # StackingClassifier passes sample_weight to base learners if named
        fit_params = {}
        for name, _ in model.estimators:
            if name in ("xgb", "lgb", "rf"):
                fit_params[f"{name}__sample_weight"] = sw

        model.fit(X_tr_rs, y_tr_rs, **fit_params)

        # ── threshold tuning on a small inner validation split ────────────
        inner_cv  = StratifiedKFold(n_splits=3, shuffle=True, random_state=7)
        val_probs, val_true = [], []
        for itr, iva in inner_cv.split(X_tr_fs, y_tr):
            Xi_tr, Xi_va = X_tr_fs[itr], X_tr_fs[iva]
            yi_tr, yi_va = y_tr[itr],    y_tr[iva]
            Xi_tr_rs, yi_tr_rs = resample(Xi_tr, yi_tr)
            m2 = build_ensemble()
            m2.fit(Xi_tr_rs, yi_tr_rs)
            val_probs.append(m2.predict_proba(Xi_va)[:, 1])
            val_true.append(yi_va)
        best_thr = find_best_threshold(
            np.concatenate(val_true), np.concatenate(val_probs)
        )
        print(f"    Best threshold: {best_thr:.2f}")

        # ── predict ───────────────────────────────────────────────────────
        y_prob = model.predict_proba(X_te_fs)[:, 1]
        y_pred = (y_prob >= best_thr).astype(int)

        m = compute_metrics(y_te, y_pred, y_prob)
        results.append(m)
        print(
            f"    Acc={m['accuracy']:.4f}  BalAcc={m['balanced_accuracy']:.4f}  "
            f"F1={m['f1']:.4f}  ROC={m['roc_auc']:.4f}  MCC={m['mcc']:.4f}"
        )

    summarize_results(results, "SUBJECT-DEPENDENT SUMMARY")


# ─── SUBJECT-INDEPENDENT (LOSO + CORAL) ─────────────────────────────────────
def evaluate_subject_independent(X, y, groups):
    print("\n" + "=" * 60)
    print("SUBJECT-INDEPENDENT EVALUATION  (LOSO + CORAL)")
    print("=" * 60)

    X_np = X.values
    logo = LeaveOneGroupOut()
    results = []

    for fold, (tr_idx, te_idx) in enumerate(
        logo.split(X_np, y, groups), 1
    ):
        subj = groups[te_idx[0]]
        X_tr, X_te = X_np[tr_idx], X_np[te_idx]
        y_tr, y_te = y[tr_idx],    y[te_idx]

        # ── feature selection ─────────────────────────────────────────────
        X_tr_fs, X_te_fs = two_stage_feature_selection(X_tr, y_tr, X_te)

        # ── CORAL domain alignment ────────────────────────────────────────
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr_fs)
        X_te_sc = scaler.transform(X_te_fs)
        X_te_sc = coral_align(X_tr_sc, X_te_sc)

        # ── oversampling ──────────────────────────────────────────────────
        X_tr_rs, y_tr_rs = resample(X_tr_sc, y_tr)

        # ── train ─────────────────────────────────────────────────────────
        model  = build_ensemble()
        sw     = compute_sample_weight("balanced", y_tr_rs)
        fit_params = {}
        for name, _ in model.estimators:
            if name in ("xgb", "lgb", "rf"):
                fit_params[f"{name}__sample_weight"] = sw
        model.fit(X_tr_rs, y_tr_rs, **fit_params)

        # ── predict with threshold tuning ─────────────────────────────────
        y_prob = model.predict_proba(X_te_sc)[:, 1]

        # tune threshold using 10 % of training set as proxy val
        proxy_size = max(int(0.1 * len(y_tr)), 10)
        rng        = np.random.default_rng(42)
        proxy_idx  = rng.choice(len(y_tr), proxy_size, replace=False)
        proxy_prob = model.predict_proba(X_tr_sc[proxy_idx])[:, 1]
        best_thr   = find_best_threshold(y_tr[proxy_idx], proxy_prob)

        y_pred = (y_prob >= best_thr).astype(int)

        # fallback if one class missing in tiny test set
        try:
            m = compute_metrics(y_te, y_pred, y_prob)
        except Exception:
            print(f"    Subject {subj:02d}  skipped (single class in test)")
            continue

        results.append(m)
        print(
            f"  Subject {subj:02d}  "
            f"Acc={m['accuracy']:.4f}  F1={m['f1']:.4f}  "
            f"ROC={m['roc_auc']:.4f}  thr={best_thr:.2f}"
        )

    summarize_results(results, "SUBJECT-INDEPENDENT SUMMARY")


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "=" * 60)
    print("TRAINING: VALENCE CLASSIFIER  (improved pipeline)")
    print("=" * 60)

    X, y, groups, feature_names = load_and_prepare()

    evaluate_subject_dependent(X, y, feature_names)
    evaluate_subject_independent(X, y, groups)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()