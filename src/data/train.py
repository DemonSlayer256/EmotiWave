from pathlib import Path

import numpy as np
import pandas as pd

from xgboost import XGBClassifier

from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV
)

from sklearn.utils.class_weight import (
    compute_sample_weight
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

def select_features(X_train, y_train, X_test):

    selector = SelectKBest(
        score_func=mutual_info_classif,
        k=100
    )

    X_train = selector.fit_transform(
        X_train,
        y_train
    )

    X_test = selector.transform(
        X_test
    )

    return X_train, X_test, selector

DATASET = Path(
    "data/processed/features_valence.csv"
)


def load_dataset():
    print("\n[LOAD DATASET]")

    df = pd.read_csv(DATASET)

    print(
        f"Dataset Shape: {df.shape}"
    )

    return df


def prepare_data(df):
    print("\n[PREPARE DATA]")

    X = df.drop(
        columns=[
            "subject_id",
            "trial_id",
            "label"
        ]
    )

    y = df["label"]

    print(
        f"Features Shape: {X.shape}"
    )

    print(
        f"Labels Shape: {y.shape}"
    )

    print("\nLabel Distribution")

    print(
        y.value_counts()
        .sort_index()
    )

    return X, y


def hyperparameter_search(X, y):
    print(
        "\n[HYPERPARAMETER SEARCH]"
    )

    negatives = (
        y == 0
    ).sum()

    positives = (
        y == 1
    ).sum()

    scale_pos_weight = (
        negatives / positives
    )

    param_dist = {

        "n_estimators": [
            100,
            200,
            300,
            500
        ],

        "max_depth": [
            3,
            4,
            5,
            6,
            8
        ],

        "learning_rate": [
            0.01,
            0.03,
            0.05,
            0.1
        ],

        "subsample": [
            0.7,
            0.8,
            0.9,
            1.0
        ],

        "colsample_bytree": [
            0.7,
            0.8,
            0.9,
            1.0
        ]
    }

    model = XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=20,
        cv=5,
        scoring="f1",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    search.fit(
        X,
        y
    )

    print("\n[BEST PARAMETERS]")
    print(
        search.best_params_
    )

    print(
        f"Best CV F1: "
        f"{search.best_score_:.4f}"
    )

    return search.best_estimator_


def cross_validation_evaluation(
    model,
    X,
    y
):
    print(
        "\n[CROSS VALIDATION]"
    )

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    roc_scores = []

    for fold, (
        train_idx,
        test_idx
    ) in enumerate(
        skf.split(X, y),
        start=1
    ):

        print(
            f"\nFold {fold}"
        )

        X_train = X.iloc[
            train_idx
        ]

        X_test = X.iloc[
            test_idx
        ]

        y_train = y.iloc[
            train_idx
        ]

        y_test = y.iloc[
            test_idx
        ]

        weights = (
            compute_sample_weight(
                class_weight="balanced",
                y=y_train
            )
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=weights
        )

        y_pred = model.predict(
            X_test
        )

        y_prob = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        acc = accuracy_score(
            y_test,
            y_pred
        )

        prec = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        rec = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        roc = roc_auc_score(
            y_test,
            y_prob
        )

        accuracy_scores.append(
            acc
        )

        precision_scores.append(
            prec
        )

        recall_scores.append(
            rec
        )

        f1_scores.append(
            f1
        )

        roc_scores.append(
            roc
        )

        print(
            f"Accuracy={acc:.4f} "
            f"Precision={prec:.4f} "
            f"Recall={rec:.4f} "
            f"F1={f1:.4f} "
            f"ROC_AUC={roc:.4f}"
        )

    print(
        "\n[FINAL RESULTS]"
    )

    print(
        f"Accuracy  : "
        f"{np.mean(accuracy_scores):.4f} "
        f"± {np.std(accuracy_scores):.4f}"
    )

    print(
        f"Precision : "
        f"{np.mean(precision_scores):.4f} "
        f"± {np.std(precision_scores):.4f}"
    )

    print(
        f"Recall    : "
        f"{np.mean(recall_scores):.4f} "
        f"± {np.std(recall_scores):.4f}"
    )

    print(
        f"F1 Score  : "
        f"{np.mean(f1_scores):.4f} "
        f"± {np.std(f1_scores):.4f}"
    )

    print(
        f"ROC AUC   : "
        f"{np.mean(roc_scores):.4f} "
        f"± {np.std(roc_scores):.4f}"
    )


def main():
    print(
        "\nTraining Valence Classifier"
    )

    df = load_dataset()

    X, y = prepare_data(
        df
    )

    best_model = (
        hyperparameter_search(
            X,
            y
        )
    )

    cross_validation_evaluation(
        best_model,
        X,
        y
    )

    print(
        "\nTraining Complete"
    )


if __name__ == "__main__":
    main()