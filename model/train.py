from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from model.splits import panel_walk_forward_splits

RANDOM_STATE = 10


def run_walk_forward(panel_df: pd.DataFrame, feature_columns: list[str], n_splits: int = 5, embargo: int = 0):
    """Walk-forward CV over panel_df, mirroring predict_stock.py's existing
    fold-loop/baseline pattern generalized to panel data via
    panel_walk_forward_splits. If feature_columns is empty (no factor passed
    validation yet), skips the RandomForest fit — sklearn can't fit zero
    features — and just reports the target's class balance and per-fold
    majority-class baseline, still exercising panel/target/split machinery
    end-to-end.

    embargo: dates dropped between train and test per fold; set to the forward
    horizon so a multi-day Target can't leak across the boundary.
    """
    Y = panel_df["Target"].astype(int)
    print(f"Target class balance: {Y.value_counts(normalize=True).round(3).to_dict()}")

    fold_accuracies, baseline_accuracies = [], []
    for fold, (train_idx, test_idx) in enumerate(
        panel_walk_forward_splits(panel_df, n_splits=n_splits, embargo=embargo), start=1
    ):
        y_train, y_test = Y.loc[train_idx], Y.loc[test_idx]
        majority_class = y_train.mode()[0]
        baseline_pred = np.full(len(y_test), majority_class)
        baseline_acc = accuracy_score(y_test, baseline_pred)
        baseline_accuracies.append(baseline_acc)

        if not feature_columns:
            print(f"Fold {fold}: majority-class baseline={baseline_acc:.3f} "
                  f"(n_train={len(y_train)}, n_test={len(y_test)}, no features — fit skipped)")
            continue

        X_train = panel_df.loc[train_idx, feature_columns]
        X_test = panel_df.loc[test_idx, feature_columns]

        model = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        fold_acc = accuracy_score(y_test, y_pred)
        fold_accuracies.append(fold_acc)
        print(f"Fold {fold}: model accuracy={fold_acc:.3f}, majority-class baseline={baseline_acc:.3f}")

    if not feature_columns:
        print("\nNo validated feature columns — RandomForest fit skipped for all folds. "
              "Panel/target/split machinery verified working; revisit once a factor passes validation.")
        return None

    print(f"\nMean walk-forward accuracy: model={np.mean(fold_accuracies):.3f}, "
          f"baseline={np.mean(baseline_accuracies):.3f}")
    return fold_accuracies, baseline_accuracies


def run_final_holdout(panel_df: pd.DataFrame, feature_columns: list[str], test_size: float = 0.3, embargo: int = 0):
    """Final chronological holdout (most recent test_size fraction of unique
    dates, not shuffled) for a detailed classification report. Skipped if
    feature_columns is empty, same reasoning as run_walk_forward.

    embargo: unique dates dropped between the train block and the test block
    (set to the forward horizon) so the last train rows' multi-day labels don't
    peek into the test window."""
    if not feature_columns:
        print("No validated feature columns — skipping final holdout model fit.")
        return None

    dates = np.sort(panel_df["Date"].unique())
    split_at = int(len(dates) * (1 - test_size))
    # Embargo: drop `embargo` dates at the end of train so its forward-looking
    # labels can't reach into the test block.
    train_dates = set(dates[: max(0, split_at - embargo)])
    test_dates = set(dates[split_at:])

    train_df = panel_df[panel_df["Date"].isin(train_dates)]
    test_df = panel_df[panel_df["Date"].isin(test_dates)]

    X_train, y_train = train_df[feature_columns], train_df["Target"].astype(int)
    X_test, y_test = test_df[feature_columns], test_df["Target"].astype(int)

    model = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    majority_class = y_train.mode()[0]
    baseline_pred = np.full(len(y_test), majority_class)

    print("\nFinal holdout:")
    print("Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Majority-class Baseline Accuracy:", accuracy_score(y_test, baseline_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    return model
