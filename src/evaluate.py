"""Evaluation helpers for the heart disease prediction model.

This module provides a structured cross-validation reporting function that
computes per-fold metrics (accuracy, precision, recall, f1, roc_auc), summary
statistics (mean/std/min/max) and the class distribution observed in each
validation fold. It also exposes a simple hold-out evaluation helper.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def cross_validate_model(
    model,
    X,
    y,
    n_splits: int = 5,
    random_state: int = 42,
    scoring: dict | None = None,
):
    """Run stratified K-fold CV and return per-metric fold scores and summaries.

    scoring: dictionary of named scoring metrics or None to use a sensible default.
    Returns a dictionary with per-metric fold arrays and summary statistics.
    """
    if scoring is None:
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    cv_result = cross_validate(
        estimator=model,
        X=X,
        y=y,
        scoring=scoring,
        cv=cv,
        return_train_score=False,
        n_jobs=-1,
    )

    # Build a metrics summary
    metrics = {}
    for key in cv_result:
        if not key.startswith("test_"):
            continue
        metric_name = key.replace("test_", "")
        values = np.asarray(cv_result[key])
        metrics[metric_name] = {
            "folds": values.tolist(),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }

    # verify class distribution in validation folds
    fold_distributions = []
    for train_idx, val_idx in cv.split(X, y):
        y_val = np.asarray(y)[val_idx]
        unique, counts = np.unique(y_val, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
        fold_distributions.append(dist)

    return {
        "n_splits": n_splits,
        "metrics": metrics,
        "fold_class_distribution": fold_distributions,
        # Backwards-compatible keys used by older tests / callers
        "fold_scores": metrics.get("accuracy", {}).get("folds", []),
        "mean_score": metrics.get("accuracy", {}).get("mean", None),
        "std_score": metrics.get("accuracy", {}).get("std", None),
    }


def evaluate_model(model, X_test, y_test) -> dict[str, float | object]:
    """Evaluate a trained model and print a concise summary."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(y_test, predictions),
    }

    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print("Classification Report:")
    print(metrics["classification_report"])

    return metrics


def plot_confusion_matrix(cm, save_path: Optional[Path | str] = None) -> None:
    """Visualize the confusion matrix for the trained model."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    fig.tight_layout()

    if save_path is not None:
        target_path = Path(save_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target_path, bbox_inches="tight")

    plt.close(fig)
