"""Evaluation helpers for the heart disease prediction model."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score


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
