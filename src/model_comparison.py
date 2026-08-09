"""Model comparison utilities.

Run multiple candidate estimators on identical StratifiedKFold splits
and produce a per-model, per-metric comparison report.

This module intentionally does not alter the preprocessing pipeline in
`src.preprocess` — it composes the project's preprocessor with each
estimator under test to ensure a fair comparison.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from .preprocess import build_preprocessor


DEFAULT_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def _summarize_cv_result(cv_result: dict) -> dict:
    metrics: dict = {}
    for key, values in cv_result.items():
        if not key.startswith("test_"):
            continue
        metric_name = key.replace("test_", "")
        arr = np.asarray(values)
        metrics[metric_name] = {
            "folds": arr.tolist(),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return metrics


def compare_models(
    X,
    y,
    models: Optional[Dict[str, object]] = None,
    n_splits: int = 5,
    random_state: int = 42,
    scoring: Optional[dict] = None,
    n_jobs: int = 1,
):
    """Compare multiple estimators using the same StratifiedKFold splits.

    Returns a dictionary keyed by model display name. Each entry contains a
    `metrics` dict (per-metric folds and summaries) and the raw `cv_result`.
    """
    if scoring is None:
        scoring = DEFAULT_SCORING

    if models is None:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
        }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # compute fold class distributions once for reporting
    fold_distributions = []
    for train_idx, val_idx in cv.split(X, y):
        y_val = np.asarray(y)[val_idx]
        unique, counts = np.unique(y_val, return_counts=True)
        dist = {int(k): int(v) for k, v in zip(unique.tolist(), counts.tolist())}
        fold_distributions.append(dist)

    results: dict = {"n_splits": n_splits, "fold_class_distribution": fold_distributions, "models": {}}

    for name, estimator in models.items():
        # compose preprocessor + estimator so preprocessing is identical
        preprocessor = build_preprocessor(X)
        pipe = Pipeline([("preprocessor", preprocessor), ("classifier", estimator)])

        cv_result = cross_validate(
            estimator=pipe,
            X=X,
            y=y,
            scoring=scoring,
            cv=cv,
            return_train_score=False,
            n_jobs=n_jobs,
        )

        metrics = _summarize_cv_result(cv_result)

        results["models"][name] = {"metrics": metrics, "cv_result": cv_result}

    return results


def print_comparison_report(results: dict) -> None:
    """Pretty-print a compact comparison table of mean±std for each metric."""
    models = results.get("models", {})
    if not models:
        print("No models to report")
        return

    metric_names = list(next(iter(models.values()))["metrics"].keys())

    header = "Model".ljust(20) + "  " + "  ".join([m.ljust(20) for m in metric_names])
    print(header)
    print("-" * len(header))

    for name, info in models.items():
        row = name.ljust(20) + "  "
        for m in metric_names:
            stat = info["metrics"][m]
            row += f"{stat['mean']:.3f}±{stat['std']:.3f}".ljust(20) + "  "
        print(row)
