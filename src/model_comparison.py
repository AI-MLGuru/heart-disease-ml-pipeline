"""Run model comparison across Logistic Regression, Random Forest, and Gradient Boosting.

Produces cross-validated summaries using the existing `cross_validate_model` helper.
"""

from pathlib import Path

import numpy as np

from .preprocess import load_data, split_features_target, build_preprocessor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
import numpy as np



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
    models: dict | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    scoring: dict | None = None,
    n_jobs: int = 1,
):
    """Compare multiple estimators using the same StratifiedKFold splits.

    Returns a dict containing `n_splits`, `fold_class_distribution`, and a
    `models` mapping where each model entry has `metrics` and the raw `cv_result`.
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


def print_results(results):
    for name, res in results.items():
        print(f"Model: {name}")
        for metric, stats in res["metrics"].items():
            mean = stats["mean"]
            std = stats["std"]
            print(f"  {metric:8s}: mean={mean:.4f} std={std:.4f}")
        print()


def main():
    results = compare_models()
    print_results(results)


if __name__ == "__main__":
    main()
