"""Run model comparison across Logistic Regression, Random Forest, and Gradient Boosting.

Produces cross-validated summaries using the existing `cross_validate_model` helper.
"""

from pathlib import Path

import numpy as np

from .preprocess import load_data, split_features_target, build_pipeline
from .evaluate import cross_validate_model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_pipeline_with_classifier(X, classifier):
    # reuse preprocessing by building a pipeline and replacing the classifier
    preproc_pipeline = build_pipeline(X)
    # preproc_pipeline is Pipeline(preprocessor + classifier), so replace last step
    steps = list(preproc_pipeline.steps)
    # set classifier
    steps[-1] = ("classifier", classifier)
    return Pipeline(steps=steps)


def compare_models(n_splits=5, random_state=42):
    df = load_data()
    X, y = split_features_target(df)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=random_state),
    }

    results = {}
    for name, clf in models.items():
        pipeline = build_pipeline_with_classifier(X, clf)
        cv = cross_validate_model(pipeline, X, y, n_splits=n_splits, random_state=random_state)
        results[name] = cv

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
