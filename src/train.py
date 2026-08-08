"""Train the heart disease prediction model and save the artifact."""

from pathlib import Path

import joblib

try:
    from .evaluate import cross_validate_model, evaluate_model
    from .preprocess import (
        build_pipeline,
        load_data,
        split_features_target,
        split_train_test,
    )
except ImportError:  # pragma: no cover - script execution fallback
    from evaluate import cross_validate_model, evaluate_model
    from preprocess import (
        build_pipeline,
        load_data,
        split_features_target,
        split_train_test,
    )

MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "logistic_regression.joblib"
)


def train_model(data_path=None, model_path=None):
    """Load data, train the model, evaluate it, and save the trained artifact."""
    df = load_data(data_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    model = build_pipeline(X_train)

    cv_results = cross_validate_model(
        model,
        X_train,
        y_train,
        n_splits=5,
        random_state=42,
        scoring="accuracy",
    )

    print("Cross-validation results:")
    for index, score in enumerate(cv_results["fold_scores"], start=1):
        print(f"Fold {index}: {score:.4f}")
    print(f"Mean CV accuracy: {cv_results['mean_score']:.4f}")
    print(f"Std CV accuracy: {cv_results['std_score']:.4f}")

    model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    metrics["cv_results"] = cv_results

    target_path = Path(model_path or MODEL_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target_path)

    return {"model": model, "model_path": target_path, "metrics": metrics}


def main():
    results = train_model()
    print(f"Saved model to {results['model_path']}")


if __name__ == "__main__":
    main()
