from pathlib import Path

from src.model_comparison import compare_models
from src.preprocess import load_data, split_features_target

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "heart.csv"


def test_compare_models_returns_metrics_for_all_models():
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    results = compare_models(X, y, n_splits=3, random_state=42, n_jobs=1)

    assert results["n_splits"] == 3
    assert "fold_class_distribution" in results
    assert len(results["fold_class_distribution"]) == 3
    assert "models" in results
    assert set(results["models"].keys()) == {
        "Logistic Regression",
        "Random Forest",
        "Gradient Boosting",
    }

    for name, info in results["models"].items():
        assert "metrics" in info
        assert "accuracy" in info["metrics"]
        assert len(info["metrics"]["accuracy"]["folds"]) == 3
        assert 0.0 <= info["metrics"]["accuracy"]["mean"] <= 1.0


def test_print_comparison_report_outputs_table(capsys):
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)
    results = compare_models(X, y, n_splits=3, random_state=42, n_jobs=1)

    from src.model_comparison import print_comparison_report

    print_comparison_report(results)
    captured = capsys.readouterr()

    assert "Logistic Regression" in captured.out
    assert "Random Forest" in captured.out
    assert "Gradient Boosting" in captured.out
    assert "accuracy" in captured.out
    assert "precision" in captured.out
