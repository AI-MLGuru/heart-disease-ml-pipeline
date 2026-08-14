from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dataset_analysis import (
    composition_summary,
    duplicate_summary,
    generate_dataset_report,
    load_unified_dataset,
    missingness_summary,
    source_missingness_summary,
    target_distribution,
)


def test_composition_summary_counts_sources():
    df = pd.DataFrame(
        {
            "source_dataset": ["a", "a", "b"],
            "age": [1, 2, 3],
            "sex": [1, 0, 1],
            "cp": [1, 2, 3],
            "trestbps": [120, 130, 140],
            "chol": [200, 210, 220],
            "fbs": [0, 1, 0],
            "restecg": [0, 1, 0],
            "thalach": [150, 160, 155],
            "exang": [0, 1, 0],
            "oldpeak": [1.0, 2.0, 0.5],
            "slope": [2, 3, 2],
            "ca": [0, 1, 0],
            "thal": [3, 6, 7],
            "target": [0, 1, 0],
        }
    )
    summary = composition_summary(df)
    assert summary["total_rows"] == 3
    assert summary["source_counts"] == {"a": 2, "b": 1}


def test_target_distribution_computes_prevalence():
    df = pd.DataFrame(
        {"target": [0, 1, 1, 0, 1]},
    )
    distribution = target_distribution(df)
    assert distribution["positive"] == 3
    assert distribution["negative"] == 2
    assert distribution["prevalence"] == 0.6


def test_missingness_summary_reports_all_features():
    df = pd.DataFrame(
        {
            "age": [1, None],
            "sex": [1, 0],
            "cp": [None, 2],
            "trestbps": [120, None],
            "chol": [200, 210],
            "fbs": [0, 1],
            "restecg": [0, 1],
            "thalach": [150, 160],
            "exang": [0, 1],
            "oldpeak": [1.0, 2.0],
            "slope": [2, 3],
            "ca": [0, 1],
            "thal": [3, 6],
            "target": [0, 1],
        }
    )
    report = missingness_summary(df)
    assert report.loc["age", "missing_count"] == 1
    assert report.loc["cp", "missing_count"] == 1
    assert report.loc["trestbps", "missing_count"] == 1


def test_source_missingness_summary_reports_by_source():
    df = pd.DataFrame(
        {
            "source_dataset": ["a", "a", "b"],
            "age": [1, None, 3],
            "sex": [1, 0, 1],
            "cp": [1, 2, 3],
            "trestbps": [120, 130, 140],
            "chol": [200, 210, 220],
            "fbs": [0, 1, 0],
            "restecg": [0, 1, 0],
            "thalach": [150, 160, 155],
            "exang": [0, 1, 0],
            "oldpeak": [1.0, 2.0, 0.5],
            "slope": [2, 3, 2],
            "ca": [0, 1, 0],
            "thal": [3, 6, 7],
            "target": [0, 1, 0],
        }
    )
    report = source_missingness_summary(df)
    assert report.loc[report["source_dataset"] == "a", "age_missing_count"].iloc[0] == 1
    assert report.loc[report["source_dataset"] == "b", "age_missing_count"].iloc[0] == 0


def test_duplicate_summary_counts_exact_and_cross_source():
    df = pd.DataFrame(
        {
            "source_dataset": ["a", "b", "b"],
            "age": [1, 1, 1],
            "sex": [1, 1, 1],
            "cp": [1, 1, 1],
            "trestbps": [120, 120, 120],
            "chol": [200, 200, 200],
            "fbs": [0, 0, 0],
            "restecg": [0, 0, 0],
            "thalach": [150, 150, 150],
            "exang": [0, 0, 0],
            "oldpeak": [1.0, 1.0, 1.0],
            "slope": [2, 2, 2],
            "ca": [0, 0, 0],
            "thal": [3, 3, 3],
            "target": [0, 0, 0],
        }
    )
    result = duplicate_summary(df)
    assert result["exact_duplicate_rows"] == 3
    assert result["cross_source_duplicate_rows"] == 3


def test_generate_dataset_report_writes_files(tmp_path: Path):
    df = pd.DataFrame(
        {
            "source_dataset": ["a", "b"],
            "age": [1, 2],
            "sex": [1, 0],
            "cp": [1, 2],
            "trestbps": [120, 130],
            "chol": [200, 220],
            "fbs": [0, 1],
            "restecg": [0, 1],
            "thalach": [150, 160],
            "exang": [0, 1],
            "oldpeak": [1.0, 2.0],
            "slope": [2, 3],
            "ca": [0, 1],
            "thal": [3, 6],
            "target": [0, 1],
        }
    )
    report = generate_dataset_report(df, output_dir=tmp_path)
    assert Path(report["report_files"]["json"]).exists()
    assert Path(report["report_files"]["markdown"]).exists()


def test_load_unified_dataset_includes_all_sources():
    df = load_unified_dataset()
    assert set(df["source_dataset"].unique()) == {
        "uci_hd_cleveland",
        "uci_hd_hungarian",
        "uci_hd_switzerland",
        "uci_hd_va",
    }
    assert len(df) == 303 + 294 + 123 + 200
