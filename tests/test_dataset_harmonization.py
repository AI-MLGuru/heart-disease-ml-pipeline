from __future__ import annotations

import pandas as pd

from src.dataset_harmonization import (
    CANONICAL_FEATURES,
    PROVENANCE_COLUMNS,
    TARGET_COLUMN,
    build_unified_dataset,
    detect_cross_source_duplicates,
    detect_exact_duplicates,
    handle_missing_values,
    harmonize_dataset,
    harmonize_target,
    normalize_schema,
    validate_harmonized_dataset,
)


def test_harmonize_target_binary_mapping():
    raw = pd.Series(["0", "1", "2", "4", "?", "NA", "", "-9", "3"])
    result = harmonize_target(raw)

    assert result.dtype.name == "Int64"
    assert result.iloc[0] == 0
    assert result.iloc[1] == 1
    assert result.iloc[2] == 1
    assert result.iloc[3] == 1
    assert pd.isna(result.iloc[4])
    assert pd.isna(result.iloc[5])
    assert pd.isna(result.iloc[6])
    assert pd.isna(result.iloc[7])
    assert result.iloc[8] == 1


def test_normalize_schema_produces_canonical_columns():
    raw = pd.DataFrame(
        [
            [63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 0],
        ],
        columns=list(range(14)),
    )
    raw["source_collection"] = "uci_heart_disease_45"
    raw["source_dataset"] = "uci_hd_cleveland"
    raw["source_file"] = "processed.cleveland.data"
    raw["source_row_id"] = 0

    normalized = normalize_schema(raw)
    assert list(normalized.columns)[: len(CANONICAL_FEATURES)] == CANONICAL_FEATURES
    assert normalized.columns[-4:].tolist() == PROVENANCE_COLUMNS
    assert normalized.loc[0, TARGET_COLUMN] == 0


def test_handle_missing_values_replaces_question_marks_with_nan():
    raw = pd.DataFrame(
        {
            "age": ["63", "?"],
            "sex": ["1", "0"],
            "cp": ["1", "2"],
            "trestbps": ["145", "?"],
            "chol": ["233", "?"],
            "fbs": ["1", "0"],
            "restecg": ["2", "0"],
            "thalach": ["150", "?"],
            "exang": ["0", "1"],
            "oldpeak": ["2.3", "?"],
            "slope": ["3", "2"],
            "ca": ["0", "?"],
            "thal": ["6", "?"],
            "target": ["0", "1"],
        }
    )

    cleaned = handle_missing_values(raw)
    assert pd.isna(cleaned.loc[1, "age"])
    assert pd.isna(cleaned.loc[1, "trestbps"])
    assert pd.isna(cleaned.loc[1, "chol"])
    assert pd.isna(cleaned.loc[1, "thalach"])
    assert pd.isna(cleaned.loc[1, "oldpeak"])
    assert pd.isna(cleaned.loc[1, "ca"])
    assert pd.isna(cleaned.loc[1, "thal"])
    assert cleaned.loc[0, "age"] == 63
    assert cleaned.loc[0, "chol"] == 233


def test_harmonize_dataset_preserves_provenance_and_binary_target():
    df = harmonize_dataset("uci_hd_cleveland")

    assert set(PROVENANCE_COLUMNS).issubset(df.columns)
    assert set(CANONICAL_FEATURES + [TARGET_COLUMN]).issubset(df.columns)
    assert df["source_dataset"].eq("uci_hd_cleveland").all()
    assert df[TARGET_COLUMN].dropna().isin([0, 1]).all()
    assert len(df) == 303


def test_build_unified_dataset_includes_all_sources_and_preserves_rows():
    df = build_unified_dataset()
    assert set(df["source_dataset"].unique()) == {
        "uci_hd_cleveland",
        "uci_hd_hungarian",
        "uci_hd_switzerland",
        "uci_hd_va",
    }
    assert len(df) == 303 + 294 + 123 + 200
    assert df[TARGET_COLUMN].dropna().isin([0, 1]).all()


def test_duplicate_detection_reports_cross_source_candidates():
    raw = pd.DataFrame(
        [
            {
                "age": 63,
                "sex": 1,
                "cp": 1,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0,
                "thal": 6,
                "target": 0,
                "source_collection": "uci_heart_disease_45",
                "source_dataset": "uci_hd_cleveland",
                "source_file": "processed.cleveland.data",
                "source_row_id": 0,
            },
            {
                "age": 63,
                "sex": 1,
                "cp": 1,
                "trestbps": 145,
                "chol": 233,
                "fbs": 1,
                "restecg": 2,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 2.3,
                "slope": 3,
                "ca": 0,
                "thal": 6,
                "target": 0,
                "source_collection": "uci_heart_disease_45",
                "source_dataset": "uci_hd_hungarian",
                "source_file": "processed.hungarian.data",
                "source_row_id": 1,
            },
        ]
    )

    exact_duplicates = detect_exact_duplicates(raw)
    cross_source_duplicates = detect_cross_source_duplicates(raw)

    assert len(exact_duplicates) == 0
    assert len(cross_source_duplicates) == 2


def test_validate_harmonized_dataset_confirms_expected_schema():
    df = harmonize_dataset("uci_hd_cleveland")
    validation = validate_harmonized_dataset(df)

    assert validation["has_expected_columns"] is True
    assert validation["has_binary_target"] is True
    assert validation["has_source_provenance"] is True
