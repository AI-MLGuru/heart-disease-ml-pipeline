from __future__ import annotations

from typing import Any

import pandas as pd

from .dataset_ingestion import ingest_dataset

CANONICAL_FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]
TARGET_COLUMN = "target"
PROVENANCE_COLUMNS = [
    "source_collection",
    "source_dataset",
    "source_file",
    "source_row_id",
]
RAW_MISSING_VALUES = {"?", "-9", "", "NA"}
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]

HARMONIZED_COLUMNS = CANONICAL_FEATURES + [TARGET_COLUMN]
EXPECTED_COLUMNS = HARMONIZED_COLUMNS + PROVENANCE_COLUMNS


def _canonical_column_mapping() -> dict[int, str]:
    return {index: column for index, column in enumerate(HARMONIZED_COLUMNS)}


def harmonize_target(raw_target: pd.Series) -> pd.Series:
    def _map_value(value: Any) -> Any:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, str) and value.strip() in RAW_MISSING_VALUES:
            return pd.NA
        try:
            number = int(str(value).strip())
        except (ValueError, TypeError):
            return pd.NA

        return 0 if number == 0 else 1

    return raw_target.map(_map_value).astype("Int64")


def normalize_schema(df: pd.DataFrame) -> pd.DataFrame:
    mapping = _canonical_column_mapping()
    result = df.rename(columns=mapping)
    if TARGET_COLUMN not in result.columns:
        raise ValueError("Unable to normalize schema: missing target column")

    ordered_columns = HARMONIZED_COLUMNS + [
        c for c in PROVENANCE_COLUMNS if c in result.columns
    ]
    return result.reindex(columns=ordered_columns)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    for column in HARMONIZED_COLUMNS:
        if column not in result.columns:
            continue
        result[column] = result[column].replace(list(RAW_MISSING_VALUES), pd.NA)

    for column in NUMERIC_FEATURES:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    if TARGET_COLUMN in result.columns:
        result[TARGET_COLUMN] = (
            result[TARGET_COLUMN]
            .astype("string")
            .replace(list(RAW_MISSING_VALUES), pd.NA)
        )

    return result


def harmonize_dataset(dataset_id: str) -> pd.DataFrame:
    raw_df = ingest_dataset(dataset_id)
    normalized = normalize_schema(raw_df)
    cleaned = handle_missing_values(normalized)
    cleaned[TARGET_COLUMN] = harmonize_target(cleaned[TARGET_COLUMN])
    return cleaned


def build_unified_dataset() -> pd.DataFrame:
    # Only include datasets that are acquired/available for ingestion
    from .dataset_registry import list_dataset_ids_by_status

    frames = [
        harmonize_dataset(dataset_id)
        for dataset_id in list_dataset_ids_by_status("ACQUIRED")
    ]
    return pd.concat(frames, ignore_index=True)


def report_missingness(df: pd.DataFrame) -> pd.Series:
    return df[HARMONIZED_COLUMNS].isna().sum()


def detect_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    subset = HARMONIZED_COLUMNS + PROVENANCE_COLUMNS
    return df[df.duplicated(subset=subset, keep=False)]


def detect_cross_source_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    subset = HARMONIZED_COLUMNS
    duplicates = df[df.duplicated(subset=subset, keep=False)].copy()
    return duplicates.groupby(subset).filter(
        lambda group: group["source_dataset"].nunique() > 1
    )


def validate_harmonized_dataset(df: pd.DataFrame) -> dict[str, bool]:
    has_expected_columns = list(df.columns) == EXPECTED_COLUMNS
    if TARGET_COLUMN in df.columns:
        target_values = df[TARGET_COLUMN].dropna().unique().tolist()
        target_values = [int(v) for v in target_values if v is not pd.NA]
        has_binary_target = set(target_values).issubset({0, 1})
    else:
        has_binary_target = False

    return {
        "has_expected_columns": has_expected_columns,
        "has_binary_target": has_binary_target,
        "has_source_provenance": all(col in df.columns for col in PROVENANCE_COLUMNS),
    }
