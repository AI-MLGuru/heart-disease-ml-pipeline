from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset_registry import DatasetMetadata, list_dataset_ids, get_dataset_metadata
from .dataset_ingestion import ingest_dataset


def dataset_audit(dataset_id: str) -> dict[str, Any]:
    metadata = get_dataset_metadata(dataset_id)
    df = ingest_dataset(dataset_id)

    # Preserve raw provenance columns from ingestion.
    raw_columns = [c for c in df.columns if c.startswith("source_")]
    data_columns = [c for c in df.columns if c not in raw_columns]

    missing = df[data_columns].apply(lambda col: col.isin(["?", "-9", "", "NA"]).sum())
    unique_values = {col: df[col].dropna().unique().tolist()[:10] for col in data_columns}
    target_column = metadata.target_column
    target_values = Counter(df.iloc[:, -1].dropna().tolist())

    return {
        "dataset_id": dataset_id,
        "source_file": metadata.source_file,
        "site": metadata.site,
        "rows": len(df),
        "columns": len(data_columns),
        "raw_columns": raw_columns,
        "data_columns": data_columns,
        "missing_values": missing.to_dict(),
        "target_distribution": dict(target_values),
        "sample_unique_values": unique_values,
        "schema_preview": [data_columns[:10]],
    }


def audit_all_datasets() -> dict[str, dict[str, Any]]:
    return {dataset_id: dataset_audit(dataset_id) for dataset_id in list_dataset_ids()}
