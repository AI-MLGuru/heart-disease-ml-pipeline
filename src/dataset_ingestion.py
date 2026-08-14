from __future__ import annotations

from pathlib import Path

import pandas as pd

from .dataset_registry import get_dataset_metadata, list_dataset_ids_for_processing


def _read_csv_preserve_raw(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path, header=None, dtype=str, keep_default_na=False, na_values=[]
    )


def ingest_dataset(dataset_id: str) -> pd.DataFrame:
    metadata = get_dataset_metadata(dataset_id)
    df = _read_csv_preserve_raw(metadata.source_path)

    provenance = {
        "source_collection": metadata.collection_id,
        "source_dataset": metadata.dataset_id,
        "source_file": metadata.source_file,
    }

    for key, value in provenance.items():
        df[key] = value

    df["source_row_id"] = df.reset_index().index.astype(int)
    return df


def ingest_all_datasets() -> pd.DataFrame:
    frames = [
        ingest_dataset(dataset_id)
        for dataset_id in list_dataset_ids_for_processing()
    ]
    return pd.concat(frames, ignore_index=True)
