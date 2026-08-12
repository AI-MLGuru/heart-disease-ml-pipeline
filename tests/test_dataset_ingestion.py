from pathlib import Path

import pandas as pd

from src.dataset_ingestion import ingest_dataset
from src.dataset_registry import get_dataset_metadata


def test_ingest_dataset_attaches_provenance_columns():
    metadata = get_dataset_metadata("uci_hd_cleveland")
    df = ingest_dataset(metadata.dataset_id)

    assert "source_collection" in df.columns
    assert "source_dataset" in df.columns
    assert "source_file" in df.columns
    assert "source_row_id" in df.columns
    assert df["source_file"].eq(metadata.source_file).all()
    assert df["source_dataset"].eq(metadata.dataset_id).all()
    assert df["source_collection"].eq(metadata.collection_id).all()


def test_ingest_dataset_preserves_raw_data_types():
    df = ingest_dataset("uci_hd_cleveland")
    assert df.shape[0] == 303
    assert df.shape[1] >= 5
    assert df.iloc[0, 0] != ""
    assert df.iloc[0, 0] is not None
