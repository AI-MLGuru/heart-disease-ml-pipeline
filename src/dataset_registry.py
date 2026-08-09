from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "heart+disease"


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    collection_id: str
    source: str
    source_file: str
    site: str
    format: str
    target_column: str

    @property
    def source_path(self) -> Path:
        return DATA_DIRECTORY / self.source_file


DATASET_REGISTRY: Dict[str, DatasetMetadata] = {
    "uci_hd_cleveland": DatasetMetadata(
        dataset_id="uci_hd_cleveland",
        collection_id="uci_heart_disease_45",
        source="UCI Machine Learning Repository",
        source_file="processed.cleveland.data",
        site="Cleveland",
        format="csv",
        target_column="num",
    ),
    "uci_hd_hungarian": DatasetMetadata(
        dataset_id="uci_hd_hungarian",
        collection_id="uci_heart_disease_45",
        source="UCI Machine Learning Repository",
        source_file="processed.hungarian.data",
        site="Hungary",
        format="csv",
        target_column="num",
    ),
    "uci_hd_switzerland": DatasetMetadata(
        dataset_id="uci_hd_switzerland",
        collection_id="uci_heart_disease_45",
        source="UCI Machine Learning Repository",
        source_file="processed.switzerland.data",
        site="Switzerland",
        format="csv",
        target_column="num",
    ),
    "uci_hd_va": DatasetMetadata(
        dataset_id="uci_hd_va",
        collection_id="uci_heart_disease_45",
        source="UCI Machine Learning Repository",
        source_file="processed.va.data",
        site="VA Long Beach",
        format="csv",
        target_column="num",
    ),
}


def list_dataset_ids() -> list[str]:
    return list(DATASET_REGISTRY.keys())


def list_datasets() -> list[DatasetMetadata]:
    return list(DATASET_REGISTRY.values())


def get_dataset_metadata(dataset_id: str) -> DatasetMetadata:
    return DATASET_REGISTRY[dataset_id]
