from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict

from .dataset_metadata import DatasetFeature, DatasetMetadata

DATA_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "heart+disease"


def _default_uci_schema() -> list[DatasetFeature]:
    return [
        DatasetFeature(name=str(i), canonical_name=field, dtype="numeric")
        for i, field in enumerate(
            [
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
                "num",
            ]
        )
    ]


DATASET_REGISTRY: Dict[str, DatasetMetadata] = {
    "uci_hd_cleveland": DatasetMetadata(
        dataset_id="uci_hd_cleveland",
        collection_id="uci_heart_disease_45",
        site="Cleveland Clinic Foundation",
        name="UCI Heart Disease Cleveland",
        version="1.0",
        source="UCI Machine Learning Repository",
        source_url="https://archive.ics.uci.edu/ml/datasets/heart+disease",
        publisher="UCI",
        original_dataset_id="uci_hd_cleveland",
        accessed_at=None,
        country="USA",
        region="North America",
        population="Adults",
        age_range="30-70",
        sex_distribution="male/female",
        sample_size=303,
        clinical_setting="Cardiology",
        disease_definition="Coronary heart disease",
        target_definition="Presence of heart disease (0=no, 1=present)",
        collection_method="Clinical examination",
        collection_year=1988,
        publication_year=1988,
        license="UCI dataset license",
        usage_restrictions="Research use",
        consent_information=None,
        privacy_status="anonymized",
        continent="North America",
        geographic_domain="US clinical sample",
        source_file=str(DATA_DIRECTORY / "processed.cleveland.data"),
        format="csv",
        target_column="num",
        schema=_default_uci_schema(),
        notes="Cleveland site from UCI heart disease dataset.",
    ),
    "uci_hd_hungarian": DatasetMetadata(
        dataset_id="uci_hd_hungarian",
        collection_id="uci_heart_disease_45",
        site="Hungarian Institute of Cardiology",
        name="UCI Heart Disease Hungarian",
        version="1.0",
        source="UCI Machine Learning Repository",
        source_url="https://archive.ics.uci.edu/ml/datasets/heart+disease",
        publisher="UCI",
        original_dataset_id="uci_hd_hungarian",
        accessed_at=None,
        country="Hungary",
        region="Europe",
        population="Adults",
        age_range="30-70",
        sex_distribution="male/female",
        sample_size=294,
        clinical_setting="Cardiology",
        disease_definition="Coronary heart disease",
        target_definition="Presence of heart disease (0=no, 1=present)",
        collection_method="Clinical examination",
        collection_year=1988,
        publication_year=1988,
        license="UCI dataset license",
        usage_restrictions="Research use",
        consent_information=None,
        privacy_status="anonymized",
        continent="Europe",
        geographic_domain="Hungarian clinical sample",
        source_file=str(DATA_DIRECTORY / "processed.hungarian.data"),
        format="csv",
        target_column="num",
        schema=_default_uci_schema(),
        notes="Hungarian site from UCI heart disease dataset.",
    ),
    "uci_hd_switzerland": DatasetMetadata(
        dataset_id="uci_hd_switzerland",
        collection_id="uci_heart_disease_45",
        site="University Hospital, Zurich",
        name="UCI Heart Disease Switzerland",
        version="1.0",
        source="UCI Machine Learning Repository",
        source_url="https://archive.ics.uci.edu/ml/datasets/heart+disease",
        publisher="UCI",
        original_dataset_id="uci_hd_switzerland",
        accessed_at=None,
        country="Switzerland",
        region="Europe",
        population="Adults",
        age_range="30-70",
        sex_distribution="male/female",
        sample_size=123,
        clinical_setting="Cardiology",
        disease_definition="Coronary heart disease",
        target_definition="Presence of heart disease (0=no, 1=present)",
        collection_method="Clinical examination",
        collection_year=1988,
        publication_year=1988,
        license="UCI dataset license",
        usage_restrictions="Research use",
        consent_information=None,
        privacy_status="anonymized",
        continent="Europe",
        geographic_domain="Swiss clinical sample",
        source_file=str(DATA_DIRECTORY / "processed.switzerland.data"),
        format="csv",
        target_column="num",
        schema=_default_uci_schema(),
        notes="Switzerland site from UCI heart disease dataset.",
    ),
    "uci_hd_va": DatasetMetadata(
        dataset_id="uci_hd_va",
        collection_id="uci_heart_disease_45",
        site="V.A. Medical Center, Long Beach",
        name="UCI Heart Disease VA",
        version="1.0",
        source="UCI Machine Learning Repository",
        source_url="https://archive.ics.uci.edu/ml/datasets/heart+disease",
        publisher="UCI",
        original_dataset_id="uci_hd_va",
        accessed_at=None,
        country="USA",
        region="North America",
        population="Veterans",
        age_range="30-70",
        sex_distribution="male",
        sample_size=200,
        clinical_setting="Cardiology",
        disease_definition="Coronary heart disease",
        target_definition="Presence of heart disease (0=no, 1=present)",
        collection_method="Clinical examination",
        collection_year=1988,
        publication_year=1988,
        license="UCI dataset license",
        usage_restrictions="Research use",
        consent_information=None,
        privacy_status="anonymized",
        continent="North America",
        geographic_domain="US veteran clinical sample",
        source_file=str(DATA_DIRECTORY / "processed.va.data"),
        format="csv",
        target_column="num",
        schema=_default_uci_schema(),
        notes="Long Beach VA site from UCI heart disease dataset.",
    ),
    "ng_kano_cad_506": DatasetMetadata(
        dataset_id="ng_kano_cad_506",
        collection_id="kano_cad_study",
        name="Kano CAD Study (reported)",
        version="0.1",
        source="Published study - Kano CAD",
        source_url=None,
        publisher=None,
        original_dataset_id=None,
        accessed_at=None,
        site="Kano Cardiology Research Group",
        country="Nigeria",
        region="North-West Nigeria",
        population="Adults",
        age_range=None,
        sex_distribution=None,
        sample_size=506,
        clinical_setting="Cardiology",
        disease_definition="Coronary artery disease",
        target_definition="CAD diagnosis reported in publication",
        collection_method="Reported in publication",
        collection_year=None,
        publication_year=None,
        license=None,
        usage_restrictions="access_pending",
        consent_information=None,
        privacy_status=None,
        continent="Africa",
        geographic_domain="Kano State, Nigeria",
        source_file="",
        format="unknown",
        target_column="num",
        schema=None,
        notes="Candidate Nigerian dataset; reported records=506; access pending (author request).",
        access_status="ACCESS_PENDING",
    ),
}


def list_dataset_ids() -> list[str]:
    return list(DATASET_REGISTRY.keys())


def list_datasets() -> list[DatasetMetadata]:
    return list(DATASET_REGISTRY.values())


def get_dataset_metadata(dataset_id: str) -> DatasetMetadata:
    return DATASET_REGISTRY[dataset_id]


def export_registry_json(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {dataset_id: asdict(metadata) for dataset_id, metadata in DATASET_REGISTRY.items()},
            indent=2,
        )
    )
    return path
