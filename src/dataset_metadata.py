from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetFeature:
    name: str
    canonical_name: str
    dtype: str
    description: str | None = None
    unit: str | None = None
    allowed_range: tuple[float, float] | None = None
    missing_policy: str | None = None


@dataclass(frozen=True)
class DatasetMetadata:
    dataset_id: str
    name: str
    version: str
    source: str
    source_url: str | None
    publisher: str | None
    original_dataset_id: str | None
    accessed_at: str | None
    country: str | None
    region: str | None
    population: str | None
    age_range: str | None
    sex_distribution: str | None
    sample_size: int | None
    clinical_setting: str | None
    disease_definition: str | None
    target_definition: str | None
    collection_method: str | None
    collection_year: int | None
    publication_year: int | None
    license: str | None
    usage_restrictions: str | None
    consent_information: str | None
    privacy_status: str | None
    continent: str | None
    geographic_domain: str | None
    source_file: str
    format: str
    target_column: str
    schema: list[DatasetFeature] | None = None
    notes: str | None = None

    @property
    def source_path(self) -> Path:
        path = Path(self.source_file)
        return path if path.is_absolute() else ROOT_DIR / path
