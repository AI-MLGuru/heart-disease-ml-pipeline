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
    # Identity / required provenance
    dataset_id: str
    collection_id: str | None
    name: str
    version: str
    source: str
    source_file: str
    format: str
    target_column: str

    # Optional descriptive / source fields
    source_url: str | None = None
    publisher: str | None = None
    original_dataset_id: str | None = None
    accessed_at: str | None = None
    site: str | None = None

    # Geography
    country: str | None = None
    region: str | None = None
    continent: str | None = None
    geographic_domain: str | None = None

    # Demographics / dataset details
    population: str | None = None
    age_range: str | None = None
    sex_distribution: str | None = None
    sample_size: int | None = None
    clinical_setting: str | None = None

    # Definitions and timing
    disease_definition: str | None = None
    target_definition: str | None = None
    collection_method: str | None = None
    collection_year: int | None = None
    publication_year: int | None = None

    # Policy / miscellaneous
    license: str | None = None
    usage_restrictions: str | None = None
    consent_information: str | None = None
    privacy_status: str | None = None
    access_status: str | None = None
    # processing_status tracks where the dataset is in the pipeline lifecycle.
    # Allowed values: NOT_STARTED, VALIDATED, AUDITED, HARMONIZED, READY
    processing_status: str = "NOT_STARTED"

    schema: list[DatasetFeature] | None = None
    notes: str | None = None

    @property
    def source_path(self) -> Path:
        path = Path(self.source_file)
        return path if path.is_absolute() else ROOT_DIR / path
