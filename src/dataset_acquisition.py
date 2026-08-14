from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol

from .dataset_manifest import DatasetManifest, compute_file_sha256
from .dataset_registry import DatasetMetadata


class DatasetAcquirer(Protocol):
    def acquire(
        self, metadata: DatasetMetadata, destination: Path
    ) -> DatasetManifest: ...


class LocalFileAcquirer:
    def acquire(self, metadata: DatasetMetadata, destination: Path) -> DatasetManifest:
        source_path = metadata.source_path
        destination.mkdir(parents=True, exist_ok=True)
        target_path = destination / source_path.name
        if not target_path.exists():
            target_path.write_bytes(source_path.read_bytes())

        sha256 = compute_file_sha256(target_path)
        row_count = 0
        column_count = 0
        if target_path.suffix.lower() in {".csv", ".data"}:
            import pandas as pd

            df = pd.read_csv(
                target_path, header=None, dtype=str, keep_default_na=False, na_values=[]
            )
            row_count, column_count = df.shape

        manifest = DatasetManifest(
            dataset_id=metadata.dataset_id,
            version="1.0",
            source_file=str(target_path.relative_to(destination)),
            acquired_at=datetime.utcnow().isoformat() + "Z",
            sha256=sha256,
            row_count=row_count,
            column_count=column_count,
            file_size_bytes=target_path.stat().st_size,
            format=metadata.format,
            description=None,
            additional_metadata={
                "source": metadata.source,
                "site": getattr(metadata, "site", None),
            },
        )
        return manifest
