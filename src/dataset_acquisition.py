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


def acquire_dataset_to_raw(
    dataset_metadata: DatasetMetadata,
    destination_root: Path | None = None,
    acquirer: DatasetAcquirer | None = None,
    update_registry: bool = False,
):
    """Acquire a dataset using the provided acquirer (default: LocalFileAcquirer).

    Writes the raw file into `destination_root / dataset_id/` and writes a JSON
    manifest alongside the file. If `update_registry` is True, the function will
    set `access_status='ACQUIRED'` and `processing_status='VALIDATED'` in the
    registry (uses `dataset_registry.update_dataset_metadata`).
    """
    from pathlib import Path

    if destination_root is None:
        destination_root = Path(__file__).resolve().parents[1] / "data" / "raw"

    if acquirer is None:
        acquirer = LocalFileAcquirer()

    dest = Path(destination_root) / dataset_metadata.dataset_id
    manifest = acquirer.acquire(dataset_metadata, dest)

    manifest_path = Path(dest) / "manifest.json"
    manifest.write_json(manifest_path)

    if update_registry:
        try:
            # lazy import to avoid cycles
            from .dataset_registry import update_dataset_metadata

            update_dataset_metadata(
                dataset_metadata.dataset_id,
                access_status="ACQUIRED",
                source_file=str((Path("data") / "raw" / dataset_metadata.dataset_id / manifest.source_file)),
                processing_status="VALIDATED",
            )
        except Exception:
            # don't fail acquisition if registry update isn't possible in this environment
            pass

    return manifest_path


def validate_manifest(manifest_path: Path) -> bool:
    """Validate a saved manifest.json by recomputing sha256 and checking row/col counts.

    Returns True if manifest is valid, False otherwise.
    """
    import json

    from .dataset_manifest import compute_file_sha256

    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return False
    payload = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    file_rel = payload.get("source_file")
    target = base / file_rel
    if not target.exists():
        return False
    sha = compute_file_sha256(target)
    if sha != payload.get("sha256"):
        return False

    # optional checks for csv/delimited
    if target.suffix.lower() in {".csv", ".data"}:
        import pandas as pd

        df = pd.read_csv(target, header=None, dtype=str, keep_default_na=False, na_values=[])
        if df.shape[0] != payload.get("row_count") or df.shape[1] != payload.get("column_count"):
            return False

    return True
