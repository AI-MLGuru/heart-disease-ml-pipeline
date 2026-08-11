from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .dataset_metadata import DatasetMetadata


def serialize_metadata(metadata: DatasetMetadata) -> dict[str, Any]:
    result = asdict(metadata)
    if metadata.schema is not None:
        result["schema"] = [asdict(feature) for feature in metadata.schema]
    return result


def write_metadata_json(metadata: DatasetMetadata, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(serialize_metadata(metadata), indent=2))
    return path
