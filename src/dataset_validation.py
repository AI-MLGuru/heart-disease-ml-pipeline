from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class DatasetValidationError(Exception):
    pass


def validate_raw_file(path: Path, expected_columns: int | None = None) -> dict[str, Any]:
    if not path.exists():
        raise DatasetValidationError(f"Raw file does not exist: {path}")
    if path.stat().st_size == 0:
        raise DatasetValidationError(f"Raw file is empty: {path}")

    if path.suffix.lower() in {".csv", ".data"}:
        try:
            df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False, na_values=[])
        except Exception as exc:
            raise DatasetValidationError(f"Failed to parse file {path}: {exc}") from exc

        if expected_columns is not None and df.shape[1] != expected_columns:
            raise DatasetValidationError(
                f"Expected {expected_columns} columns but found {df.shape[1]} in {path}"
            )

        return {
            "row_count": int(df.shape[0]),
            "column_count": int(df.shape[1]),
            "missing_count": int(df.isna().sum().sum()),
        }

    return {"file_size_bytes": int(path.stat().st_size)}
