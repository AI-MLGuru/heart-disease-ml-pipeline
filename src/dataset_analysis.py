from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .dataset_harmonization import (
    CANONICAL_FEATURES,
    TARGET_COLUMN,
    PROVENANCE_COLUMNS,
    build_unified_dataset,
    detect_cross_source_duplicates,
    detect_exact_duplicates,
)

NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
BINARY_FEATURES = ["sex", "fbs", "exang"]
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "ca", "thal"]

AUDIT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "reports"


def load_unified_dataset() -> pd.DataFrame:
    return build_unified_dataset()


def composition_summary(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "total_rows": int(len(df)),
        "total_features": len(CANONICAL_FEATURES),
        "source_counts": df["source_dataset"].value_counts().to_dict(),
    }


def target_distribution(df: pd.DataFrame) -> dict[str, Any]:
    counts = df[TARGET_COLUMN].value_counts(dropna=False).to_dict()
    total = len(df)
    positive = int(counts.get(1, 0))
    negative = int(counts.get(0, 0))
    return {
        "counts": {str(k): int(v) for k, v in counts.items()},
        "prevalence": positive / total if total else 0.0,
        "total": total,
        "positive": positive,
        "negative": negative,
    }


def missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = CANONICAL_FEATURES + [TARGET_COLUMN]
    missing = df[columns].isna().sum()
    percent = (missing / len(df) * 100).round(2)
    report = pd.DataFrame(
        {
            "missing_count": missing.astype(int),
            "missing_percent": percent,
        }
    )
    return report


def source_missingness_summary(df: pd.DataFrame) -> pd.DataFrame:
    columns = CANONICAL_FEATURES + [TARGET_COLUMN]
    groups = df.groupby("source_dataset")
    report = []
    for source, group in groups:
        missing = group[columns].isna().sum()
        percent = (missing / len(group) * 100).round(2)
        row = {"source_dataset": source}
        for column in columns:
            row[f"{column}_missing_count"] = int(missing[column])
            row[f"{column}_missing_percent"] = float(percent[column])
        report.append(row)
    return pd.DataFrame(report).sort_values("source_dataset").reset_index(drop=True)


def duplicate_summary(df: pd.DataFrame) -> dict[str, Any]:
    exact = detect_exact_duplicates(df)
    cross_source = detect_cross_source_duplicates(df)
    return {
        "exact_duplicate_rows": int(len(exact)),
        "exact_duplicate_groups": int(exact.duplicated(keep=False).sum() // 2),
        "cross_source_duplicate_rows": int(len(cross_source)),
        "cross_source_duplicate_groups": int(cross_source.duplicated(keep=False).sum() // 2),
    }


def numeric_feature_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for column in NUMERICAL_FEATURES:
        if column not in df.columns:
            continue
        series = pd.to_numeric(df[column], errors="coerce")
        summary[column] = {
            "missing_count": int(series.isna().sum()),
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std(ddof=0)),
            "min": float(series.min()),
            "max": float(series.max()),
            "unique_values": int(series.nunique(dropna=True)),
        }
    return summary


def categorical_feature_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for column in CATEGORICAL_FEATURES + BINARY_FEATURES:
        if column not in df.columns:
            continue
        values = df[column].astype("string")
        freq = values.value_counts(dropna=False).to_dict()
        summary[column] = {
            "missing_count": int(values.isna().sum()),
            "unique_values": int(values.nunique(dropna=True)),
            "top_values": {str(k): int(v) for k, v in list(freq.items())[:5]},
        }
    return summary


def source_shift_summary(df: pd.DataFrame) -> dict[str, Any]:
    numeric = {}
    categorical = {}
    for source, group in df.groupby("source_dataset"):
        numeric[source] = {
            column: {
                "mean": float(pd.to_numeric(group[column], errors="coerce").mean()),
                "median": float(pd.to_numeric(group[column], errors="coerce").median()),
                "missing_count": int(group[column].isna().sum()),
            }
            for column in NUMERICAL_FEATURES
            if column in group.columns
        }
        categorical[source] = {
            column: group[column].astype("string").value_counts(normalize=True, dropna=False).round(4).to_dict()
            for column in CATEGORICAL_FEATURES + BINARY_FEATURES
            if column in group.columns
        }
    return {"numerical": numeric, "categorical": categorical}


def generate_dataset_report(df: pd.DataFrame, output_dir: Path | str | None = None) -> dict[str, Any]:
    if output_dir is None:
        output_dir = AUDIT_OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "composition": composition_summary(df),
        "target_distribution": target_distribution(df),
        "missingness": missingness_summary(df).to_dict(orient="index"),
        "source_missingness": source_missingness_summary(df).to_dict(orient="records"),
        "duplicate_summary": duplicate_summary(df),
        "numeric_feature_summary": numeric_feature_summary(df),
        "categorical_feature_summary": categorical_feature_summary(df),
        "source_shift_summary": source_shift_summary(df),
    }

    json_path = output_dir / "dataset_audit.json"
    md_path = output_dir / "dataset_audit.md"

    json_path.write_text(json.dumps(report, indent=2))
    md_lines = [
        "# Dataset Audit Report",
        "",
        "## Composition Summary",
        "",
        f"Total rows: {report['composition']['total_rows']}",
        f"Total features: {report['composition']['total_features']}",
        "",
        "### Source counts",
        "",
    ]
    for source, count in report["composition"]["source_counts"].items():
        md_lines.append(f"- {source}: {count}")
    md_lines.extend([
        "",
        "## Target Distribution",
        "",
        f"Positive: {report['target_distribution']['positive']}",
        f"Negative: {report['target_distribution']['negative']}",
        f"Prevalence: {report['target_distribution']['prevalence']:.4f}",
        "",
        "## Duplicate Summary",
        "",
        f"Exact duplicate rows: {report['duplicate_summary']['exact_duplicate_rows']}",
        f"Cross-source duplicate rows: {report['duplicate_summary']['cross_source_duplicate_rows']}",
        "",
        "## Missingness Summary",
        "",
    ])
    missing_df = missingness_summary(df)
    md_lines.append(missing_df.to_markdown())
    md_lines.extend([
        "",
        "## Source Missingness Summary",
        "",
    ])
    source_missing_df = source_missingness_summary(df)
    md_lines.append(source_missing_df.to_markdown(index=False))
    md_lines.extend([
        "",
        "## Numeric Feature Summary",
        "",
    ])
    for feature, stats in report["numeric_feature_summary"].items():
        md_lines.append(f"### {feature}")
        for metric, value in stats.items():
            md_lines.append(f"- {metric}: {value}")
        md_lines.append("")
    md_lines.extend([
        "## Categorical Feature Summary",
        "",
    ])
    for feature, stats in report["categorical_feature_summary"].items():
        md_lines.append(f"### {feature}")
        md_lines.append(f"- missing_count: {stats['missing_count']}")
        md_lines.append(f"- unique_values: {stats['unique_values']}")
        md_lines.append("- top_values:")
        for value, count in stats["top_values"].items():
            md_lines.append(f"  - {value}: {count}")
        md_lines.append("")

    md_path.write_text("\n".join(md_lines))
    report["report_files"] = {
        "json": str(json_path),
        "markdown": str(md_path),
    }
    return report
