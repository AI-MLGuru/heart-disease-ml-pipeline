# Heart Disease Predictor
A modular, reproducible machine-learning research pipeline for **heart disease prediction, real-world clinical dataset evaluation, dataset harmonization, and cross-dataset generalization analysis**.

The project began as a traditional heart-disease classification model and is evolving into a research-oriented data and ML infrastructure designed to answer a more important question:

> **Can a machine-learning model trained on one population and dataset distribution continue to perform reliably when exposed to real-world data from different institutions, populations, regions, and collection protocols?**

Rather than optimizing solely for model accuracy on a single benchmark dataset, this project prioritizes:

- Comparing multiple machine-learning models under identical evaluation conditions
- Ingesting heterogeneous real-world heart-disease datasets
- Preserving dataset provenance and source lineage
- Auditing dataset quality before harmonization
- Detecting exact and cross-source duplicate candidates
- Harmonizing heterogeneous datasets into a canonical representation
- Preserving source-specific information after harmonization
- Building a validated unified real-world dataset
- Evaluating model generalization across independent datasets
- Investigating dataset shift and population differences
- Maintaining reproducible, testable ML infrastructure
- Progressively developing a production-ready ML/MLOps architecture

> **Important:** This project does **not** use synthetic data. Dataset expansion is based exclusively on real-world datasets obtained from documented sources.

---

# Project Status

## Stage 1 — Core ML Pipeline ✅

The original machine-learning pipeline is implemented and tested.

- Modular project structure
- Data loading
- Data preprocessing
- Target handling
- Model training
- Model persistence
- Model loading
- Prediction interface
- Automated regression testing

---

## Stage 2 — Cross-Validation ✅

The pipeline uses stratified cross-validation to provide consistent and reproducible model evaluation.

Implemented:

- Stratified cross-validation
- Consistent validation folds
- Cross-validation scoring
- Mean performance
- Standard deviation
- Per-fold class distribution inspection

Current evaluation uses:

```
StratifiedKFold
```

Stratification helps maintain class proportions across validation folds and reduces the risk of producing misleading fold-level evaluations.

---

## Stage 3 — Evaluation Metrics ✅

The evaluation layer currently supports:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report

The objective is to evaluate models using multiple complementary metrics rather than relying on accuracy alone.

---

## Stage 4 — Model Comparison 🚧

Models are being evaluated under the **same cross-validation strategy and evaluation metrics**.

Current models:

- Logistic Regression
- Random Forest
- Gradient Boosting

The goal is to make model comparison fair and reproducible.

Rather than optimizing one model in isolation, every candidate model should be evaluated under equivalent conditions.

Planned comparison output:

```
Model              Accuracy    Precision    Recall    F1    ROC-AUC
--------------------------------------------------------------------
Logistic Regression
Random Forest
Gradient Boosting
```

---

# Dataset Infrastructure

The project has moved beyond a single CSV-based workflow and now contains dedicated infrastructure for working with multiple real-world heart-disease datasets.

The dataset architecture is designed around the following principle:

```
Source
   ↓
Registry
   ↓
Acquisition
   ↓
Raw Dataset
   ↓
Validation
   ↓
Quality Audit
   ↓
Harmonization
   ↓
Processed Dataset
   ↓
Analysis / ML
```

Each stage has a distinct responsibility.

This prevents data acquisition, cleaning, harmonization, analysis, and model training from becoming one opaque operation.

---

# UCI Heart Disease Collection

The UCI Heart Disease collection is currently the primary benchmark and development source.

It contains data from four clinical sites:

| Site | Institution | Records |
| --- | --- | --- |
| Cleveland | Cleveland Clinic Foundation | 303 |
| Hungarian | Hungarian Institute of Cardiology | 294 |
| Switzerland | University Hospital, Zurich | 123 |
| VA | V.A. Medical Center, Long Beach | 200 |

The processed UCI datasets use the standard heart-disease schema.

Original UCI files are preserved under:

```
data/heart+disease/
```

Including:

```
processed.cleveland.data
processed.hungarian.data
processed.switzerland.data
processed.va.data

cleveland.data
hungarian.data
switzerland.data
long-beach-va.data
```

The source datasets contain missing values represented by:

```
?
```

They also have different target distributions.

For example:

```
Cleveland    → 0–4
Hungarian    → 0–1
Switzerland  → 0–4
VA           → 0–4
```

These differences are **not silently harmonized during ingestion**.

Target transformation and clinical harmonization are explicit downstream operations.

---

# Important UCI Research Boundary

The four UCI sites are treated as **one collection with four source domains**, rather than four completely independent external datasets.

This distinction is important for research validity.

The intended progression is:

```
UCI Collection
      ↓
Harmonize Four Sites
      ↓
Unified UCI Baseline
      ↓
Add Independent Real-World Datasets
      ↓
External Validation
```

The four sites can still be analyzed separately for source-level differences, missingness, target distributions, and dataset shift.

However, they should not automatically be presented as four independent external validation datasets.

This distinction helps prevent overstating model generalization.

---

# Dataset Provenance

Every ingested observation receives provenance metadata.

Current provenance fields include:

```
source_collection
source_dataset
source_file
source_row_id
```

Example:

```
source_collection = UCI Heart Disease
source_dataset    = uci_hd_cleveland
source_file       = processed.cleveland.data
source_row_id     = 42
```

This allows an individual observation to be traced back to its original dataset and source file.

Provenance is preserved through the harmonization process.

This is essential because combining datasets without retaining source information makes it difficult to investigate:

- Dataset shift
- Population differences
- Source-specific missingness
- Duplicate observations
- Collection bias
- Model performance differences
- Data quality issues
- Reproducibility

---

# Dataset Registry

The dataset registry provides a central representation of the datasets available to the project.

Current module:

```
src/dataset_registry.py
```

Current responsibilities include:

- Dataset identification
- Dataset metadata
- Source information
- File locations
- Schema information
- Target metadata
- Dataset discovery

Current registered datasets:

```
uci_hd_cleveland
uci_hd_hungarian
uci_hd_switzerland
uci_hd_va
```

Current interfaces include:

```
list_dataset_ids()
list_datasets()
get_dataset_metadata()
```

---

# Global Dataset Acquisition & Registry

## Next Architecture Layer 🚧

The next major development stage is the **Global Dataset Acquisition & Registry layer**.

The objective is to evolve the project from a collection of locally managed datasets into a structured, provenance-aware ecosystem capable of onboarding real-world datasets from:

```
Nigeria
Africa
Global Sources
```

UCI will remain an important benchmark source, but it will no longer be the architectural limit of the system.

The registry will eventually capture richer metadata including:

```
dataset_id
name
version
description

source
source_url
publisher
original_dataset_id
acquisition_timestamp
checksum

country
region
continent
geographic_domain
population
age_range
sex_distribution
sample_size

collection_year
publication_year
clinical_setting

feature_schema
target_definition
feature_types
units
missing_value_conventions

data_quality
missingness
duplicate_rate
invalid_value_rate
target_distribution

license
usage_restrictions
privacy_information
```

The registry will become the **source of truth for every dataset entering the ML pipeline**.

---

# Dataset Acquisition Architecture

Dataset acquisition will be separated from dataset registration.

The intended architecture is:

```
Dataset Registry
       │
       │ Dataset metadata
       ▼
Dataset Acquisition
       │
       ▼
data/raw/
       │
       ▼
Dataset Manifest
       │
       ▼
Validation
       │
       ▼
Quality Audit
       │
       ▼
Harmonization
       │
       ▼
data/processed/
```

This separation makes acquisition reproducible and prevents download logic from becoming coupled to preprocessing or model training.

Future acquisition adapters may support different documented dataset sources.

---

# Raw Data Preservation

Raw datasets are treated as **immutable source data**.

The project should never modify the original dataset in place.

Conceptually:

```
data/raw/
```

contains the source representation exactly as acquired.

Transformations occur downstream.

For example:

```
Raw Dataset
    ↓
Validation
    ↓
Transformation
    ↓
Harmonized Dataset
```

If a transformation is incorrect, the processed representation can be regenerated without corrupting the original source.

---

# Dataset Manifests

Each acquired dataset will eventually receive a manifest describing exactly what was obtained.

A manifest is intended to capture information such as:

```
dataset_id
version
source
source_url
acquired_at
file_path
file_format
sha256
row_count
column_count
```

The SHA-256 checksum provides an immutable fingerprint that can help determine whether the acquired file has changed.

This supports reproducibility and future dataset version tracking.

---

# Dataset Ingestion

Module:

```
src/dataset_ingestion.py
```

The ingestion layer is responsible for loading source datasets while preserving their original representation.

Current responsibilities:

- Load raw/processed UCI files
- Preserve original values
- Avoid premature target transformation
- Attach provenance metadata
- Produce a consistent DataFrame interface
- Maintain source identity

Example provenance columns:

```
source_collection
source_dataset
source_file
source_row_id
```

The ingestion layer intentionally does **not** perform clinical harmonization.

---

# Dataset Validation

Dataset validation is positioned between acquisition and harmonization.

The intended validation flow is:

```
Acquisition
    ↓
Structural Validation
    ↓
Schema Validation
    ↓
Quality Validation
    ↓
Duplicate Validation
    ↓
Harmonization
```

Validation will progressively cover:

### Structural Validation

- File exists
- File isn't empty
- File can be parsed
- Expected format
- Row count > 0
- Columns exist

### Schema Validation

- Required target exists
- Feature names are recognized or mappable
- Data types are valid
- Units are known where applicable

### Quality Validation

- Missingness
- Invalid values
- Outliers
- Duplicate rows
- Target distribution

### Provenance Validation

- Source identifier
- Source file
- Source row identifier
- Acquisition metadata

A dataset should not enter the harmonization stage until its basic structural and provenance requirements are satisfied.

---

# Dataset Quality

Instead of simply saying:

> "Dataset looks good."

We should eventually generate something like:

```
Dataset Quality Report
──────────────────────────
Rows:              12,430
Features:          27

Missingness:        3.2%
Duplicate rows:     0.4%
Invalid values:     0.1%

Target distribution:
HIGH:              12.7%
LOW:               87.3%

Schema validity:    PASS
Provenance:        PASS
License:           VERIFIED

Quality score:      91/100
```

This fits perfectly with the reporting infrastructure already built.

---

# Geographic Metadata

This is where the "global" part becomes powerful.

We're not simply collecting:

> Dataset A, Dataset B, Dataset C.

We're building something where we can ask:

> **Where does this data actually represent?**

For example:

```
Dataset A
Country: Nigeria
Region: West Africa
Population: Nigerian adults

Dataset B
Country: United States
Region: North America
Population: US adults

Dataset C
Country: United Kingdom
Region: Europe
Population: UK adults
```

Then later our ML system can investigate:

**Does a model trained primarily on Western populations generalize to Nigerian populations?**

That's a much more interesting research question.

---

# Nigeria/Africa as a Deliberate Acquisition Track

The registry should be split into:

### Benchmark

UCI and other established datasets.

Purpose:

- Regression testing
- Baseline models
- Algorithm comparison
- Pipeline validation

### African datasets

Nigeria first where possible, then:

- Ghana
- Kenya
- South Africa
- etc.

Purpose:

- Geographic representation
- Population-specific analysis
- Generalization

### Global datasets

Other regions.

Purpose:

- Diversity
- External validation
- Distribution-shift analysis

That gives the project a much stronger scientific foundation.

---

# Dataset Integration Philosophy

We shouldn't immediately dump every dataset into one training set.

Initially, datasets should remain distinct.

For example:

```
UCI
 ↓
harmonized UCI

Nigeria
 ↓
harmonized Nigeria

UK
 ↓
harmonized UK
```

Only later should we decide whether to:

```
                    ┌── UCI
                    │
Harmonization ──────┼── Nigeria
                    │
                    └── UK
                         ↓
                 Global dataset
```

Why?

Because combining datasets too early can hide **dataset shift** and population differences.

We want to be able to say:

> "This model performs 94% on Dataset A but only 71% on Dataset B."

That information can be scientifically and clinically important.

---

# Versioning

Eventually:

```
Dataset
    v1.0
    v1.1
    v2.0

Schema
    v1
    v2

Harmonization
    v1
    v2
```

This means if our model suddenly changes performance, we can determine exactly which data or transformation changed.

---

# Testing Strategy

The new layer should be added incrementally without breaking the current suite.

Tests should be structured as:

```
tests/
├── test_dataset_registry.py
├── test_dataset_metadata.py
├── test_dataset_acquisition.py
├── test_dataset_manifest.py
├── test_dataset_validation.py
├── test_dataset_harmonization.py
└── test_dataset_analysis.py
```

The existing `25/25` and `14/14` test counts must remain green.

---

# Implementation Order

### Phase 1 — Architecture

Create:

```
data/raw/
data/processed/
data/interim/
data/metadata/
```

Don't download anything yet.

---

### Phase 2 — Metadata model

Implement:

```
DatasetMetadata
DatasetFeature
DatasetQuality
DatasetProvenance
```

with validation.

---

### Phase 3 — Registry

Upgrade:

```
dataset_registry.py
```

to support:

```
register()
get()
list()
exists()
update()
```

with unique dataset IDs.

---

### Phase 4 — Manifest

Implement:

```
DatasetManifest
```

with:

- checksum
- acquisition timestamp
- file metadata
- row/column counts
- source information

---

### Phase 5 — Acquisition

Implement the first acquisition adapter.

Start with **UCI**, because it's already the benchmark.

That gives us a known-good integration test.

---

### Phase 6 — Validation

Connect the acquisition output to:

```
validation → harmonization
```

and reuse existing duplicate detection and reporting logic.

---

### Phase 7 — Registry expansion

Only after the architecture is stable do we start onboarding:

```
Nigeria → Africa → Global
```

Each dataset gets:

```
metadata
manifest
quality report
schema
provenance
```

---

### Phase 8 — Global dataset analysis

Then we can start doing genuinely interesting things:

```
Population comparison
        ↓
Feature availability
        ↓
Distribution analysis
        ↓
Missingness comparison
        ↓
Target prevalence
        ↓
Cross-dataset validation
        ↓
Model generalization
```

That's where this stops being merely a data pipeline and becomes a serious **research/ML infrastructure layer**.

---

# The Architecture

```
                  ┌──────────────────────┐
                  │  Dataset Registry    │
                  │                      │
                  │ Metadata             │
                  │ Provenance           │
                  │ Schema               │
                  │ License              │
                  │ Geography            │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Dataset Acquisition  │
                  └──────────┬───────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ data/raw/      │
                    │ IMMUTABLE      │
                    └───────┬────────┘
                            │
                            ▼
                  ┌──────────────────────┐
                  │ Data Validation      │
                  │                      │
                  │ Structure            │
                  │ Schema               │
                  │ Quality              │
                  │ Duplicates           │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Harmonization        │
                  │                      │
                  │ Names                │
                  │ Types                │
                  │ Units                │
                  │ Targets              │
                  └──────────┬───────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ data/processed │
                    └───────┬────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
        Dataset Analysis          ML Pipeline
                │                       │
                └───────────┬───────────┘
                            ▼
                  Reports / Models /
                  Generalization
```

## The key principle

**Don't build the registry as a glorified list of CSV files.**

Build it as the **source of truth for every dataset entering the project**.

That means six months from now, when we have 20–50 datasets, we can answer:

> Where did this data come from?

> Who does it represent?

> When was it collected?

> What license does it have?

> What features does it contain?

> How good is it?

> Has it changed?

> How was it transformed?

> Which records were removed?

> How was the target transformed?

> How was the dataset harmonized?

> Which model used the resulting data?

> How was the model evaluated?

If we build this layer properly now, the eventual ML system will be dramatically more trustworthy and reproducible.

**So I would not start by acquiring the Nigerian datasets yet.** First, we build the registry + metadata contracts + manifests + acquisition interface + tests. Then UCI becomes our first registered/acquired dataset under the new architecture. Once that works end-to-end, we start onboarding Nigeria/Africa/global sources one at a time.

- country/region or clinical source metadata
- target encoding rules
- feature semantics

The dataset-analysis/reporting layer is explicitly built to identify:

- dataset composition differences
- feature missingness patterns
- source shift and population differences
- exact duplicates and cross-source duplicate candidates

This makes the project suitable for evolving from a UCI benchmark into a multi-country dataset ecosystem.

---

# Testing
The project uses `pytest` for automated regression testing.

Current test coverage includes:

### Dataset Infrastructure

- Dataset registry
- Dataset ingestion
- Provenance tracking
- Raw data preservation
- Dataset quality auditing
- Multi-dataset registry validation

### ML Pipeline

- Data loading
- Target encoding
- Pipeline construction
- Model fitting
- Prediction
- Cross-validation
- Model comparison
- Evaluation reporting

Current test status:

```
11 passed
8 warnings
```

The latest full test run completed successfully.
The warnings are currently related to a NumPy/joblib deprecation warning and do not cause test failures.

Run the full suite with:

```
pytest -v -rA
```

---

# Project Structure

```
Heart Disease Predictor/
│
├── data/
│   ├── heart.csv
│   │
│   └── heart+disease/
│       ├── cleveland.data
│       ├── hungarian.data
│       ├── switzerland.data
│       ├── long-beach-va.data
│       ├── processed.cleveland.data
│       ├── processed.hungarian.data
│       ├── processed.switzerland.data
│       └── processed.va.data
│
├── models/
│   └── logistic_regression.joblib
│
├── notebooks/
│   └── playground.ipynb
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── evaluate.py
│   ├── model_comparison.py
│   ├── dataset_registry.py
│   ├── dataset_ingestion.py
│   └── dataset_quality.py
│
├── tests/
│   ├── test_pipeline.py
│   ├── test_model_comparison.py
│   ├── test_dataset_registry.py
│   ├── test_dataset_ingestion.py
│   └── test_dataset_quality.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# Roadmap

## Current

- Modular ML project
- Automated tests
- Train/predict pipeline
- Stratified cross-validation
- Multiple evaluation metrics
- Dataset registry
- Dataset ingestion
- Dataset provenance
- Dataset quality auditing

## Next — Model Comparison

- Logistic Regression
- Random Forest
- Gradient Boosting
- Same StratifiedKFold splits
- Same evaluation metrics
- Comparison table

## Dataset Harmonization

- Independent real-world heart-disease datasets
- Dataset provenance
- Schema mapping
- Feature normalization
- Target harmonization
- Missing-value strategy
- Duplicate detection
- Dataset-source tracking

## Unified Real-World Dataset

- Combine validated datasets
- Preserve source provenance
- Validate merged schema
- Validate target consistency
- Analyze dataset composition
- Re-run model comparison
- Compare performance before vs. after dataset expansion

## Advanced ML

- Hyperparameter tuning
- GridSearchCV
- RandomizedSearchCV
- Permutation importance
- SHAP
- LIME
- External/holdout validation
- Dataset-shift analysis

## MLOps

- MLflow / experiment tracking
- GitHub Actions
- pytest CI
- Ruff
- Black
- pre-commit
- Model versioning

## Deployment

- FastAPI inference service
- Docker
- Production inference pipeline
- API testing
- Deployment

---

# Research Philosophy
The central design principle of this project is:

> **Do not manufacture more data when real data can be obtained, validated, and harmonized.**

The project therefore prioritizes:

1. **Real datasets**
2. **Dataset provenance**
3. **Transparent preprocessing**
4. **Reproducible evaluation**
5. **Fair model comparison**
6. **External validation**
7. **Production-ready engineering**

Synthetic data is **not part of the planned dataset expansion strategy**.

The objective is not simply to maximize the number of rows.

The objective is to determine whether a model continues to perform reliably when exposed to data originating from different populations, institutions, collection protocols, and dataset distributions.

---

# Disclaimer
This project is intended for **educational, research, and software-engineering purposes**.

It is not a medical diagnostic system and should not be used to make clinical decisions.

### One important change
The four UCI sites should be treated as **one collection with four source domains**, not as four fully independent external datasets.

The research progression should be:

**UCI collection → harmonize the 4 sites → establish unified UCI baseline → add genuinely independent datasets → external validation.**

That distinction makes generalization claims much more defensible.

And yes: **synthetic data is completely removed from our planned workflow.** We will use real datasets only.
