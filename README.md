# Heart Disease Predictor
A modular, reproducible machine-learning pipeline for **heart disease prediction and clinical dataset evaluation**.

The project began as a traditional heart-disease classification model and is evolving into a research-oriented ML pipeline focused on:

- Comparing multiple machine-learning models under identical evaluation conditions
- Ingesting heterogeneous real-world heart-disease datasets
- Preserving dataset provenance and source lineage
- Auditing dataset quality before harmonization
- Building a unified real-world dataset
- Evaluating model generalization across datasets
- Maintaining reproducible, testable ML infrastructure

> **Important:** This project does **not** use synthetic data. Dataset expansion is based exclusively on real-world datasets from documented sources.

---

## Project Status

### Stage 1 — Core ML Pipeline ✅

- Modular project structure
- Data preprocessing
- Model training
- Model saving/loading
- Prediction interface
- Automated tests

### Stage 2 — Cross-Validation ✅

- Stratified cross-validation
- Consistent folds
- Cross-validation scoring
- Mean performance
- Standard deviation
- Per-fold class distribution inspection

Current evaluation uses **StratifiedKFold** to maintain class proportions across validation folds.

### Stage 3 — Evaluation Metrics ✅
The pipeline currently evaluates:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- Classification Report

### Stage 4 — Model Comparison 🚧
Models being evaluated under the **same cross-validation strategy and metrics**:

- Logistic Regression
- Random Forest
- Gradient Boosting

The goal is to compare models fairly rather than optimizing one model in isolation.

---

# Dataset Infrastructure
The project now includes infrastructure for working with multiple real-world heart-disease datasets.

## UCI Heart Disease Dataset
The canonical UCI Heart Disease collection contains data from four clinical sites:

- Cleveland — Cleveland Clinic Foundation — 303 records
- Hungarian — Hungarian Institute of Cardiology — 294 records
- Switzerland — University Hospital, Zurich — 123 records
- VA — V.A. Medical Center, Long Beach — 200 records

The processed datasets contain the standard 14-column heart-disease schema.

The original UCI files are preserved under:

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

The processed datasets contain missing values represented by `?` and use different target distributions across collection sites.

For example:

- Cleveland: target values `0–4`
- Hungarian: target values `0–1`
- Switzerland: target values `0–4`
- VA: target values `0–4`

These differences are intentionally **not silently harmonized during ingestion**.

Harmonization is a separate pipeline stage.

---

# Dataset Provenance
Every ingested record receives provenance metadata.

Current provenance fields include:

```
source_collection
source_dataset
source_file
source_row_id
```

This allows individual observations to be traced back to their original dataset and source file.

Example:

```
source_collection = UCI Heart Disease
source_dataset    = uci_hd_cleveland
source_file       = processed.cleveland.data
source_row_id     = 42
```

This is important because combining datasets without preserving source information makes later validation and dataset-shift analysis difficult.

---

# Dataset Registry
`src/dataset_registry.py`

Provides a central registry describing available datasets.

Responsibilities include:

- Dataset identification
- Dataset metadata
- Source information
- File locations
- Schema information
- Target-column metadata
- Dataset discovery

Main interfaces:

```
list_dataset_ids()
list_datasets()
get_dataset_metadata()
```

Current registered datasets:

```
uci_hd_cleveland
uci_hd_hungarian
uci_hd_switzerland
uci_hd_va
```

---

# Dataset Ingestion
`src/dataset_ingestion.py`

Responsible for loading source datasets while preserving their original representation.

The ingestion layer:

- Loads raw/processed UCI files
- Preserves original values
- Avoids premature target transformation
- Attaches provenance metadata
- Produces a consistent DataFrame interface

Example provenance columns:

```
source_collection
source_dataset
source_file
source_row_id
```

The ingestion layer intentionally does **not** perform clinical harmonization.

---

# Dataset Quality Auditing
`src/dataset_quality.py`

Provides dataset-level inspection before datasets are merged.

Current auditing includes:

- Dataset dimensions
- Column structure
- Missing-value counts
- Target distribution
- Source metadata
- Raw schema inspection

Example:

```
dataset_audit("uci_hd_cleveland")
```

All registered datasets can also be audited:

```
audit_all_datasets()
```

The audit layer is intentionally conservative.
It reports what exists in the source data rather than making assumptions about how different datasets should be combined.

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
