"""Load a saved model and make predictions for patient records."""

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "models" / "logistic_regression.joblib"
)


def load_model(model_path: str | Path | None = None):
    """Load a trained model from disk."""
    target_path = Path(model_path or DEFAULT_MODEL_PATH)
    if not target_path.exists():
        raise FileNotFoundError(
            f"Model not found at {target_path}. Train the model first with src.train."
        )
    return joblib.load(target_path)


def _coerce_patient(patient: Any, model=None) -> pd.DataFrame:
    """Convert a patient record into a dataframe with the expected feature columns."""
    if isinstance(patient, pd.DataFrame):
        patient_df = patient.copy()
    elif isinstance(patient, dict):
        patient_df = pd.DataFrame([patient])
    else:
        raise TypeError("patient must be a dictionary or a pandas DataFrame")

    if model is not None and hasattr(model, "feature_names_in_"):
        patient_df = patient_df.reindex(
            columns=list(model.feature_names_in_), fill_value=0
        )

    return patient_df


def predict_from_patient(
    patient: Any, model_path: str | Path | None = None
) -> dict[str, float | int]:
    """Return a prediction and probability for a patient record."""
    model = load_model(model_path)
    patient_df = _coerce_patient(patient, model=model)
    prediction = int(model.predict(patient_df)[0])
    probability = float(model.predict_proba(patient_df)[0, 1])
    return {"prediction": prediction, "probability": probability}
