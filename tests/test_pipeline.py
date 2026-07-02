from pathlib import Path

from sklearn.model_selection import train_test_split

from src.preprocess import build_pipeline, load_data, split_features_target
from src.predict import predict_from_patient
from src.train import train_model

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "heart.csv"


def test_load_data_encodes_target_column():
    df = load_data(DATA_PATH)

    assert not df.empty
    assert "Heart Disease" in df.columns
    assert set(df["Heart Disease"].unique()).issubset({0, 1})


def test_build_pipeline_fits_and_predicts():
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline(X_train)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    assert len(predictions) == len(y_test)


def test_predict_from_patient_returns_probability_and_prediction(tmp_path):
    model_path = tmp_path / "model.joblib"
    train_model(data_path=DATA_PATH, model_path=model_path)

    patient = {
        "Age": 63,
        "Sex": 1,
        "Chest Pain Type": "typical angina",
        "Resting BP": 145,
        "Cholesterol": 233,
        "Fasting Blood Sugar": 1,
        "Resting ECG": "normal",
        "Max HR": 150,
        "Exercise Angina": "no",
        "Oldpeak": 2.3,
        "ST Slope": "upsloping",
    }

    result = predict_from_patient(patient, model_path=model_path)

    assert set(result.keys()) == {"prediction", "probability"}
    assert result["prediction"] in {0, 1}
    assert 0 <= result["probability"] <= 1
