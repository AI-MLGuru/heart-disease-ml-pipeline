"""Reusable preprocessing utilities for the heart disease model."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "Heart Disease"
TARGET_MAPPING = {"Presence": 1, "Absence": 0}


def load_data(data_path: Optional[Path | str] = None) -> pd.DataFrame:
    """Load the heart disease dataset and encode the target column."""
    if data_path is None:
        data_path = Path(__file__).resolve().parents[1] / "data" / "heart.csv"

    df = pd.read_csv(data_path)

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = (
            df[TARGET_COLUMN].map(TARGET_MAPPING).fillna(df[TARGET_COLUMN]).astype(int)
        )

    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split the dataframe into feature and target arrays."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int)
    return X, y


def get_feature_types(X: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Separate numeric and categorical columns."""
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    return numerical_cols, categorical_cols


def build_preprocessor(X: Optional[pd.DataFrame] = None) -> ColumnTransformer:
    """Create a preprocessing transformer for numeric and categorical columns."""
    if X is None:
        df = load_data()
        X, _ = split_features_target(df)

    numerical_cols, categorical_cols = get_feature_types(X)

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def build_pipeline(X: Optional[pd.DataFrame] = None) -> Pipeline:
    """Build the full training pipeline with preprocessing and logistic regression."""
    preprocessor = build_preprocessor(X)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split the input data into train and test sets."""
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
