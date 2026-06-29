## Import Libaries
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# Load Dataset
df = pd.read_csv("Heart_Disease_Prediction.csv")

print(df.head())

print("\nShape")
print(df.shape)

# Create Target Variable
df["Heart Disease"] = (df["Heart Disease"].map({
        "Presence": 1,
        "Absence": 0
    })
)

print(
    df["Heart Disease"].value_counts()
)

# Seperate Features and Target
X = df.drop(
    "Heart Disease",
    axis=1 ## Droping the Heart disease Column
)

y = df["Heart Disease"] ## Target Variable

# Detect Numerical & Categorical columns
numerical_cols=(
    X.select_dtypes(
        include=np.number ## Only include numbers
    )
    .columns
)

categorical_cols=(
    X.select_dtypes(
        exclude=np.number ## Exclude numbers
    )
    .columns
)

print(numerical_cols)
print(categorical_cols)

# Build Preprocessing / Cleaning Data
numeric_transformer= Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median" ## Fill Missing -> Impute column median at null value
            )
        ),
        (
            "scaler",
            StandardScaler() ## Scaling so big numbers don't dominate
        )
    ]
)

categorical_transformer= Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent" ## Fill Missing -> Impute most frequently occured category
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

# Combine preprocessing
preprocessor = (
    ColumnTransformer( ## Applying Transformers to respective columns
        transformers=[
            (
                "num",
                numeric_transformer,
                numerical_cols
            ),
            (
                "cat",
                categorical_transformer,
                categorical_cols
            ),
        ]
    )
)

# Build Pipeline -> Logistic Regression
model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        ),
    ]
)

# Split Dataset -> Train & Test sets
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2, # 20% Test/Evaluate & 80% Traain/Learn
        random_state=42,
        stratify=y
    )
)

# Train Model
model.fit(
    X_train,
    y_train
)

# Predict Outcome and Probability Estimation
predictions=(
    model.predict(
        X_test
    )
)

probabilities=(
    model.predict_proba(
        X_test
    )[:, 1]
)

# Overall Evaluation / Insights
print("\nAccuracy:") ## Overall Accuracy/Correctness
print(
    accuracy_score(
        y_test,
        predictions
    )
)

print("\nROU AUC:") ##  Seperation Ability (between Heart Diseased (1) and Healthy(0))
print(
    roc_auc_score(
        y_test,
        probabilities
    )
)

print("\nConfusion Matrix:") # Breakdown of mistakes made by model
print(
    confusion_matrix(
        y_test,
        predictions
    )
)

print("\nClassification Report:") ## Overall Precision / recall / F1 report
print(
    classification_report(
        y_test,
        predictions
    )
)

# Test On Sample Patient
patient=(
    X_test.iloc[[0]]
)

prob=(
    model.predict_proba(
        patient
    )[0][1]
)

pred=(
    model.predict(
        patient
    )[0]
)

print(
    f"\nProbability: {prob:.2%}"
)

print(
    "Predictions:",
    (
        "Presence" if pred == 1
        else "Absence"
    )
)