
"""Model training and evaluation for credit default prediction."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data_processing import TARGET_COLUMN


def train_credit_model(data: pd.DataFrame):
    """Train a baseline model and return the model and test results."""

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' is missing."
        )

    X = data.drop(columns=[TARGET_COLUMN])
    y = data[TARGET_COLUMN]

    # Keep numeric predictors for this initial version.
    X = X.select_dtypes(include="number")

    if X.empty:
        raise ValueError("No numeric predictor columns were found.")

    if y.nunique() != 2:
        raise ValueError("The target must contain exactly two classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    )

    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    results = {
        "roc_auc": roc_auc_score(y_test, probabilities),
        "average_precision": average_precision_score(
            y_test, probabilities
        ),
        "brier_score": brier_score_loss(y_test, probabilities),
        "classification_report": classification_report(
            y_test, predictions, zero_division=0
        ),
        "test_observations": int(len(y_test)),
    }

    return model, results


def save_model(model, output_path="models/credit_risk_model.joblib"):
    """Save a trained model locally."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
