import numpy as np
import pandas as pd
import pytest

from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

from src.credit_risk_engine import (
    build_risk_portfolio,
    summarize_portfolio,
)


@pytest.fixture
def fitted_models():
    X_array, y = make_classification(
        n_samples=200,
        n_features=6,
        n_informative=4,
        n_redundant=0,
        random_state=42,
    )

    X = pd.DataFrame(
        X_array,
        columns=[f"feature_{i}" for i in range(6)],
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)

    anomaly_model = IsolationForest(
        n_estimators=50,
        random_state=42,
    )
    anomaly_model.fit(X.to_numpy())

    return X, model, anomaly_model


def test_build_risk_portfolio(fitted_models):
    X, model, anomaly_model = fitted_models

    result = build_risk_portfolio(
        X=X,
        model=model,
        anomaly_model=anomaly_model,
        threshold=0.30,
        anomaly_percentile=95,
    )

    assert len(result) == len(X)
    assert "probabilite_defaut" in result.columns
    assert "anomaly_score" in result.columns
    assert "is_anomaly" in result.columns
    assert "priorite" in result.columns

    assert result["probabilite_defaut"].between(0, 1).all()
    assert result["is_anomaly"].dtype == bool
    assert result["priorite"].notna().all()


def test_summarize_portfolio(fitted_models):
    X, model, anomaly_model = fitted_models

    portfolio = build_risk_portfolio(
        X=X,
        model=model,
        anomaly_model=anomaly_model,
    )

    summary = summarize_portfolio(portfolio)

    assert summary["nombre_dossiers"].sum() == len(X)
    assert "probabilite_moyenne" in summary.columns
    assert "anomalies" in summary.columns


def test_invalid_threshold_is_rejected(fitted_models):
    X, model, anomaly_model = fitted_models

    with pytest.raises(ValueError):
        build_risk_portfolio(
            X=X,
            model=model,
            anomaly_model=anomaly_model,
            threshold=1.5,
        )
