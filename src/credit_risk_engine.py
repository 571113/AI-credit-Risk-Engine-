
"""Credit Risk Engine: scoring and portfolio monitoring."""

import numpy as np
import pandas as pd


def build_risk_portfolio(
    X,
    model,
    anomaly_model,
    threshold=0.30,
    anomaly_percentile=95,
):
    """
    Build a credit-risk monitoring table.

    Parameters
    ----------
    X : pandas.DataFrame
        Features in the exact order expected by the fitted model.
    model : fitted sklearn Pipeline
        Model providing predict_proba().
    anomaly_model : fitted IsolationForest
        Anomaly detector fitted on training data.
    threshold : float
        Default-probability threshold for elevated credit risk.
    anomaly_percentile : float
        Percentile above which observations are flagged as anomalies.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X doit être un DataFrame pandas.")

    if X.empty:
        raise ValueError("X ne doit pas être vide.")

    if not 0 < threshold < 1:
        raise ValueError("threshold doit être compris entre 0 et 1.")

    if not 0 < anomaly_percentile < 100:
        raise ValueError("anomaly_percentile doit être entre 0 et 100.")

    probabilities = model.predict_proba(X)[:, 1]

    # IsolationForest decision_function:
    # smaller values indicate more unusual observations.
    anomaly_scores = -anomaly_model.decision_function(X.to_numpy())

    cutoff = np.percentile(anomaly_scores, anomaly_percentile)
    is_anomaly = anomaly_scores >= cutoff
    high_credit_risk = probabilities >= threshold

    priority = np.select(
        [
            high_credit_risk & is_anomaly,
            is_anomaly,
            high_credit_risk,
        ],
        [
            "PRIORITE 1 - Risque élevé et anomalie",
            "PRIORITE 2 - Profil atypique",
            "PRIORITE 3 - Risque élevé",
        ],
        default="Surveillance standard",
    )

    result = X.copy()
    result["probabilite_defaut"] = probabilities
    result["anomaly_score"] = anomaly_scores
    result["is_anomaly"] = is_anomaly
    result["priorite"] = priority

    priority_order = {
        "PRIORITE 1 - Risque élevé et anomalie": 0,
        "PRIORITE 3 - Risque élevé": 1,
        "PRIORITE 2 - Profil atypique": 2,
        "Surveillance standard": 3,
    }

    result["_ordre"] = result["priorite"].map(priority_order)
    result = result.sort_values(
        ["_ordre", "probabilite_defaut"],
        ascending=[True, False],
    )
    result = result.drop(columns="_ordre")

    return result


def summarize_portfolio(portfolio):
    """Summarize the portfolio by monitoring priority."""
    summary = (
        portfolio.groupby("priorite", observed=True)
        .agg(
            nombre_dossiers=("probabilite_defaut", "size"),
            probabilite_moyenne=(
                "probabilite_defaut", "mean"
            ),
            anomalies=("is_anomaly", "sum"),
        )
        .reset_index()
    )

    return summary.sort_values(
        "nombre_dossiers", ascending=False
    )
