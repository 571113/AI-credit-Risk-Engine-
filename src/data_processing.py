
"""Data loading and validation for the AI Credit Risk Engine."""

from pathlib import Path

import pandas as pd


TARGET_COLUMN = "default.payment.next.month"


def load_credit_data(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset and validate its basic structure."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            "Download the dataset and place the CSV at this path."
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("The dataset is empty.")

    if TARGET_COLUMN not in data.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(data.columns)}"
        )

    if data[TARGET_COLUMN].isna().any():
        raise ValueError("The target column contains missing values.")

    return data


def summarize_target(data: pd.DataFrame) -> dict:
    """Return basic statistics for the default target."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN}")

    target = data[TARGET_COLUMN]

    return {
        "observations": int(len(data)),
        "defaults": int(target.sum()),
        "default_rate": float(target.mean()),
        "non_defaults": int((target == 0).sum()),
    }
