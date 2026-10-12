
import pandas as pd
import pytest

from src.data_processing import TARGET_COLUMN, summarize_target


def test_summarize_target():
    data = pd.DataFrame(
        {TARGET_COLUMN: [0, 0, 1, 0, 1]}
    )

    result = summarize_target(data)

    assert result["observations"] == 5
    assert result["defaults"] == 2
    assert result["non_defaults"] == 3
    assert result["default_rate"] == pytest.approx(0.4)
