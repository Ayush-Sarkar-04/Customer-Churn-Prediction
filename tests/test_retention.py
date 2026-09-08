import pandas as pd
import pytest

from src.analytics.retention import (
    calculate_customer_value_percentile,
    classify_retention_priority,
    calculate_retention_priority,
)


def test_customer_value_percentile():
    monetary = pd.Series(
        [100, 200, 300]
    )

    result = calculate_customer_value_percentile(
        monetary
    )

    assert len(result) == 3
    assert result.min() >= 0
    assert result.max() <= 1
    assert result.iloc[0] < result.iloc[1] < result.iloc[2]


def test_classify_retention_priority():
    assert classify_retention_priority(10) == "Low"
    assert classify_retention_priority(25) == "Medium"
    assert classify_retention_priority(50) == "High"
    assert classify_retention_priority(75) == "Critical"


def test_classify_retention_priority_invalid():
    with pytest.raises(ValueError):
        classify_retention_priority(-1)

    with pytest.raises(ValueError):
        classify_retention_priority(101)


def test_calculate_retention_priority():
    df = pd.DataFrame({
        "customer_id": [
            "C001",
            "C002",
            "C003",
        ],
        "churn_probability": [
            0.10,
            0.50,
            0.90,
        ],
        "monetary": [
            1000,
            5000,
            10000,
        ],
    })

    result = calculate_retention_priority(df)

    assert "customer_value_percentile" in result.columns
    assert "retention_priority_score" in result.columns
    assert "retention_priority" in result.columns
    assert "expected_revenue_at_risk" in result.columns

    assert result["retention_priority_score"].between(
        0, 100
    ).all()

    assert (
        result["expected_revenue_at_risk"].iloc[0]
        == 100
    )

    assert (
        result["expected_revenue_at_risk"].iloc[2]
        == 9000
    )


def test_calculate_retention_priority_missing_columns():
    df = pd.DataFrame({
        "customer_id": ["C001"]
    })

    with pytest.raises(ValueError):
        calculate_retention_priority(df)


def test_calculate_retention_priority_invalid_probability():
    df = pd.DataFrame({
        "churn_probability": [1.2],
        "monetary": [5000],
    })

    with pytest.raises(ValueError):
        calculate_retention_priority(df)