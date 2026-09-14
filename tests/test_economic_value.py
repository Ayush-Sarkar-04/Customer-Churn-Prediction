import pandas as pd
import pytest

from src.analytics.economic_value import (
    calculate_economic_value_at_risk,
    calculate_economic_value_summary,
    rank_customers_by_economic_value,
)


def test_expected_value_at_risk():
    df = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
            ],
            "churn_probability": [
                0.20,
                0.50,
            ],
            "customer_value": [
                1000.0,
                2000.0,
            ],
        }
    )

    result = calculate_economic_value_at_risk(df)

    assert result[
        "expected_value_at_risk"
    ].tolist() == [
        200.0,
        1000.0,
    ]


def test_net_value_at_risk():
    df = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
            ],
            "churn_probability": [
                0.20,
                0.50,
            ],
            "customer_value": [
                1000.0,
                2000.0,
            ],
            "retention_cost": [
                50.0,
                300.0,
            ],
        }
    )

    result = calculate_economic_value_at_risk(
        df,
        retention_cost_column="retention_cost",
    )

    assert result[
        "net_value_at_risk"
    ].tolist() == [
        150.0,
        700.0,
    ]


def test_retention_cost_defaults_to_zero():
    df = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "churn_probability": [0.25],
            "customer_value": [1000.0],
        }
    )

    result = calculate_economic_value_at_risk(df)

    assert result[
        "retention_cost"
    ].iloc[0] == 0.0

    assert result[
        "net_value_at_risk"
    ].iloc[0] == 250.0


def test_zero_churn_probability():
    df = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "churn_probability": [0.0],
            "customer_value": [1000.0],
        }
    )

    result = calculate_economic_value_at_risk(df)

    assert result[
        "expected_value_at_risk"
    ].iloc[0] == 0.0


def test_one_churn_probability():
    df = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "churn_probability": [1.0],
            "customer_value": [1000.0],
        }
    )

    result = calculate_economic_value_at_risk(df)

    assert result[
        "expected_value_at_risk"
    ].iloc[0] == 1000.0


def test_missing_required_column():
    df = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "customer_value": [1000.0],
        }
    )

    with pytest.raises(ValueError):
        calculate_economic_value_at_risk(df)


def test_invalid_probability_above_one():
    df = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "churn_probability": [1.5],
            "customer_value": [1000.0],
        }
    )

    with pytest.raises(ValueError):
        calculate_economic_value_at_risk(df)


def test_invalid_probability_below_zero():
    df = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "churn_probability": [-0.1],
            "customer_value": [1000.0],
        }
    )

    with pytest.raises(ValueError):
        calculate_economic_value_at_risk(df)


def test_negative_customer_value_rejected():
    df = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "churn_probability": [0.5],
            "customer_value": [-100.0],
        }
    )

    with pytest.raises(ValueError):
        calculate_economic_value_at_risk(df)


def test_negative_retention_cost_rejected():
    df = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "churn_probability": [0.5],
            "customer_value": [1000.0],
            "retention_cost": [-10.0],
        }
    )

    with pytest.raises(ValueError):
        calculate_economic_value_at_risk(
            df,
            retention_cost_column="retention_cost",
        )


def test_original_dataframe_not_modified():
    df = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "churn_probability": [0.5],
            "customer_value": [1000.0],
        }
    )

    original_columns = df.columns.tolist()

    calculate_economic_value_at_risk(df)

    assert df.columns.tolist() == original_columns


def test_rank_customers_by_economic_value():
    df = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
            ],
            "net_value_at_risk": [
                100.0,
                500.0,
                300.0,
            ],
        }
    )

    result = rank_customers_by_economic_value(df)

    assert result[
        "customer_id"
    ].tolist() == [
        "C002",
        "C003",
        "C001",
    ]

    assert result[
        "economic_priority_rank"
    ].tolist() == [
        1,
        2,
        3,
    ]


def test_rank_top_n():
    df = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
            ],
            "net_value_at_risk": [
                100.0,
                500.0,
                300.0,
            ],
        }
    )

    result = rank_customers_by_economic_value(
        df,
        top_n=2,
    )

    assert len(result) == 2

    assert result[
        "customer_id"
    ].tolist() == [
        "C002",
        "C003",
    ]


def test_invalid_top_n():
    df = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "net_value_at_risk": [100.0],
        }
    )

    with pytest.raises(ValueError):
        rank_customers_by_economic_value(
            df,
            top_n=0,
        )


def test_economic_value_summary():
    df = pd.DataFrame(
        {
            "customer_value": [
                1000.0,
                2000.0,
            ],
            "expected_value_at_risk": [
                200.0,
                1000.0,
            ],
            "retention_cost": [
                50.0,
                300.0,
            ],
            "net_value_at_risk": [
                150.0,
                700.0,
            ],
        }
    )

    summary = calculate_economic_value_summary(df)

    assert summary[
        "total_customer_value"
    ] == 3000.0

    assert summary[
        "total_expected_value_at_risk"
    ] == 1200.0

    assert summary[
        "total_retention_cost"
    ] == 350.0

    assert summary[
        "total_net_value_at_risk"
    ] == 850.0

    assert summary[
        "average_expected_value_at_risk"
    ] == 600.0

    assert summary[
        "average_net_value_at_risk"
    ] == 425.0


def test_empty_economic_value_summary():
    df = pd.DataFrame(
        columns=[
            "customer_value",
            "expected_value_at_risk",
            "retention_cost",
            "net_value_at_risk",
        ]
    )

    summary = calculate_economic_value_summary(df)

    assert summary == {
        "total_customer_value": 0.0,
        "total_expected_value_at_risk": 0.0,
        "total_retention_cost": 0.0,
        "total_net_value_at_risk": 0.0,
        "average_expected_value_at_risk": 0.0,
        "average_net_value_at_risk": 0.0,
    }


def test_summary_missing_column():
    df = pd.DataFrame(
        {
            "customer_value": [1000.0],
            "expected_value_at_risk": [200.0],
            "retention_cost": [50.0],
        }
    )

    with pytest.raises(ValueError):
        calculate_economic_value_summary(df)