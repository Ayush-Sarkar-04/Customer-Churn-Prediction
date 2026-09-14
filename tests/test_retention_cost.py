import pandas as pd
import pytest

from src.analytics.retention_cost import (
    aggregate_retention_cost_by_campaign_type,
    calculate_average_retention_cost,
    calculate_customer_retention_cost,
    calculate_retention_cost,
    calculate_total_retention_cost,
)


def test_calculate_retention_cost_with_campaign_cost_only():
    df = pd.DataFrame(
        {
            "campaign_id": ["C1", "C2"],
            "campaign_cost": [10.0, 20.0],
        }
    )

    result = calculate_retention_cost(df)

    assert result["incentive_cost"].tolist() == [
        0.0,
        0.0,
    ]

    assert result["other_retention_cost"].tolist() == [
        0.0,
        0.0,
    ]

    assert result["retention_cost"].tolist() == [
        10.0,
        20.0,
    ]


def test_calculate_retention_cost_with_all_components():
    df = pd.DataFrame(
        {
            "campaign_cost": [10.0, 20.0],
            "incentive_cost_raw": [5.0, 7.0],
            "other_cost_raw": [2.0, 3.0],
        }
    )

    result = calculate_retention_cost(
        df,
        incentive_cost_column="incentive_cost_raw",
        other_cost_column="other_cost_raw",
    )

    assert result["retention_cost"].tolist() == [
        17.0,
        30.0,
    ]


def test_original_dataframe_is_not_modified():
    df = pd.DataFrame(
        {
            "campaign_cost": [10.0, 20.0],
        }
    )

    original_columns = df.columns.tolist()

    calculate_retention_cost(df)

    assert df.columns.tolist() == original_columns


def test_missing_campaign_cost_column():
    df = pd.DataFrame(
        {
            "campaign_id": ["C1"],
        }
    )

    with pytest.raises(ValueError):
        calculate_retention_cost(df)


def test_missing_optional_cost_column():
    df = pd.DataFrame(
        {
            "campaign_cost": [10.0],
        }
    )

    with pytest.raises(ValueError):
        calculate_retention_cost(
            df,
            incentive_cost_column="missing_column",
        )


def test_negative_campaign_cost_rejected():
    df = pd.DataFrame(
        {
            "campaign_cost": [-10.0],
        }
    )

    with pytest.raises(ValueError):
        calculate_retention_cost(df)


def test_missing_cost_value_rejected():
    df = pd.DataFrame(
        {
            "campaign_cost": [10.0, None],
        }
    )

    with pytest.raises(ValueError):
        calculate_retention_cost(df)


def test_total_retention_cost():
    df = pd.DataFrame(
        {
            "retention_cost": [10.0, 20.0, 30.0],
        }
    )

    assert calculate_total_retention_cost(df) == 60.0


def test_average_retention_cost():
    df = pd.DataFrame(
        {
            "retention_cost": [10.0, 20.0, 30.0],
        }
    )

    assert calculate_average_retention_cost(df) == 20.0


def test_empty_average_retention_cost():
    df = pd.DataFrame(
        {
            "retention_cost": pd.Series(dtype=float),
        }
    )

    assert calculate_average_retention_cost(df) == 0.0


def test_campaign_type_aggregation():
    df = pd.DataFrame(
        {
            "campaign_type": [
                "Discount",
                "Discount",
                "Birthday",
            ],
            "retention_cost": [
                10.0,
                20.0,
                15.0,
            ],
        }
    )

    result = aggregate_retention_cost_by_campaign_type(df)

    discount = result[
        result["campaign_type"] == "Discount"
    ].iloc[0]

    birthday = result[
        result["campaign_type"] == "Birthday"
    ].iloc[0]

    assert discount["campaign_count"] == 2
    assert discount["total_retention_cost"] == 30.0
    assert discount["average_retention_cost"] == 15.0

    assert birthday["campaign_count"] == 1
    assert birthday["total_retention_cost"] == 15.0


def test_customer_retention_cost():
    df = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C001",
                "C002",
            ],
            "retention_cost": [
                10.0,
                20.0,
                15.0,
            ],
        }
    )

    result = calculate_customer_retention_cost(df)

    customer_1 = result[
        result["customer_id"] == "C001"
    ].iloc[0]

    customer_2 = result[
        result["customer_id"] == "C002"
    ].iloc[0]

    assert customer_1["campaign_count"] == 2
    assert customer_1["total_retention_cost"] == 30.0
    assert customer_1["average_retention_cost"] == 15.0

    assert customer_2["campaign_count"] == 1
    assert customer_2["total_retention_cost"] == 15.0


def test_customer_retention_cost_missing_customer_id():
    df = pd.DataFrame(
        {
            "customer_id": ["C001", None],
            "retention_cost": [10.0, 20.0],
        }
    )

    with pytest.raises(ValueError):
        calculate_customer_retention_cost(df)


def test_customer_retention_cost_negative_cost():
    df = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "retention_cost": [-10.0],
        }
    )

    with pytest.raises(ValueError):
        calculate_customer_retention_cost(df)