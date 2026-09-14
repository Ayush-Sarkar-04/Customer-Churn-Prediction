from __future__ import annotations

import pandas as pd
import pytest

from src.analytics.economic_priority import (
    calculate_economic_priority,
    rank_economic_priority,
    summarize_economic_priority,
)


def make_dataframe() -> pd.DataFrame:
    """
    Create a deterministic test dataset.

    Net Value at Risk:
        C001 = 680
        C002 = 1000
        C003 = 1500
        C004 = 2500

    Total Net Value at Risk = 5680.

    Percentile scores:
        C001 = 25
        C002 = 50
        C003 = 75
        C004 = 100

    Priority tiers:
        C001 = Medium
        C002 = High
        C003 = Critical
        C004 = Critical
    """

    return pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
            ],
            "churn_probability": [
                0.80,
                0.60,
                0.60,
                0.60,
            ],
            "customer_value": [
                1000.0,
                2000.0,
                3000.0,
                5000.0,
            ],
            "expected_value_at_risk": [
                800.0,
                1200.0,
                1800.0,
                3000.0,
            ],
            "retention_cost": [
                120.0,
                200.0,
                300.0,
                500.0,
            ],
            "net_value_at_risk": [
                680.0,
                1000.0,
                1500.0,
                2500.0,
            ],
        }
    )


def test_calculate_economic_priority():
    df = make_dataframe()

    result = calculate_economic_priority(df)

    assert "economic_priority_score" in result.columns
    assert "economic_priority_tier" in result.columns

    assert len(result) == 4


def test_highest_net_value_gets_highest_score():
    df = make_dataframe()

    result = calculate_economic_priority(df)

    highest_score_customer = result.loc[
        result["economic_priority_score"].idxmax(),
        "customer_id",
    ]

    assert highest_score_customer == "C004"


def test_priority_tiers():
    df = make_dataframe()

    result = calculate_economic_priority(df)

    tiers = dict(
        zip(
            result["customer_id"],
            result["economic_priority_tier"],
        )
    )

    assert tiers["C001"] == "Medium"
    assert tiers["C002"] == "High"
    assert tiers["C003"] == "Critical"
    assert tiers["C004"] == "Critical"


def test_priority_score_matches_percentile_rank():
    df = make_dataframe()

    result = calculate_economic_priority(df)

    expected_scores = [
        25.0,
        50.0,
        75.0,
        100.0,
    ]

    assert (
        result["economic_priority_score"].tolist()
        == expected_scores
    )


def test_original_dataframe_not_modified():
    df = make_dataframe()

    original = df.copy(deep=True)

    calculate_economic_priority(df)

    pd.testing.assert_frame_equal(
        df,
        original,
    )


def test_invalid_probability():
    df = make_dataframe()

    df.loc[0, "churn_probability"] = 1.5

    with pytest.raises(ValueError):
        calculate_economic_priority(df)


def test_negative_customer_value_rejected():
    df = make_dataframe()

    df.loc[0, "customer_value"] = -100.0

    with pytest.raises(ValueError):
        calculate_economic_priority(df)


def test_negative_retention_cost_rejected():
    df = make_dataframe()

    df.loc[0, "retention_cost"] = -50.0

    with pytest.raises(ValueError):
        calculate_economic_priority(df)


def test_inconsistent_expected_value_at_risk():
    df = make_dataframe()

    df.loc[0, "expected_value_at_risk"] = 999.0

    with pytest.raises(ValueError):
        calculate_economic_priority(df)


def test_inconsistent_net_value_at_risk():
    df = make_dataframe()

    df.loc[0, "net_value_at_risk"] = 999.0

    with pytest.raises(ValueError):
        calculate_economic_priority(df)


def test_missing_required_column():
    df = make_dataframe()

    df = df.drop(
        columns=["customer_value"]
    )

    with pytest.raises(ValueError):
        calculate_economic_priority(df)


def test_invalid_dataframe_type():
    with pytest.raises(TypeError):
        calculate_economic_priority(
            "not a dataframe"
        )


def test_empty_dataframe():
    df = pd.DataFrame(
        columns=[
            "customer_id",
            "churn_probability",
            "customer_value",
            "expected_value_at_risk",
            "retention_cost",
            "net_value_at_risk",
        ]
    )

    result = calculate_economic_priority(df)

    assert result.empty

    assert (
        "economic_priority_score"
        in result.columns
    )

    assert (
        "economic_priority_tier"
        in result.columns
    )


def test_rank_economic_priority():
    df = make_dataframe()

    result = calculate_economic_priority(df)

    ranked = rank_economic_priority(result)

    assert len(ranked) == 4

    assert (
        ranked.iloc[0]["customer_id"]
        == "C004"
    )

    assert (
        ranked.iloc[0]["economic_priority_rank"]
        == 1
    )


def test_rank_top_n():
    df = make_dataframe()

    result = calculate_economic_priority(df)

    ranked = rank_economic_priority(
        result,
        top_n=2,
    )

    assert len(ranked) == 2

    assert (
        ranked.iloc[0]["customer_id"]
        == "C004"
    )

    assert (
        ranked.iloc[1]["customer_id"]
        == "C003"
    )


def test_invalid_top_n():
    df = make_dataframe()

    result = calculate_economic_priority(df)

    with pytest.raises(ValueError):
        rank_economic_priority(
            result,
            top_n=0,
        )


def test_summary():
    df = make_dataframe()

    result = calculate_economic_priority(df)

    summary = summarize_economic_priority(result)

    assert summary["customer_count"] == 4

    assert (
        summary["total_net_value_at_risk"]
        == 5680.0
    )

    assert (
        summary["critical_customers"]
        == 2
    )


def test_summary_high_or_critical_value():
    df = make_dataframe()

    result = calculate_economic_priority(df)

    summary = summarize_economic_priority(result)

    assert (
        summary["critical_net_value_at_risk"]
        == 4000.0
    )

    assert (
        summary[
            "high_or_critical_net_value_at_risk"
        ]
        == 5000.0
    )


def test_empty_summary():
    df = pd.DataFrame(
        columns=[
            "customer_id",
            "economic_priority_score",
            "economic_priority_tier",
            "net_value_at_risk",
        ]
    )

    summary = summarize_economic_priority(df)

    assert summary["customer_count"] == 0
    assert summary["critical_customers"] == 0
    assert summary["high_customers"] == 0
    assert summary["medium_customers"] == 0
    assert summary["low_customers"] == 0

    assert (
        summary["critical_net_value_at_risk"]
        == 0.0
    )

    assert (
        summary[
            "high_or_critical_net_value_at_risk"
        ]
        == 0.0
    )

    assert (
        summary["total_net_value_at_risk"]
        == 0.0
    )