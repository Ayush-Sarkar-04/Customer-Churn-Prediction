import pandas as pd
import pytest

from src.analytics.customer_search import (
    search_customers,
    get_customer_profile,
)


def create_customer_data():
    return pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
                "C010",
            ],
            "recency": [
                10,
                25,
                60,
                100,
                15,
            ],
            "frequency": [
                20,
                15,
                5,
                2,
                12,
            ],
            "monetary": [
                5000,
                4000,
                1500,
                500,
                3200,
            ],
            "customer_segment": [
                "Champions",
                "Loyal Customers",
                "At Risk",
                "Lost",
                "Champions",
            ],
            "churn_probability": [
                0.05,
                0.20,
                0.65,
                0.90,
                0.10,
            ],
            "risk_level": [
                "Low",
                "Low",
                "High",
                "Churned",
                "Low",
            ],
        }
    )


def test_search_customers_exact_match():
    df = create_customer_data()

    result = search_customers(
        df,
        "C003",
    )

    assert len(result) == 1
    assert result.iloc[0]["customer_id"] == "C003"


def test_search_customers_partial_match():
    df = create_customer_data()

    result = search_customers(
        df,
        "C00",
    )

    assert len(result) == 4
    assert set(result["customer_id"]) == {
        "C001",
        "C002",
        "C003",
        "C004",
    }


def test_search_customers_case_insensitive():
    df = create_customer_data()

    result = search_customers(
        df,
        "c003",
    )

    assert len(result) == 1
    assert result.iloc[0]["customer_id"] == "C003"


def test_search_customers_empty_search():
    df = create_customer_data()

    result = search_customers(
        df,
        "",
    )

    assert len(result) == len(df)


def test_search_customers_missing_customer_id():
    df = pd.DataFrame(
        {
            "recency": [10],
            "frequency": [5],
        }
    )

    with pytest.raises(ValueError, match="customer_id"):
        search_customers(df, "C001")


def test_get_customer_profile():
    df = create_customer_data()

    result = get_customer_profile(
        df,
        "C003",
    )

    assert result["customer_id"] == "C003"
    assert result["recency"] == 60
    assert result["frequency"] == 5
    assert result["monetary"] == 1500
    assert result["customer_segment"] == "At Risk"
    assert result["risk_level"] == "High"


def test_get_customer_profile_not_found():
    df = create_customer_data()

    with pytest.raises(
        ValueError,
        match="was not found",
    ):
        get_customer_profile(
            df,
            "C999",
        )


def test_get_customer_profile_empty_id():
    df = create_customer_data()

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        get_customer_profile(
            df,
            "",
        )