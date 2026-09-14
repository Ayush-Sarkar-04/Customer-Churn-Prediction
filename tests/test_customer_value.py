import pandas as pd
import pytest

from src.analytics.customer_value import (
    calculate_customer_value,
    calculate_customer_value_summary,
    rank_customers_by_value,
)


def test_calculate_customer_value():
    transactions = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C001",
                "C002",
            ],
            "transaction_date": [
                "2026-01-01",
                "2026-01-11",
                "2026-02-01",
            ],
            "bill_amount": [
                100.0,
                200.0,
                500.0,
            ],
        }
    )

    result = calculate_customer_value(
        transactions
    )

    customer_1 = result[
        result["customer_id"] == "C001"
    ].iloc[0]

    customer_2 = result[
        result["customer_id"] == "C002"
    ].iloc[0]

    assert customer_1["total_historical_revenue"] == 300.0
    assert customer_1["purchase_count"] == 2
    assert customer_1["average_order_value"] == 150.0
    assert customer_1["active_days"] == 10
    assert customer_1["customer_value"] == 300.0

    assert customer_2["total_historical_revenue"] == 500.0
    assert customer_2["purchase_count"] == 1
    assert customer_2["average_order_value"] == 500.0
    assert customer_2["active_days"] == 1
    assert customer_2["customer_value"] == 500.0


def test_single_transaction_uses_one_active_day():
    transactions = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "transaction_date": ["2026-01-01"],
            "bill_amount": [250.0],
        }
    )

    result = calculate_customer_value(
        transactions
    )

    row = result.iloc[0]

    assert row["active_days"] == 1
    assert row["annualized_revenue"] == 250.0 * 365
    assert row["customer_value"] == 250.0


def test_customer_value_equals_historical_revenue():
    transactions = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
            ],
            "transaction_date": [
                "2026-01-01",
                "2026-01-02",
            ],
            "bill_amount": [
                100.0,
                400.0,
            ],
        }
    )

    result = calculate_customer_value(
        transactions
    )

    assert (
        result["customer_value"]
        == result["total_historical_revenue"]
    ).all()


def test_multiple_customers_are_returned():
    transactions = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
            ],
            "transaction_date": [
                "2026-01-01",
                "2026-01-02",
                "2026-01-03",
            ],
            "bill_amount": [
                100.0,
                200.0,
                300.0,
            ],
        }
    )

    result = calculate_customer_value(
        transactions
    )

    assert len(result) == 3
    assert set(result["customer_id"]) == {
        "C001",
        "C002",
        "C003",
    }


def test_empty_dataframe():
    transactions = pd.DataFrame(
        columns=[
            "customer_id",
            "transaction_date",
            "bill_amount",
        ]
    )

    result = calculate_customer_value(
        transactions
    )

    assert result.empty
    assert "customer_value" in result.columns


def test_missing_required_column():
    transactions = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "transaction_date": ["2026-01-01"],
        }
    )

    with pytest.raises(ValueError):
        calculate_customer_value(
            transactions
        )


def test_missing_customer_id():
    transactions = pd.DataFrame(
        {
            "customer_id": [None],
            "transaction_date": ["2026-01-01"],
            "bill_amount": [100.0],
        }
    )

    with pytest.raises(ValueError):
        calculate_customer_value(
            transactions
        )


def test_negative_bill_amount_rejected():
    transactions = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "transaction_date": ["2026-01-01"],
            "bill_amount": [-100.0],
        }
    )

    with pytest.raises(ValueError):
        calculate_customer_value(
            transactions
        )


def test_invalid_date_rejected():
    transactions = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "transaction_date": ["not-a-date"],
            "bill_amount": [100.0],
        }
    )

    with pytest.raises(
        (ValueError, TypeError)
    ):
        calculate_customer_value(
            transactions
        )


def test_customer_value_summary():
    customer_value = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
            ],
            "customer_value": [
                100.0,
                200.0,
                300.0,
            ],
            "annualized_revenue": [
                365.0,
                730.0,
                1095.0,
            ],
        }
    )

    summary = calculate_customer_value_summary(
        customer_value
    )

    assert summary[
        "total_customer_value"
    ] == 600.0

    assert summary[
        "average_customer_value"
    ] == 200.0

    assert summary[
        "median_customer_value"
    ] == 200.0

    assert summary[
        "total_annualized_revenue"
    ] == 2190.0

    assert summary[
        "average_annualized_revenue"
    ] == 730.0


def test_empty_customer_value_summary():
    customer_value = pd.DataFrame(
        columns=[
            "customer_value",
            "annualized_revenue",
        ]
    )

    summary = calculate_customer_value_summary(
        customer_value
    )

    assert summary == {
        "total_customer_value": 0.0,
        "average_customer_value": 0.0,
        "median_customer_value": 0.0,
        "total_annualized_revenue": 0.0,
        "average_annualized_revenue": 0.0,
    }


def test_rank_customers_by_value():
    customer_value = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
            ],
            "customer_value": [
                100.0,
                500.0,
                300.0,
            ],
        }
    )

    result = rank_customers_by_value(
        customer_value
    )

    assert result["customer_id"].tolist() == [
        "C002",
        "C003",
        "C001",
    ]

    assert result["value_rank"].tolist() == [
        1,
        2,
        3,
    ]


def test_rank_customers_top_n():
    customer_value = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
            ],
            "customer_value": [
                100.0,
                500.0,
                300.0,
            ],
        }
    )

    result = rank_customers_by_value(
        customer_value,
        top_n=2,
    )

    assert len(result) == 2
    assert result["customer_id"].tolist() == [
        "C002",
        "C003",
    ]


def test_invalid_top_n():
    customer_value = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "customer_value": [100.0],
        }
    )

    with pytest.raises(ValueError):
        rank_customers_by_value(
            customer_value,
            top_n=0,
        )


def test_rank_missing_column():
    customer_value = pd.DataFrame(
        {
            "customer_id": ["C001"],
        }
    )

    with pytest.raises(ValueError):
        rank_customers_by_value(
            customer_value
        )