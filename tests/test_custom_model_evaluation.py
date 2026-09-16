import pandas as pd
import pytest

from src.analytics.custom_model_evaluation import (
    _calculate_future_90d_outcomes,

)


def test_future_purchase_within_90_days_is_non_churn():
    transactions = pd.DataFrame(
        {
            "customer_id": ["C001", "C001"],
            "transaction_date": ["01-01-2025", "31-01-2025"],
        }
    )

    result = _calculate_future_90d_outcomes(
        ["C001"],
        pd.Timestamp("2025-01-01"),
        transactions,
    )

    assert result.loc[0, "future_90d_purchases"] == 1
    assert result.loc[0, "churn"] == 0


def test_no_future_purchase_is_churn():
    transactions = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "transaction_date": ["01-01-2025"],
        }
    )

    result = _calculate_future_90d_outcomes(
        ["C001"],
        pd.Timestamp("2025-01-01"),
        transactions,
    )

    assert result.loc[0, "future_90d_purchases"] == 0
    assert result.loc[0, "churn"] == 1


def test_purchase_on_day_90_is_included():
    transactions = pd.DataFrame(
        {
            "customer_id": ["C001", "C001"],
            "transaction_date": ["01-01-2025", "01-04-2025"],
        }
    )

    result = _calculate_future_90d_outcomes(
        ["C001"],
        pd.Timestamp("2025-01-01"),
        transactions,
    )

    assert result.loc[0, "future_90d_purchases"] == 1
    assert result.loc[0, "churn"] == 0


def test_purchase_after_day_90_is_not_included():
    transactions = pd.DataFrame(
        {
            "customer_id": ["C001", "C001"],
            "transaction_date": ["01-01-2025", "02-04-2025"],
        }
    )

    result = _calculate_future_90d_outcomes(
        ["C001"],
        pd.Timestamp("2025-01-01"),
        transactions,
    )

    assert result.loc[0, "future_90d_purchases"] == 0
    assert result.loc[0, "churn"] == 1


def test_customer_purchases_do_not_affect_another_customer():
    transactions = pd.DataFrame(
        {
            "customer_id": ["C001", "C002"],
            "transaction_date": ["01-01-2025", "15-01-2025"],
        }
    )

    result = _calculate_future_90d_outcomes(
        ["C001"],
        pd.Timestamp("2025-01-01"),
        transactions,
    )

    assert result.loc[0, "churn"] == 1


def test_multiple_future_purchases_are_counted():
    transactions = pd.DataFrame(
        {
            "customer_id": ["C001", "C001", "C001"],
            "transaction_date": [
                "01-01-2025",
                "15-01-2025",
                "15-02-2025",
            ],
        }
    )

    result = _calculate_future_90d_outcomes(
        ["C001"],
        pd.Timestamp("2025-01-01"),
        transactions,
    )

    assert result.loc[0, "future_90d_purchases"] == 2
    assert result.loc[0, "churn"] == 0


def test_invalid_future_window_is_rejected():
    transactions = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "transaction_date": ["01-01-2025"],
        }
    )

    with pytest.raises(ValueError, match="future_window_days"):
        _calculate_future_90d_outcomes(
            ["C001"],
            pd.Timestamp("2025-01-01"),
            transactions,
            future_window_days=0,
        )


def test_missing_transaction_date_is_rejected():
    transactions = pd.DataFrame(
        {
            "customer_id": ["C001"],
        }
    )

    with pytest.raises(
        ValueError,
        match="transaction_date",
    ):
        _calculate_future_90d_outcomes(
            ["C001"],
            pd.Timestamp("2025-01-01"),
            transactions,
        )


def test_original_transactions_are_not_modified():
    transactions = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "transaction_date": ["01-01-2025"],
        }
    )
    original = transactions.copy(deep=True)

    _calculate_future_90d_outcomes(
        ["C001"],
        pd.Timestamp("2025-01-01"),
        transactions,
    )

    pd.testing.assert_frame_equal(transactions, original)


def test_output_contains_required_observed_outcome_fields():
    transactions = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "transaction_date": ["01-01-2025"],
        }
    )

    result = _calculate_future_90d_outcomes(
        ["C001"],
        pd.Timestamp("2025-01-01"),
        transactions,
    )

    assert list(result.columns) == [
        "customer_id",
        "future_90d_purchases",
        "churn",
    ]
