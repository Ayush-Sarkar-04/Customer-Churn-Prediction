import pandas as pd
import pytest

from src.analytics.future_value import estimate_future_customer_value


def _inputs():
    observations = pd.DataFrame({
        "customer_id": ["A", "B", "C"],
        "observation_date": ["2026-10-01"] * 3,
        "other_column": [1, 2, 3],
    })
    transactions = pd.DataFrame({
        "customer_id": ["A", "A", "A", "B", "C"],
        "transaction_date": [
            "2026-09-01", "2026-09-20", "2026-05-01",
            "2026-01-01", "2026-10-02",
        ],
        "bill_amount": [100, 200, 600, 365, 999],
    })
    return observations, transactions


def test_outputs_window_metrics_and_preserves_observation_columns():
    observations, transactions = _inputs()
    result = estimate_future_customer_value(observations, transactions)
    assert "other_column" in result
    assert result.loc[0, "historical_revenue_90d"] == 300
    assert result.loc[0, "historical_revenue_180d"] == 900
    assert result.loc[0, "transaction_count_90d"] == 2
    assert result.loc[0, "future_value_estimate_90d"] == 300
    assert result.loc[0, "future_value_method"] == "90d_history"


def test_fallback_uses_longer_window_when_recent_history_is_sparse():
    observations, transactions = _inputs()
    result = estimate_future_customer_value(observations, transactions, minimum_transactions=1)
    row = result.loc[result["customer_id"] == "B"].iloc[0]
    assert row["future_value_method"] == "365d_fallback"
    assert row["future_value_estimate_90d"] == pytest.approx(90)
    assert row["future_value_data_status"] == "fallback_history_used"


def test_no_history_is_explicit_and_estimate_is_missing():
    observations, transactions = _inputs()
    result = estimate_future_customer_value(observations, transactions)
    row = result.loc[result["customer_id"] == "C"].iloc[0]
    assert pd.isna(row["future_value_estimate_90d"])
    assert row["future_value_method"] == "no_history"
    assert row["future_value_data_status"] == "no_transaction_history"


def test_transactions_after_observation_date_are_excluded():
    observations = pd.DataFrame({
        "customer_id": ["A"], "observation_date": ["2026-10-01"]
    })
    transactions = pd.DataFrame({
        "customer_id": ["A", "A"],
        "transaction_date": ["2026-09-15", "2026-10-02"],
        "bill_amount": [100, 900],
    })
    result = estimate_future_customer_value(observations, transactions)
    assert result.loc[0, "historical_revenue_90d"] == 100


def test_same_day_transactions_are_included():
    observations = pd.DataFrame({
        "customer_id": ["A"], "observation_date": ["2026-10-01"]
    })
    transactions = pd.DataFrame({
        "customer_id": ["A"], "transaction_date": ["2026-10-01"],
        "bill_amount": [100],
    })
    result = estimate_future_customer_value(observations, transactions)
    assert result.loc[0, "historical_revenue_90d"] == 100


def test_invalid_probability_like_amount_is_not_accepted():
    observations = pd.DataFrame({
        "customer_id": ["A"], "observation_date": ["2026-10-01"]
    })
    transactions = pd.DataFrame({
        "customer_id": ["A"], "transaction_date": ["2026-09-01"],
        "bill_amount": [-1],
    })
    with pytest.raises(ValueError, match="negative"):
        estimate_future_customer_value(observations, transactions)


def test_missing_columns_are_rejected():
    with pytest.raises(ValueError, match="missing required columns"):
        estimate_future_customer_value(pd.DataFrame({"customer_id": ["A"]}), pd.DataFrame())


def test_invalid_windows_are_rejected():
    observations, transactions = _inputs()
    with pytest.raises(ValueError, match="positive integer"):
        estimate_future_customer_value(observations, transactions, windows=(90, 0))


def test_fallback_must_be_in_windows():
    observations, transactions = _inputs()
    with pytest.raises(ValueError, match="one of windows"):
        estimate_future_customer_value(
            observations, transactions, windows=(90, 180), fallback_window_days=365
        )


def test_inputs_are_not_modified():
    observations, transactions = _inputs()
    original_obs = observations.copy(deep=True)
    original_tx = transactions.copy(deep=True)
    estimate_future_customer_value(observations, transactions)
    pd.testing.assert_frame_equal(observations, original_obs)
    pd.testing.assert_frame_equal(transactions, original_tx)
