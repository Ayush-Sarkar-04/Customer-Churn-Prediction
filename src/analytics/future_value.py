"""Recency-aware customer value estimates for churn economics.

All calculations are relative to a customer's observation date and exclude
transactions after that date. Estimates are descriptive extrapolations from
historical spend, not predictive lifetime value or causal estimates of revenue
saved by retention activity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_TRANSACTION_COLUMNS = {"customer_id", "transaction_date", "bill_amount"}
REQUIRED_OBSERVATION_COLUMNS = {"customer_id", "observation_date"}
WINDOW_DAYS = (90, 180, 365)


def _validate_frame(df: pd.DataFrame, required: set[str], name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _prepare_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    _validate_frame(transactions, REQUIRED_TRANSACTION_COLUMNS, "transactions")
    data = transactions.copy()

    if data["customer_id"].isna().any():
        raise ValueError("transactions contains missing customer IDs.")

    data["customer_id"] = data["customer_id"].astype(str).str.strip()
    if data["customer_id"].eq("").any():
        raise ValueError("transactions contains blank customer IDs.")

    data["transaction_date"] = pd.to_datetime(
        data["transaction_date"], errors="coerce", dayfirst=True, format="mixed"
    )
    if data["transaction_date"].isna().any():
        raise ValueError("transactions contains invalid transaction dates.")

    data["bill_amount"] = pd.to_numeric(data["bill_amount"], errors="coerce")
    if data["bill_amount"].isna().any() or not np.isfinite(data["bill_amount"]).all():
        raise ValueError("bill_amount must contain finite numeric values.")
    if (data["bill_amount"] < 0).any():
        raise ValueError("bill_amount cannot contain negative values.")

    return data


def _prepare_observations(observations: pd.DataFrame) -> pd.DataFrame:
    _validate_frame(observations, REQUIRED_OBSERVATION_COLUMNS, "observations")
    data = observations.copy()

    if data["customer_id"].isna().any():
        raise ValueError("observations contains missing customer IDs.")
    data["customer_id"] = data["customer_id"].astype(str).str.strip()
    if data["customer_id"].eq("").any():
        raise ValueError("observations contains blank customer IDs.")

    data["observation_date"] = pd.to_datetime(
        data["observation_date"], errors="coerce", dayfirst=True, format="mixed"
    )
    if data["observation_date"].isna().any():
        raise ValueError("observations contains invalid observation dates.")
    return data


def estimate_future_customer_value(
    observations: pd.DataFrame,
    transactions: pd.DataFrame,
    windows: tuple[int, ...] = WINDOW_DAYS,
    fallback_window_days: int = 365,
    minimum_transactions: int = 2,
) -> pd.DataFrame:
    """Estimate value over the next 90 days using trailing transaction history.

    Parameters
    ----------
    observations:
        One or more rows with ``customer_id`` and ``observation_date``.
        All additional columns are preserved.
    transactions:
        Transaction rows with ``customer_id``, ``transaction_date`` and
        ``bill_amount``.
    windows:
        Trailing lookback windows to calculate, defaulting to 90, 180 and 365
        days. Each window's observed revenue is scaled to a 90-day equivalent.
    fallback_window_days:
        Lookback window used for the primary estimate when the 90-day window
        has fewer than ``minimum_transactions`` transactions. Must be present
        in ``windows``.
    minimum_transactions:
        Minimum count in the 90-day window required to use the 90-day rate.

    Returns
    -------
    DataFrame
        Original observation rows plus historical-window metrics and:
        ``future_value_estimate_90d``, ``future_value_method``,
        ``future_value_transaction_count`` and ``future_value_data_status``.

    Notes
    -----
    The estimate is a simple spend-rate extrapolation, not a trained forecast.
    It does not adjust for seasonality, margin, churn intervention effect or
    uncertainty. Insufficient history is explicitly flagged.
    """
    if not windows or any(not isinstance(day, (int, np.integer)) or day <= 0 for day in windows):
        raise ValueError("windows must contain positive integer day counts.")
    if len(set(windows)) != len(windows):
        raise ValueError("windows must not contain duplicate day counts.")
    if fallback_window_days not in windows:
        raise ValueError("fallback_window_days must be one of windows.")
    if not isinstance(minimum_transactions, int) or minimum_transactions < 1:
        raise ValueError("minimum_transactions must be a positive integer.")

    obs = _prepare_observations(observations)
    tx = _prepare_transactions(transactions)

    result = obs.copy()
    # Avoid accidental collision with input columns when recomputing outputs.
    output_columns = []
    for days in windows:
        output_columns.extend([
            f"historical_revenue_{days}d",
            f"transaction_count_{days}d",
            f"revenue_rate_90d_{days}d",
        ])
    output_columns.extend([
        "future_value_estimate_90d",
        "future_value_method",
        "future_value_transaction_count",
        "future_value_data_status",
    ])
    result = result.drop(columns=[c for c in output_columns if c in result.columns])

    tx_by_customer = {
        customer_id: group.sort_values("transaction_date")
        for customer_id, group in tx.groupby("customer_id", sort=False)
    }

    metrics: list[dict[str, object]] = []
    for row in obs.itertuples(index=False):
        customer_id = getattr(row, "customer_id")
        observation_date = getattr(row, "observation_date")
        customer_tx = tx_by_customer.get(customer_id)
        if customer_tx is None:
            customer_tx = tx.iloc[0:0]

        # Same calendar-date transactions are considered observed by end of day.
        eligible = customer_tx.loc[
            customer_tx["transaction_date"] < observation_date + pd.Timedelta(days=1)
        ]
        record: dict[str, object] = {}
        for days in windows:
            start_date = observation_date - pd.Timedelta(days=days - 1)
            window_tx = eligible.loc[eligible["transaction_date"] >= start_date]
            revenue = float(window_tx["bill_amount"].sum())
            count = int(len(window_tx))
            record[f"historical_revenue_{days}d"] = revenue
            record[f"transaction_count_{days}d"] = count
            record[f"revenue_rate_90d_{days}d"] = revenue * 90.0 / days

        recent_count = int(record["transaction_count_90d"]) if 90 in windows else 0
        if 90 in windows and recent_count >= minimum_transactions:
            estimate = float(record["revenue_rate_90d_90d"])
            method = "90d_history"
            used_count = recent_count
            status = "sufficient_recent_history"
        else:
            fallback_count = int(record[f"transaction_count_{fallback_window_days}d"])
            fallback_revenue = float(record[f"historical_revenue_{fallback_window_days}d"])
            if fallback_count >= minimum_transactions:
                estimate = fallback_revenue * 90.0 / fallback_window_days
                method = f"{fallback_window_days}d_fallback"
                used_count = fallback_count
                status = "fallback_history_used"
            elif fallback_count > 0:
                # Preserve the observed spend signal, but explicitly mark it weak.
                estimate = fallback_revenue * 90.0 / fallback_window_days
                method = f"{fallback_window_days}d_low_history"
                used_count = fallback_count
                status = "insufficient_history"
            else:
                estimate = np.nan
                method = "no_history"
                used_count = 0
                status = "no_transaction_history"

        record["future_value_estimate_90d"] = estimate
        record["future_value_method"] = method
        record["future_value_transaction_count"] = used_count
        record["future_value_data_status"] = status
        metrics.append(record)

    metrics_df = pd.DataFrame(metrics, index=obs.index)
    return pd.concat([result, metrics_df], axis=1)
