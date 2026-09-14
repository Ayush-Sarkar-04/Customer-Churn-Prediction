from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "customer_id",
    "transaction_date",
    "bill_amount",
]


def _validate_required_columns(
    df: pd.DataFrame,
) -> None:
    """Validate that all required transaction columns exist."""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def _prepare_transactions(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare transaction data for customer value calculations."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    _validate_required_columns(df)

    result = df.copy()

    result["transaction_date"] = pd.to_datetime(
        result["transaction_date"],
        errors="raise",
    )

    result["bill_amount"] = pd.to_numeric(
        result["bill_amount"],
        errors="raise",
    )

    if result["customer_id"].isna().any():
        raise ValueError(
            "Transaction data contains missing customer IDs."
        )

    if result["transaction_date"].isna().any():
        raise ValueError(
            "Transaction data contains missing transaction dates."
        )

    if result["bill_amount"].isna().any():
        raise ValueError(
            "Transaction data contains missing bill amounts."
        )

    if (result["bill_amount"] < 0).any():
        raise ValueError(
            "Transaction data contains negative bill amounts."
        )

    return result


def calculate_customer_value(
    transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate historical and annualized customer value.

    Metrics:
        - total_historical_revenue
        - purchase_count
        - average_order_value
        - active_days
        - annualized_revenue
        - customer_value

    customer_value is currently defined as historical revenue.

    annualized_revenue provides a normalized value estimate based
    on the observed customer activity period.

    This is not a predictive lifetime value model.
    """

    df = _prepare_transactions(transactions)

    if df.empty:
        return pd.DataFrame(
            columns=[
                "customer_id",
                "total_historical_revenue",
                "purchase_count",
                "average_order_value",
                "active_days",
                "annualized_revenue",
                "customer_value",
            ]
        )

    grouped = (
        df.groupby("customer_id")
        .agg(
            total_historical_revenue=(
                "bill_amount",
                "sum",
            ),
            purchase_count=(
                "bill_amount",
                "count",
            ),
            first_transaction_date=(
                "transaction_date",
                "min",
            ),
            last_transaction_date=(
                "transaction_date",
                "max",
            ),
        )
        .reset_index()
    )

    grouped["average_order_value"] = (
        grouped["total_historical_revenue"]
        / grouped["purchase_count"]
    )

    grouped["active_days"] = (
        grouped["last_transaction_date"]
        - grouped["first_transaction_date"]
    ).dt.days

    grouped["active_days"] = (
        grouped["active_days"]
        .clip(lower=1)
    )

    grouped["annualized_revenue"] = (
        grouped["total_historical_revenue"]
        / grouped["active_days"]
        * 365
    )

    grouped["customer_value"] = (
        grouped["total_historical_revenue"]
    )

    return grouped[
        [
            "customer_id",
            "total_historical_revenue",
            "purchase_count",
            "average_order_value",
            "active_days",
            "annualized_revenue",
            "customer_value",
        ]
    ]


def calculate_customer_value_summary(
    customer_value: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate portfolio-level customer value summary.

    Returns:
        total_customer_value
        average_customer_value
        median_customer_value
        total_annualized_revenue
        average_annualized_revenue
    """

    required_columns = [
        "customer_value",
        "annualized_revenue",
    ]

    missing = [
        column
        for column in required_columns
        if column not in customer_value.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if customer_value.empty:
        return {
            "total_customer_value": 0.0,
            "average_customer_value": 0.0,
            "median_customer_value": 0.0,
            "total_annualized_revenue": 0.0,
            "average_annualized_revenue": 0.0,
        }

    return {
        "total_customer_value": float(
            customer_value["customer_value"].sum()
        ),
        "average_customer_value": float(
            customer_value["customer_value"].mean()
        ),
        "median_customer_value": float(
            customer_value["customer_value"].median()
        ),
        "total_annualized_revenue": float(
            customer_value["annualized_revenue"].sum()
        ),
        "average_annualized_revenue": float(
            customer_value["annualized_revenue"].mean()
        ),
    }


def rank_customers_by_value(
    customer_value: pd.DataFrame,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Rank customers by historical customer value.

    Higher customer value receives a higher rank.
    """

    required_columns = [
        "customer_id",
        "customer_value",
    ]

    missing = [
        column
        for column in required_columns
        if column not in customer_value.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    result = customer_value.copy()

    result["value_rank"] = (
        result["customer_value"]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    result = result.sort_values(
        by=[
            "customer_value",
            "customer_id",
        ],
        ascending=[
            False,
            True,
        ],
    ).reset_index(drop=True)

    if top_n is not None:
        if top_n < 1:
            raise ValueError(
                "top_n must be at least 1."
            )

        result = result.head(top_n)

    return result