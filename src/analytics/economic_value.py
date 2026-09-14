from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "customer_id",
    "churn_probability",
    "customer_value",
]


def _validate_required_columns(
    df: pd.DataFrame,
) -> None:
    """Validate required columns."""

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


def _validate_probability(
    series: pd.Series,
    column_name: str,
) -> None:
    """Validate that probabilities are between 0 and 1."""

    values = pd.to_numeric(
        series,
        errors="raise",
    )

    if values.isna().any():
        raise ValueError(
            f"Column '{column_name}' contains missing values."
        )

    if ((values < 0) | (values > 1)).any():
        raise ValueError(
            f"Column '{column_name}' must contain values "
            "between 0 and 1."
        )


def _validate_non_negative(
    series: pd.Series,
    column_name: str,
) -> None:
    """Validate that a numeric series contains no negative values."""

    values = pd.to_numeric(
        series,
        errors="raise",
    )

    if values.isna().any():
        raise ValueError(
            f"Column '{column_name}' contains missing values."
        )

    if (values < 0).any():
        raise ValueError(
            f"Column '{column_name}' contains negative values."
        )


def calculate_economic_value_at_risk(
    df: pd.DataFrame,
    churn_probability_column: str = "churn_probability",
    customer_value_column: str = "customer_value",
    retention_cost_column: str | None = None,
) -> pd.DataFrame:
    """
    Calculate Expected Value at Risk and Net Value at Risk.

    Expected Value at Risk (EVaR):
        churn_probability × customer_value

    Net Value at Risk:
        EVaR − retention_cost

    EVaR represents the customer's value exposed to predicted churn.

    Net Value at Risk accounts for the cost of a potential retention
    intervention.

    These metrics do NOT represent:
        - guaranteed revenue loss
        - guaranteed revenue saved
        - campaign ROI
        - incremental treatment effect

    Parameters
    ----------
    df:
        DataFrame containing customer-level churn probability
        and customer value.

    churn_probability_column:
        Column containing predicted churn probabilities.

    customer_value_column:
        Column containing customer value.

    retention_cost_column:
        Optional column containing retention cost.

        If omitted, retention cost is assumed to be zero.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with:
            - retention_cost
            - expected_value_at_risk
            - net_value_at_risk
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    required_columns = [
        churn_probability_column,
        customer_value_column,
    ]

    if retention_cost_column is not None:
        required_columns.append(
            retention_cost_column
        )

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    result = df.copy()

    _validate_probability(
        result[churn_probability_column],
        churn_probability_column,
    )

    _validate_non_negative(
        result[customer_value_column],
        customer_value_column,
    )

    result[churn_probability_column] = pd.to_numeric(
        result[churn_probability_column],
        errors="raise",
    )

    result[customer_value_column] = pd.to_numeric(
        result[customer_value_column],
        errors="raise",
    )

    if retention_cost_column is None:
        result["retention_cost"] = 0.0
    else:
        _validate_non_negative(
            result[retention_cost_column],
            retention_cost_column,
        )

        result[retention_cost_column] = pd.to_numeric(
            result[retention_cost_column],
            errors="raise",
        )

        result["retention_cost"] = result[
            retention_cost_column
        ]

    result["expected_value_at_risk"] = (
        result[churn_probability_column]
        * result[customer_value_column]
    )

    result["net_value_at_risk"] = (
        result["expected_value_at_risk"]
        - result["retention_cost"]
    )

    return result


def rank_customers_by_economic_value(
    df: pd.DataFrame,
    value_column: str = "net_value_at_risk",
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Rank customers by economic value at risk.

    Higher economic value at risk receives a higher priority rank.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    required_columns = [
        "customer_id",
        value_column,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df["customer_id"].isna().any():
        raise ValueError(
            "Customer ID column contains missing values."
        )

    values = pd.to_numeric(
        df[value_column],
        errors="raise",
    )

    if values.isna().any():
        raise ValueError(
            f"Column '{value_column}' contains missing values."
        )

    result = df.copy()

    result["economic_priority_rank"] = (
        result[value_column]
        .rank(
            method="min",
            ascending=False,
        )
        .astype(int)
    )

    result = result.sort_values(
        by=[
            value_column,
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


def calculate_economic_value_summary(
    df: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate portfolio-level economic value metrics.

    Returns:
        total_customer_value
        total_expected_value_at_risk
        total_retention_cost
        total_net_value_at_risk
        average_expected_value_at_risk
        average_net_value_at_risk
    """

    required_columns = [
        "customer_value",
        "expected_value_at_risk",
        "retention_cost",
        "net_value_at_risk",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    if df.empty:
        return {
            "total_customer_value": 0.0,
            "total_expected_value_at_risk": 0.0,
            "total_retention_cost": 0.0,
            "total_net_value_at_risk": 0.0,
            "average_expected_value_at_risk": 0.0,
            "average_net_value_at_risk": 0.0,
        }

    return {
        "total_customer_value": float(
            df["customer_value"].sum()
        ),
        "total_expected_value_at_risk": float(
            df["expected_value_at_risk"].sum()
        ),
        "total_retention_cost": float(
            df["retention_cost"].sum()
        ),
        "total_net_value_at_risk": float(
            df["net_value_at_risk"].sum()
        ),
        "average_expected_value_at_risk": float(
            df["expected_value_at_risk"].mean()
        ),
        "average_net_value_at_risk": float(
            df["net_value_at_risk"].mean()
        ),
    }