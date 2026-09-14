from __future__ import annotations

import pandas as pd


REQUIRED_COLUMNS = [
    "customer_id",
    "churn_probability",
    "customer_value",
    "expected_value_at_risk",
    "retention_cost",
    "net_value_at_risk",
]


PRIORITY_TIERS = {
    "Low": (0, 25),
    "Medium": (25, 50),
    "High": (50, 75),
    "Critical": (75, 101),
}


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


def _validate_numeric_columns(
    df: pd.DataFrame,
) -> None:
    """Validate numeric economic-priority inputs."""

    for column in REQUIRED_COLUMNS[1:]:
        values = pd.to_numeric(
            df[column],
            errors="raise",
        )

        if values.isna().any():
            raise ValueError(
                f"Column '{column}' contains missing values."
            )


def _validate_probability(
    series: pd.Series,
) -> None:
    """Validate churn probability."""

    values = pd.to_numeric(
        series,
        errors="raise",
    )

    if ((values < 0) | (values > 1)).any():
        raise ValueError(
            "churn_probability must contain values between 0 and 1."
        )


def _validate_non_negative(
    series: pd.Series,
    column_name: str,
) -> None:
    """Validate non-negative economic values."""

    values = pd.to_numeric(
        series,
        errors="raise",
    )

    if (values < 0).any():
        raise ValueError(
            f"Column '{column_name}' cannot contain negative values."
        )


def calculate_economic_priority(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate an economic priority score for each customer.

    The score is based on Net Value at Risk.

    Economic Priority Score:
        percentile rank of Net Value at Risk × 100

    Higher scores indicate customers with greater economic exposure
    after accounting for retention cost.

    Priority tiers:
        Low       < 25
        Medium    25 to < 50
        High      50 to < 75
        Critical  >= 75

    This is a prioritization metric. It does not represent:
        - guaranteed revenue loss
        - guaranteed revenue saved
        - campaign ROI
        - incremental treatment effect
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    _validate_required_columns(df)

    # Handle an empty DataFrame before performing numeric
    # calculations or consistency checks.
    if df.empty:
        result = df.copy()

        result["economic_priority_score"] = pd.Series(
            index=result.index,
            dtype=float,
        )

        result["economic_priority_tier"] = pd.Series(
            index=result.index,
            dtype=str,
        )

        return result

    _validate_numeric_columns(df)

    _validate_probability(
        df["churn_probability"]
    )

    _validate_non_negative(
        df["customer_value"],
        "customer_value",
    )

    _validate_non_negative(
        df["expected_value_at_risk"],
        "expected_value_at_risk",
    )

    _validate_non_negative(
        df["retention_cost"],
        "retention_cost",
    )

    result = df.copy()

    # Recalculate Economic Value at Risk from the underlying inputs.
    calculated_evar = (
        result["churn_probability"]
        * result["customer_value"]
    )

    # Net Value at Risk is the expected economic exposure
    # after accounting for retention cost.
    calculated_net_value = (
        calculated_evar
        - result["retention_cost"]
    )

    # Validate that the supplied EVaR is internally consistent.
    if not (
        calculated_evar
        .round(10)
        .eq(
            pd.to_numeric(
                result["expected_value_at_risk"]
            ).round(10)
        )
    ).all():
        raise ValueError(
            "expected_value_at_risk is inconsistent with "
            "churn_probability × customer_value."
        )

    # Validate that the supplied Net Value at Risk is internally
    # consistent with EVaR minus retention cost.
    if not (
        calculated_net_value
        .round(10)
        .eq(
            pd.to_numeric(
                result["net_value_at_risk"]
            ).round(10)
        )
    ).all():
        raise ValueError(
            "net_value_at_risk is inconsistent with "
            "expected_value_at_risk − retention_cost."
        )

    # Percentile rank gives a relative 0–100 priority score
    # within the current customer population.
    result["economic_priority_score"] = (
        pd.to_numeric(
            result["net_value_at_risk"]
        )
        .rank(
            method="average",
            pct=True,
        )
        * 100
    )

    def assign_tier(
        score: float,
    ) -> str:
        if score < 25:
            return "Low"

        if score < 50:
            return "Medium"

        if score < 75:
            return "High"

        return "Critical"

    result["economic_priority_tier"] = (
        result["economic_priority_score"]
        .apply(assign_tier)
    )

    return result


def rank_economic_priority(
    df: pd.DataFrame,
    top_n: int | None = None,
) -> pd.DataFrame:
    """
    Rank customers from highest to lowest economic priority.

    Customers are sorted by:
        1. economic priority score
        2. net value at risk
        3. customer_id

    Parameters
    ----------
    df:
        DataFrame containing calculated economic-priority fields.

    top_n:
        Optional number of highest-priority customers to return.

    Returns
    -------
    pd.DataFrame
        Ranked DataFrame with economic_priority_rank.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    required_columns = [
        "customer_id",
        "economic_priority_score",
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

    if df["customer_id"].isna().any():
        raise ValueError(
            "customer_id contains missing values."
        )

    if top_n is not None and top_n < 1:
        raise ValueError(
            "top_n must be at least 1."
        )

    result = df.copy()

    result["economic_priority_score"] = pd.to_numeric(
        result["economic_priority_score"],
        errors="raise",
    )

    result["net_value_at_risk"] = pd.to_numeric(
        result["net_value_at_risk"],
        errors="raise",
    )

    result = result.sort_values(
        by=[
            "economic_priority_score",
            "net_value_at_risk",
            "customer_id",
        ],
        ascending=[
            False,
            False,
            True,
        ],
    ).reset_index(drop=True)

    result["economic_priority_rank"] = (
        result.index + 1
    )

    if top_n is not None:
        result = result.head(top_n)

    return result


def summarize_economic_priority(
    df: pd.DataFrame,
) -> dict:
    """
    Generate portfolio-level economic priority summary.

    Returns:
        customer_count
        critical_customers
        high_customers
        medium_customers
        low_customers
        critical_net_value_at_risk
        high_or_critical_net_value_at_risk
        total_net_value_at_risk
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    required_columns = [
        "customer_id",
        "economic_priority_score",
        "economic_priority_tier",
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
            "customer_count": 0,
            "critical_customers": 0,
            "high_customers": 0,
            "medium_customers": 0,
            "low_customers": 0,
            "critical_net_value_at_risk": 0.0,
            "high_or_critical_net_value_at_risk": 0.0,
            "total_net_value_at_risk": 0.0,
        }

    net_value_at_risk = pd.to_numeric(
        df["net_value_at_risk"],
        errors="raise",
    )

    tier_counts = (
        df["economic_priority_tier"]
        .value_counts()
    )

    critical_mask = (
        df["economic_priority_tier"]
        == "Critical"
    )

    high_or_critical_mask = (
        df["economic_priority_tier"]
        .isin(
            [
                "High",
                "Critical",
            ]
        )
    )

    return {
        "customer_count": int(
            df["customer_id"].nunique()
        ),
        "critical_customers": int(
            tier_counts.get("Critical", 0)
        ),
        "high_customers": int(
            tier_counts.get("High", 0)
        ),
        "medium_customers": int(
            tier_counts.get("Medium", 0)
        ),
        "low_customers": int(
            tier_counts.get("Low", 0)
        ),
        "critical_net_value_at_risk": float(
            net_value_at_risk.loc[
                critical_mask
            ].sum()
        ),
        "high_or_critical_net_value_at_risk": float(
            net_value_at_risk.loc[
                high_or_critical_mask
            ].sum()
        ),
        "total_net_value_at_risk": float(
            net_value_at_risk.sum()
        ),
    }