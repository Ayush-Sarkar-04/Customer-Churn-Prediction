import pandas as pd


def calculate_customer_value_percentile(monetary):
    """
    Calculate each customer's relative value based on
    their monetary spend.

    Returns values between 0 and 1.
    Higher values represent higher customer value.
    """

    values = pd.to_numeric(
        monetary,
        errors="coerce"
    )

    if values.isna().all():
        return pd.Series(
            0.0,
            index=monetary.index
        )

    return values.rank(
        method="average",
        pct=True
    ).fillna(0.0)


def classify_retention_priority(score):
    """
    Classify customers based on retention priority score.

    Thresholds:
        < 25  -> Low
        < 50  -> Medium
        < 75  -> High
        >= 75 -> Critical
    """

    if pd.isna(score):
        return "Low"

    score = float(score)

    if not 0 <= score <= 100:
        raise ValueError(
            "Retention priority score must be between 0 and 100"
        )

    if score < 25:
        return "Low"

    if score < 50:
        return "Medium"

    if score < 75:
        return "High"

    return "Critical"


def calculate_retention_priority(df):
    """
    Calculate retention priority using churn probability
    and customer value.

    Required columns:
        - churn_probability
        - monetary

    Returns
    -------
    pandas.DataFrame
        Original dataframe with:
        - customer_value_percentile
        - retention_priority_score
        - retention_priority
        - expected_revenue_at_risk
    """

    required_columns = {
        "churn_probability",
        "monetary",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    result = df.copy()

    # ---------------------------------------------------------
    # Clean inputs
    # ---------------------------------------------------------

    result["churn_probability"] = pd.to_numeric(
        result["churn_probability"],
        errors="coerce"
    )

    result["monetary"] = pd.to_numeric(
        result["monetary"],
        errors="coerce"
    )

    # ---------------------------------------------------------
    # Validate churn probability
    # ---------------------------------------------------------

    valid_probability = (
        result["churn_probability"].isna()
        | result["churn_probability"].between(0, 1)
    )

    if not valid_probability.all():
        raise ValueError(
            "churn_probability must be between 0 and 1"
        )

    # ---------------------------------------------------------
    # Customer value
    # ---------------------------------------------------------

    result["customer_value_percentile"] = (
        calculate_customer_value_percentile(
            result["monetary"]
        )
    )

    # ---------------------------------------------------------
    # Retention priority
    #
    # 60% churn probability
    # 40% customer value
    # ---------------------------------------------------------

    result["retention_priority_score"] = (
        (
            result["churn_probability"] * 0.60
        )
        +
        (
            result["customer_value_percentile"] * 0.40
        )
    ) * 100

    result["retention_priority_score"] = (
        result["retention_priority_score"]
        .round(2)
    )

    # ---------------------------------------------------------
    # Priority classification
    # ---------------------------------------------------------

    result["retention_priority"] = (
        result["retention_priority_score"]
        .apply(classify_retention_priority)
    )

    # ---------------------------------------------------------
    # Expected revenue at risk
    # ---------------------------------------------------------

    result["expected_revenue_at_risk"] = (
        result["churn_probability"]
        * result["monetary"]
    ).round(2)

    return result