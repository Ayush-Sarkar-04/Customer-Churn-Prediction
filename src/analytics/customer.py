import pandas as pd

from src.analytics.segmentation import assign_customer_segments
from src.ml.predict import predict_customers


def build_customer_analytics(df, model_path=None):
    """
    Build a combined customer analytics table containing:

    - Observation date
    - RFM features
    - RFM scores
    - Customer segments
    - Churn prediction
    - Churn probability
    - Risk level

    The input dataset may contain multiple observation
    records for the same customer.

    Parameters
    ----------
    df : pandas.DataFrame
        Customer feature / observation dataset.

    model_path : str, optional
        Path to the saved churn prediction model.

    Returns
    -------
    pandas.DataFrame
        Combined observation-level customer analytics.
    """

    required_columns = {
        "customer_id",
        "recency",
        "frequency",
        "monetary",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Work on a copy so the original dataframe is not modified.
    data = df.reset_index(drop=True).copy()

    # Create a unique identifier for every observation.
    data["_observation_id"] = range(len(data))

    # ---------------------------------------------------------
    # Customer segmentation
    # ---------------------------------------------------------
    segmentation_columns = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
    ]

    segmentation = assign_customer_segments(
        data[segmentation_columns]
    ).reset_index(drop=True)

    segmentation["_observation_id"] = data["_observation_id"]

    # ---------------------------------------------------------
    # Preserve observation date
    # ---------------------------------------------------------
    if "observation_date" in data.columns:
        segmentation["observation_date"] = (
            data["observation_date"].values
        )

    # ---------------------------------------------------------
    # Churn prediction and risk classification
    # ---------------------------------------------------------
    if model_path is None:
        predictions = predict_customers(data)
    else:
        predictions = predict_customers(
            data,
            model_path=model_path
        )

    predictions = predictions.reset_index(drop=True)

    predictions["_observation_id"] = data["_observation_id"]

    # We already have customer_id from segmentation.
    # Remove it from predictions to avoid
    # customer_id_x / customer_id_y.
    predictions = predictions.drop(
        columns=["customer_id"]
    )

    # ---------------------------------------------------------
    # Combine segmentation + prediction results
    # ---------------------------------------------------------
    result = segmentation.merge(
        predictions,
        on="_observation_id",
        how="left",
        validate="one_to_one",
    )

    # ---------------------------------------------------------
    # Restore useful source columns
    # ---------------------------------------------------------
    source_columns = [
        "customer_tenure",
        "average_bill",
        "avg_purchase_gap",
        "campaigns_received",
        "campaigns_sent",
        "campaigns_delivered",
        "campaign_clicks",
        "previous_redemptions",
        "delivery_rate",
        "click_rate",
        "redemption_rate",
        "campaigns_since_last_purchase",
        "future_90d_purchases",
        "churn",
    ]

    for column in source_columns:
        if column in data.columns and column not in result.columns:
            result[column] = data[column].values

    # Remove temporary technical identifier.
    result = result.drop(
        columns=["_observation_id"]
    )

    return result


def calculate_segment_risk_counts(customer_analytics):
    """
    Calculate customer counts by segment and risk level.

    Parameters
    ----------
    customer_analytics : pandas.DataFrame
        Combined customer analytics containing:
        - customer_segment
        - risk_level

    Returns
    -------
    pandas.DataFrame
        Customer counts grouped by segment and risk level.
    """

    required_columns = {
        "customer_segment",
        "risk_level",
    }

    missing_columns = (
        required_columns
        - set(customer_analytics.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    result = (
        customer_analytics
        .groupby(
            ["customer_segment", "risk_level"]
        )
        .size()
        .reset_index(name="customer_count")
    )

    return result