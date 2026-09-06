import pandas as pd

from src.analytics.segmentation import assign_customer_segments
from src.ml.predict import predict_customers


def build_customer_analytics(df, model_path=None):
    """
    Build a combined customer analytics table containing:

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
    segmentation = assign_customer_segments(
        data[
            [
                "customer_id",
                "recency",
                "frequency",
                "monetary",
            ]
        ]
    ).reset_index(drop=True)

    segmentation["_observation_id"] = data["_observation_id"]

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

    # We already have customer_id from the segmentation dataframe.
    # Remove it from predictions to avoid customer_id_x/customer_id_y.
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

    missing_columns = required_columns - set(customer_analytics.columns)

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