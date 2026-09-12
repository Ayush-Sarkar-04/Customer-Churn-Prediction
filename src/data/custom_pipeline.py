from pathlib import Path

import joblib
import pandas as pd

from src.analytics.retention import calculate_retention_priority
from src.analytics.segmentation import assign_customer_segments
from src.features.feature_engineering import build_customer_features
from src.ml.prepare_dataset import FEATURE_COLUMNS
from src.ml.risk import assign_risk_levels


MODEL_PATH = Path("models/random_forest.joblib")


def _get_observation_date(transactions, campaigns):
    """Return the latest date available in the uploaded data."""

    dates = []

    if "transaction_date" in transactions.columns:
        transaction_dates = pd.to_datetime(
            transactions["transaction_date"],
            dayfirst=True,
            errors="coerce",
        )

        if transaction_dates.notna().any():
            dates.append(transaction_dates.max())

    if "campaign_date" in campaigns.columns:
        campaign_dates = pd.to_datetime(
            campaigns["campaign_date"],
            dayfirst=True,
            errors="coerce",
        )

        if campaign_dates.notna().any():
            dates.append(campaign_dates.max())

    if not dates:
        raise ValueError(
            "Unable to determine observation date from uploaded data."
        )

    return max(dates)


def _prepare_custom_features(
    customers,
    transactions,
    campaigns,
):
    """Build customer-level features using the existing feature pipeline."""

    observation_date = _get_observation_date(
        transactions,
        campaigns,
    )

    # IMPORTANT:
    # build_customer_features in the existing project expects
    # positional arguments, not customers= / transactions= / campaigns=.
    features = build_customer_features(
        customers.copy(),
        transactions.copy(),
        campaigns.copy(),
        observation_date,
    )

    features = features.copy()

    # ---------------------------------------------------------
    # Align feature names with the ML training schema
    # ---------------------------------------------------------

    if (
        "avg_purchase_gap" not in features.columns
        and "average_purchase_gap" in features.columns
    ):
        features["avg_purchase_gap"] = features[
            "average_purchase_gap"
        ]

    if (
        "campaigns_received" not in features.columns
        and "campaigns_delivered" in features.columns
    ):
        features["campaigns_received"] = features[
            "campaigns_delivered"
        ]

    if (
        "campaign_clicks" not in features.columns
        and "campaigns_clicked" in features.columns
    ):
        features["campaign_clicks"] = features[
            "campaigns_clicked"
        ]

    # ---------------------------------------------------------
    # Calculate campaigns since last purchase
    # ---------------------------------------------------------

    transactions_copy = transactions.copy()
    campaigns_copy = campaigns.copy()

    transactions_copy["transaction_date"] = pd.to_datetime(
        transactions_copy["transaction_date"],
        dayfirst=True,
        errors="coerce",
    )

    campaigns_copy["campaign_date"] = pd.to_datetime(
        campaigns_copy["campaign_date"],
        dayfirst=True,
        errors="coerce",
    )

    last_purchase = (
        transactions_copy
        .groupby("customer_id")["transaction_date"]
        .max()
        .rename("last_purchase_date")
    )

    campaign_dates = campaigns_copy.merge(
        last_purchase,
        left_on="customer_id",
        right_index=True,
        how="left",
    )

    campaign_dates = campaign_dates[
        campaign_dates["campaign_date"].notna()
    ]

    campaign_dates = campaign_dates[
        campaign_dates["campaign_date"] <= observation_date
    ]

    campaigns_since_purchase = campaign_dates[
        campaign_dates["last_purchase_date"].isna()
        | (
            campaign_dates["campaign_date"]
            > campaign_dates["last_purchase_date"]
        )
    ]

    campaigns_since_last_purchase = (
        campaigns_since_purchase
        .groupby("customer_id")
        .size()
        .rename("campaigns_since_last_purchase")
    )

    features = features.merge(
        campaigns_since_last_purchase,
        left_on="customer_id",
        right_index=True,
        how="left",
    )

    features["campaigns_since_last_purchase"] = (
        features["campaigns_since_last_purchase"]
        .fillna(0)
        .astype(int)
    )

    return features, observation_date


def _assign_segments(features):
    """
    Assign the project's standard RFM customer segments.

    For datasets with fewer than 5 customers, use a deterministic
    fallback instead of quantile-based segmentation.
    """

    if len(features) < 5:
        result = features.copy()

        def fallback_segment(row):
            recency = float(row.get("recency", 0))
            frequency = float(row.get("frequency", 0))
            monetary = float(row.get("monetary", 0))

            if recency >= 180:
                return "Lost"

            if recency >= 90:
                return "Inactive"

            if recency >= 60:
                return "At Risk"

            if frequency >= 4 and monetary >= 4:
                return "Champions"

            if frequency >= 3:
                return "Potential Loyalists"

            return "Regular Customers"

        result["customer_segment"] = result.apply(
            fallback_segment,
            axis=1,
        )

        return result

    return assign_customer_segments(features)


def _unpack_custom_data(
    custom_data,
    transactions=None,
    campaigns=None,
):
    """
    Accept either:

    1. A dictionary containing:
       customers, transactions, campaigns

    2. Three separate DataFrames.
    """

    if isinstance(custom_data, dict):
        required_keys = {
            "customers",
            "transactions",
            "campaigns",
        }

        missing_keys = required_keys - set(custom_data.keys())

        if missing_keys:
            raise ValueError(
                "Custom data is missing required datasets: "
                + ", ".join(sorted(missing_keys))
            )

        customers = custom_data["customers"]
        transactions = custom_data["transactions"]
        campaigns = custom_data["campaigns"]

    else:
        customers = custom_data

        if transactions is None or campaigns is None:
            raise ValueError(
                "Provide either a custom data dictionary or "
                "customers, transactions and campaigns separately."
            )

    if not isinstance(customers, pd.DataFrame):
        raise TypeError(
            "customers must be a pandas DataFrame."
        )

    if not isinstance(transactions, pd.DataFrame):
        raise TypeError(
            "transactions must be a pandas DataFrame."
        )

    if not isinstance(campaigns, pd.DataFrame):
        raise TypeError(
            "campaigns must be a pandas DataFrame."
        )

    return (
        customers.copy(),
        transactions.copy(),
        campaigns.copy(),
    )


def build_custom_customer_analytics(
    custom_data,
    transactions=None,
    campaigns=None,
    model_path=MODEL_PATH,
):
    """
    Run the complete custom-data analytics pipeline.

    Pipeline:

        Customers
             +
        Transactions
             +
        Campaigns
             ↓
        Feature Engineering
             ↓
        Random Forest Prediction
             ↓
        Churn Probability
             ↓
        Risk Level
             ↓
        Customer Segment
             ↓
        Retention Priority
             ↓
        Expected Revenue at Risk

    Custom data produces predicted churn. A true historical churn
    label requires a future 90-day outcome window and is therefore
    not created here.
    """

    # ---------------------------------------------------------
    # 1. Get uploaded datasets
    # ---------------------------------------------------------

    customers, transactions, campaigns = _unpack_custom_data(
        custom_data,
        transactions=transactions,
        campaigns=campaigns,
    )

    # ---------------------------------------------------------
    # 2. Build customer features
    # ---------------------------------------------------------

    features, observation_date = _prepare_custom_features(
        customers,
        transactions,
        campaigns,
    )

    # ---------------------------------------------------------
    # 3. Verify model feature availability
    # ---------------------------------------------------------

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in features.columns
    ]

    if missing_features:
        raise ValueError(
            "Custom data could not produce the required model "
            "features: "
            + ", ".join(missing_features)
        )

    X = features[FEATURE_COLUMNS].copy()

    # ---------------------------------------------------------
    # 4. Convert model features to numeric values
    # ---------------------------------------------------------

    for column in FEATURE_COLUMNS:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    if X.isna().any().any():
        missing_columns = X.columns[
            X.isna().any()
        ].tolist()

        raise ValueError(
            "Model features contain missing or non-numeric "
            "values: "
            + ", ".join(missing_columns)
        )

    # ---------------------------------------------------------
    # 5. Load the trained Random Forest
    # ---------------------------------------------------------

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Saved model not found: {model_path}"
        )

    model = joblib.load(model_path)

    if not hasattr(model, "predict"):
        raise ValueError(
            "Saved model does not provide a predict method."
        )

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Saved model does not provide a predict_proba method."
        )

    # ---------------------------------------------------------
    # 6. Generate churn prediction and probability
    # ---------------------------------------------------------

    churn_predictions = model.predict(X)

    churn_probabilities = model.predict_proba(X)[:, 1]

    predictions_df = features.loc[X.index].copy()

    predictions_df["churn_prediction"] = (
        churn_predictions.astype(int)
    )

    predictions_df["churn_probability"] = (
        churn_probabilities.astype(float)
    )

    # ---------------------------------------------------------
    # 7. Assign risk level
    # ---------------------------------------------------------

    try:
        # Supports versions where assign_risk_levels accepts
        # both probability and prediction.
        risk_levels = assign_risk_levels(
            predictions_df["churn_probability"],
            predictions_df["churn_prediction"],
        )
    except TypeError:
        # Supports the current probability-only implementation.
        risk_levels = assign_risk_levels(
            predictions_df["churn_probability"]
        )

    predictions_df["risk_level"] = risk_levels

    # ---------------------------------------------------------
    # 8. Assign RFM customer segments
    # ---------------------------------------------------------

    predictions_df = _assign_segments(
        predictions_df
    )

    # ---------------------------------------------------------
    # 9. Calculate retention priority
    # ---------------------------------------------------------

    predictions_df = calculate_retention_priority(
        predictions_df
    )

    # ---------------------------------------------------------
    # 10. Calculate expected revenue at risk
    # ---------------------------------------------------------

    if "expected_revenue_at_risk" not in predictions_df.columns:
        predictions_df["expected_revenue_at_risk"] = (
            predictions_df["churn_probability"]
            * predictions_df["monetary"]
        )

    # ---------------------------------------------------------
    # 11. Add custom-data metadata
    # ---------------------------------------------------------

    predictions_df["predicted_churn"] = (
        predictions_df["churn_prediction"]
    )

    predictions_df["data_mode"] = "Custom Data"

    predictions_df["prediction_observation_date"] = (
        observation_date
    )

    return predictions_df