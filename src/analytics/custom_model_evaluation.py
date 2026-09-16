from pathlib import Path

import joblib
import pandas as pd

from src.analytics.segmentation import assign_customer_segments
from src.features.feature_engineering import build_customer_features
from src.ml.prepare_dataset import FEATURE_COLUMNS


MODEL_PATH = Path("models/random_forest.joblib")
DEFAULT_FUTURE_WINDOW_DAYS = 90


def _parse_dates(series):
    return pd.to_datetime(series, dayfirst=True, errors="coerce")


def _add_model_aliases(features):
    result = features.copy()

    if (
        "avg_purchase_gap" not in result.columns
        and "average_purchase_gap" in result.columns
    ):
        result["avg_purchase_gap"] = result["average_purchase_gap"]

    if (
        "campaigns_received" not in result.columns
        and "campaigns_delivered" in result.columns
    ):
        result["campaigns_received"] = result["campaigns_delivered"]

    if (
        "campaign_clicks" not in result.columns
        and "campaigns_clicked" in result.columns
    ):
        result["campaign_clicks"] = result["campaigns_clicked"]

    return result


def _add_campaigns_since_last_purchase(features, transactions, campaigns, observation_date):
    result = features.copy()
    tx = transactions.copy()
    cp = campaigns.copy()

    tx["transaction_date"] = _parse_dates(tx["transaction_date"])
    cp["campaign_date"] = _parse_dates(cp["campaign_date"])

    last_purchase = (
        tx.groupby("customer_id")["transaction_date"]
        .max()
        .rename("last_purchase_date")
    )

    campaign_dates = cp.merge(
        last_purchase,
        left_on="customer_id",
        right_index=True,
        how="left",
    )

    campaign_dates = campaign_dates[
        campaign_dates["campaign_date"].notna()
        & (campaign_dates["campaign_date"] <= observation_date)
    ]

    campaigns_since_purchase = campaign_dates[
        campaign_dates["last_purchase_date"].isna()
        | (campaign_dates["campaign_date"] > campaign_dates["last_purchase_date"])
    ]

    counts = (
        campaigns_since_purchase.groupby("customer_id")
        .size()
        .rename("campaigns_since_last_purchase")
    )

    result = result.merge(
        counts,
        left_on="customer_id",
        right_index=True,
        how="left",
    )

    result["campaigns_since_last_purchase"] = (
        result["campaigns_since_last_purchase"].fillna(0).astype(int)
    )

    return result


def _build_features_at_observation(
    customers,
    transactions,
    campaigns,
    observation_date,
):
    features = build_customer_features(
        customers.copy(),
        transactions.copy(),
        campaigns.copy(),
        observation_date,
    )
    features = _add_model_aliases(features)
    features = _add_campaigns_since_last_purchase(
        features,
        transactions,
        campaigns,
        observation_date,
    )

    missing_features = [
        column for column in FEATURE_COLUMNS if column not in features.columns
    ]
    if missing_features:
        raise ValueError(
            "Custom evaluation could not produce required model features: "
            + ", ".join(missing_features)
        )

    features = features.copy()
    for column in FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    features[FEATURE_COLUMNS] = features[FEATURE_COLUMNS].fillna(0)
    return features


def _calculate_future_90d_outcomes(
    customer_ids,
    observation_date,
    transactions,
    future_window_days=DEFAULT_FUTURE_WINDOW_DAYS,
):
    if future_window_days <= 0:
        raise ValueError("future_window_days must be greater than 0.")

    if "customer_id" not in transactions.columns:
        raise ValueError("Transactions are missing required column: customer_id.")
    if "transaction_date" not in transactions.columns:
        raise ValueError(
            "Transactions are missing required column: transaction_date."
        )

    tx = transactions.copy()
    tx["transaction_date"] = _parse_dates(tx["transaction_date"])

    future_end = observation_date + pd.Timedelta(days=future_window_days)

    future_tx = tx[
        tx["transaction_date"].notna()
        & (tx["transaction_date"] > observation_date)
        & (tx["transaction_date"] <= future_end)
    ]

    future_counts = future_tx.groupby("customer_id").size()

    outcomes = pd.DataFrame({"customer_id": pd.Series(customer_ids).astype(str)})
    outcomes["future_90d_purchases"] = (
        outcomes["customer_id"]
        .map(future_counts)
        .fillna(0)
        .astype(int)
    )
    outcomes["churn"] = (outcomes["future_90d_purchases"] == 0).astype(int)

    return outcomes


def build_custom_model_evaluation(
    customers,
    transactions,
    campaigns,
    model_path=MODEL_PATH,
    future_window_days=DEFAULT_FUTURE_WINDOW_DAYS,
):
    """
    Build a leakage-safe custom-data evaluation cohort for Model Diagnostics.

    The evaluation date is the latest transaction date minus the future
    observation window. Model probabilities are generated by rebuilding the
    project's existing customer features at that historical date and applying
    the persisted Random Forest. Actual churn is independently derived from
    transactions after that date.

    Observations whose full future window cannot be observed are therefore
    excluded by construction.
    """
    if not isinstance(customers, pd.DataFrame):
        raise TypeError("customers must be a pandas DataFrame.")
    if not isinstance(transactions, pd.DataFrame):
        raise TypeError("transactions must be a pandas DataFrame.")
    if not isinstance(campaigns, pd.DataFrame):
        raise TypeError("campaigns must be a pandas DataFrame.")

    required_customer = {"customer_id"}
    required_transaction = {"customer_id", "transaction_date"}

    missing_customer = required_customer - set(customers.columns)
    if missing_customer:
        raise ValueError(
            "Customers are missing required columns: "
            + ", ".join(sorted(missing_customer))
        )

    missing_transaction = required_transaction - set(transactions.columns)
    if missing_transaction:
        raise ValueError(
            "Transactions are missing required columns: "
            + ", ".join(sorted(missing_transaction))
        )

    if future_window_days <= 0:
        raise ValueError("future_window_days must be greater than 0.")

    transaction_dates = _parse_dates(transactions["transaction_date"])
    if transaction_dates.notna().sum() == 0:
        raise ValueError("Transactions do not contain any valid transaction dates.")

    latest_transaction_date = transaction_dates.max()
    observation_date = latest_transaction_date - pd.Timedelta(
        days=future_window_days
    )

    if observation_date < transaction_dates.min():
        raise ValueError(
            "Custom data does not contain enough transaction history "
            "for a complete evaluation window."
        )

    features = _build_features_at_observation(
        customers,
        transactions,
        campaigns,
        observation_date,
    )

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    model = joblib.load(model_path)

    if not hasattr(model, "predict_proba"):
        raise ValueError("Saved model does not provide predict_proba.")

    probabilities = model.predict_proba(features[FEATURE_COLUMNS])[:, 1]

    predictions = features[
        ["customer_id"]
    ].copy()
    predictions["churn_probability"] = probabilities.astype(float)

    outcomes = _calculate_future_90d_outcomes(
        predictions["customer_id"],
        observation_date,
        transactions,
        future_window_days,
    )

    evaluation = predictions.merge(
        outcomes,
        on="customer_id",
        how="inner",
        validate="one_to_one",
    )

    segmentation_input = features[
        ["customer_id", "recency", "frequency", "monetary"]
    ].copy()

    try:
        segmentation = assign_customer_segments(segmentation_input)
    except Exception:
        segmentation = pd.DataFrame()

    if not segmentation.empty and "customer_segment" in segmentation.columns:
        segment_columns = [
            column
            for column in ["customer_id", "customer_segment"]
            if column in segmentation.columns
        ]
        evaluation = evaluation.merge(
            segmentation[segment_columns],
            on="customer_id",
            how="left",
            validate="one_to_one",
        )

    if "customer_segment" not in evaluation.columns:
        raise ValueError(
            "Custom evaluation could not produce customer_segment."
        )

    evaluation["observation_date"] = observation_date
    evaluation["churn"] = evaluation["churn"].astype(int)
    evaluation["future_90d_purchases"] = evaluation[
        "future_90d_purchases"
    ].astype(int)

    return evaluation[
        [
            "customer_id",
            "observation_date",
            "churn",
            "churn_probability",
            "customer_segment",
            "future_90d_purchases",
        ]
    ].copy()
