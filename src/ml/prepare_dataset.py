import pandas as pd


# Features used for churn prediction
FEATURE_COLUMNS = [
    "customer_tenure",
    "recency",
    "frequency",
    "monetary",
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
    "campaigns_since_last_purchase"
]

TARGET_COLUMN = "churn"


def load_ml_dataset(path):
    """
    Load the prepared customer feature dataset.
    """

    df = pd.read_csv(path)

    return df


def prepare_ml_dataset(df):
    """
    Prepare features and target for machine learning.
    """

    # Check that all required columns exist
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Select model features
    X = df[FEATURE_COLUMNS].copy()

    # Select churn target
    y = df[TARGET_COLUMN].copy()

    # Convert values to numeric
    X = X.apply(pd.to_numeric, errors="coerce")

    y = pd.to_numeric(y, errors="coerce")

    # Remove rows with invalid values
    valid_rows = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    # Convert target to integer
    y = y.astype(int)

    return X, y