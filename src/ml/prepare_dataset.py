import os

import pandas as pd


# ---------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------
DEFAULT_DATASET_PATH = "data/training/customer_features.csv"

TARGET_COLUMN = "churn"


# ---------------------------------------------------------
# Features used for churn prediction
# ---------------------------------------------------------
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
    "campaigns_since_last_purchase",
]


# ---------------------------------------------------------
# Load ML dataset
# ---------------------------------------------------------
def load_ml_dataset(
    path=DEFAULT_DATASET_PATH,
):
    """
    Load the prepared customer feature dataset.

    Parameters
    ----------
    path : str
        Path to the customer feature CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded customer feature dataset.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"ML dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(
            "ML dataset is empty"
        )

    return df


# ---------------------------------------------------------
# Prepare ML dataset
# ---------------------------------------------------------
def prepare_ml_dataset(df):
    """
    Prepare features and target for machine learning.

    The function:
    - validates the required feature columns
    - validates the churn target
    - selects the approved ML features
    - converts feature values to numeric
    - removes rows containing invalid values
    - returns the feature matrix and target

    Parameters
    ----------
    df : pandas.DataFrame
        Customer feature dataset.

    Returns
    -------
    X : pandas.DataFrame
        Prepared ML feature matrix.

    y : pandas.Series
        Churn target.
    """

    # -----------------------------------------------------
    # Validate input
    # -----------------------------------------------------
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "df must be a pandas DataFrame"
        )

    data = df.copy()

    # -----------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------
    required_columns = (
        FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    # -----------------------------------------------------
    # Select model features
    # -----------------------------------------------------
    X = data[FEATURE_COLUMNS].copy()

    # -----------------------------------------------------
    # Select churn target
    # -----------------------------------------------------
    y = data[TARGET_COLUMN].copy()

    # -----------------------------------------------------
    # Convert features to numeric
    # -----------------------------------------------------
    X = X.apply(
        pd.to_numeric,
        errors="coerce",
    )

    # -----------------------------------------------------
    # Convert target to numeric
    # -----------------------------------------------------
    y = pd.to_numeric(
        y,
        errors="coerce",
    )

    # -----------------------------------------------------
    # Remove invalid rows
    # -----------------------------------------------------
    valid_rows = (
        X.notna().all(axis=1)
        & y.notna()
    )

    X = (
        X.loc[valid_rows]
        .reset_index(drop=True)
    )

    y = (
        y.loc[valid_rows]
        .reset_index(drop=True)
    )

    # -----------------------------------------------------
    # Convert target to integer
    # -----------------------------------------------------
    y = y.astype(int)

    return X, y


# ---------------------------------------------------------
# Load and prepare dataset
# ---------------------------------------------------------
def load_and_prepare_ml_dataset(
    path=DEFAULT_DATASET_PATH,
):
    """
    Load and prepare the ML dataset in one step.

    Parameters
    ----------
    path : str
        Path to the customer feature CSV file.

    Returns
    -------
    X : pandas.DataFrame
        Prepared ML feature matrix.

    y : pandas.Series
        Churn target.
    """

    df = load_ml_dataset(path)

    return prepare_ml_dataset(df)