import csv
import os

import pandas as pd


DEFAULT_DATASET_PATH = "data/training/customer_features.csv"
TARGET_COLUMN = "churn"

DATASET_COLUMNS = [
    "customer_id",
    "observation_date",
    "customer_tenure",
    "recency",
    "frequency",
    "monetary",
    "average_bill",
    "average_purchase_gap",
    "campaigns_sent",
    "campaigns_delivered",
    "campaigns_clicked",
    "campaigns_redeemed",
    "previous_redemptions",
    "delivery_rate",
    "click_rate",
    "redemption_rate",
    "campaigns_since_last_purchase",
    "future_90d_purchases",
    "churn",
]

# The current file's header omits this label although each data row contains
# 19 fields. The loader assigns the verified canonical names in memory only.
LEGACY_HEADER_WITH_MISSING_CAMPAIGNS_REDEEMED = [
    name for name in DATASET_COLUMNS if name != "campaigns_redeemed"
]

FEATURE_COLUMNS = [
    "customer_tenure",
    "recency",
    "frequency",
    "monetary",
    "average_bill",
    "average_purchase_gap",
    "campaigns_delivered",
    "campaigns_sent",
    "campaigns_clicked",
    "previous_redemptions",
    "delivery_rate",
    "click_rate",
    "redemption_rate",
    "campaigns_since_last_purchase",
]


def _validate_csv_width(path):
    """Validate row widths and accepted header without changing the source."""
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("ML dataset is empty") from None

        canonical_header = header == DATASET_COLUMNS
        known_legacy_header = (
            header == LEGACY_HEADER_WITH_MISSING_CAMPAIGNS_REDEEMED
        )
        if not canonical_header and not known_legacy_header:
            raise ValueError(
                "Unexpected ML dataset header. Expected the canonical 19-column "
                "schema or the known header missing 'campaigns_redeemed'. "
                "The source CSV was not modified."
            )

        expected_width = len(DATASET_COLUMNS)
        for row_number, row in enumerate(reader, start=2):
            if len(row) != expected_width:
                raise ValueError(
                    f"Malformed ML dataset: row {row_number} has {len(row)} "
                    f"fields; expected {expected_width}. The source CSV was "
                    "not modified."
                )

        return header


def load_ml_dataset(path=DEFAULT_DATASET_PATH):
    """Load the training data; source file is never rewritten."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"ML dataset not found: {path}")

    header = _validate_csv_width(path)
    if header == LEGACY_HEADER_WITH_MISSING_CAMPAIGNS_REDEEMED:
        df = pd.read_csv(
            path,
            header=0,
            names=DATASET_COLUMNS,
            encoding="utf-8-sig",
        )
    else:
        df = pd.read_csv(path, encoding="utf-8-sig")

    if df.empty:
        raise ValueError("ML dataset is empty")
    if TARGET_COLUMN not in df.columns:
        raise ValueError("ML dataset is missing the churn target")
    return df


def prepare_ml_dataset(df):
    """Select the existing 14 ML features and validate the churn target."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame")

    data = df.copy()
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]
    if missing_columns:
        raise ValueError(
            "Missing required columns: " + ", ".join(missing_columns)
        )

    X = data[FEATURE_COLUMNS].copy().apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    valid_rows = X.notna().all(axis=1) & y.notna()
    X = X.loc[valid_rows].reset_index(drop=True)
    y = y.loc[valid_rows].reset_index(drop=True)

    if not y.isin([0, 1]).all():
        raise ValueError("Churn target must contain only binary values 0 and 1")
    return X, y.astype(int)


def load_and_prepare_ml_dataset(path=DEFAULT_DATASET_PATH):
    return prepare_ml_dataset(load_ml_dataset(path))
