from pathlib import Path

import pandas as pd


CAMPAIGN_OUTCOME_COLUMNS = [
    "campaign_id",
    "customer_id",
    "campaign_type",
    "campaign_date",
    "campaign_cost",
    "delivered",
    "clicked",
    "redeemed",
    "purchase_7d",
    "revenue_7d",
    "purchase_14d",
    "revenue_14d",
    "purchase_30d",
    "revenue_30d",
    "days_to_next_purchase",
    "next_purchase_amount",
]

EXPERIMENT_COLUMNS = [
    "experiment_id",
    "customer_id",
    "campaign_type",
    "experiment_date",
    "eligible",
    "treatment_group",
    "campaign_exposed",
    "baseline_30d_purchase_probability",
    "synthetic_incremental_lift_assumption",
    "purchase_7d",
    "revenue_7d",
    "purchase_14d",
    "revenue_14d",
    "purchase_30d",
    "revenue_30d",
    "synthetic_campaign_cost",
]

VALID_CAMPAIGN_TYPES = {
    "Birthday",
    "Discount",
    "Festival",
    "Loyalty",
    "New Product",
    "Win-back",
}

VALID_TREATMENT_GROUPS = {
    "treatment",
    "control",
}


def _validate_required_columns(
    df: pd.DataFrame,
    required_columns: list[str],
    dataset_name: str,
) -> None:
    """Validate that all required columns are present."""

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing required columns: {missing}"
        )


def _validate_no_duplicate_ids(
    df: pd.DataFrame,
    column: str,
    dataset_name: str,
) -> None:
    """Validate that a column contains unique values."""

    if df[column].duplicated().any():
        raise ValueError(
            f"{dataset_name} contains duplicate values in '{column}'."
        )


def _validate_non_negative(
    df: pd.DataFrame,
    columns: list[str],
    dataset_name: str,
) -> None:
    """Validate that numeric columns do not contain negative values."""

    for column in columns:
        if (df[column] < 0).any():
            raise ValueError(
                f"{dataset_name} contains negative values in '{column}'."
            )


def _validate_purchase_windows(
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """
    Validate purchase and revenue windows.

    7-day values must be <= 14-day values,
    and 14-day values must be <= 30-day values.
    """

    purchase_columns = [
        "purchase_7d",
        "purchase_14d",
        "purchase_30d",
    ]

    revenue_columns = [
        "revenue_7d",
        "revenue_14d",
        "revenue_30d",
    ]

    for column in purchase_columns:
        if not df[column].isin([0, 1]).all():
            raise ValueError(
                f"{dataset_name} contains values other than 0/1 "
                f"in '{column}'."
            )

    invalid_purchase_order = (
        (df["purchase_7d"] > df["purchase_14d"])
        | (df["purchase_14d"] > df["purchase_30d"])
    )

    if invalid_purchase_order.any():
        raise ValueError(
            f"{dataset_name} contains invalid purchase-window ordering."
        )

    invalid_revenue_order = (
        (df["revenue_7d"] > df["revenue_14d"])
        | (df["revenue_14d"] > df["revenue_30d"])
    )

    if invalid_revenue_order.any():
        raise ValueError(
            f"{dataset_name} contains invalid revenue-window ordering."
        )

    for purchase_column, revenue_column in zip(
        purchase_columns,
        revenue_columns,
    ):
        invalid_zero_purchase = (
            (df[purchase_column] == 0)
            & (df[revenue_column] != 0)
        )

        if invalid_zero_purchase.any():
            raise ValueError(
                f"{dataset_name} has revenue without purchase "
                f"in '{purchase_column}'."
            )


def _prepare_campaign_outcomes(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert campaign outcome columns to appropriate data types."""

    df = df.copy()

    df["campaign_date"] = pd.to_datetime(
        df["campaign_date"],
        errors="raise",
    )

    numeric_columns = [
        "campaign_cost",
        "delivered",
        "clicked",
        "redeemed",
        "purchase_7d",
        "revenue_7d",
        "purchase_14d",
        "revenue_14d",
        "purchase_30d",
        "revenue_30d",
        "days_to_next_purchase",
        "next_purchase_amount",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    return df


def _prepare_experiment(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert synthetic experiment columns to appropriate data types."""

    df = df.copy()

    df["experiment_date"] = pd.to_datetime(
        df["experiment_date"],
        errors="raise",
    )

    numeric_columns = [
        "eligible",
        "campaign_exposed",
        "baseline_30d_purchase_probability",
        "synthetic_incremental_lift_assumption",
        "purchase_7d",
        "revenue_7d",
        "purchase_14d",
        "revenue_14d",
        "purchase_30d",
        "revenue_30d",
        "synthetic_campaign_cost",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    return df


def validate_campaign_outcomes(
    df: pd.DataFrame,
) -> None:
    """
    Validate the historical campaign outcomes dataset.

    This dataset is observational and contains one row per
    historical campaign/customer exposure.
    """

    dataset_name = "campaign_outcomes"

    _validate_required_columns(
        df,
        CAMPAIGN_OUTCOME_COLUMNS,
        dataset_name,
    )

    _validate_no_duplicate_ids(
        df,
        "campaign_id",
        dataset_name,
    )

    if df["customer_id"].isna().any():
        raise ValueError(
            f"{dataset_name} contains missing customer IDs."
        )

    if not df["campaign_type"].isin(
        VALID_CAMPAIGN_TYPES
    ).all():
        raise ValueError(
            f"{dataset_name} contains invalid campaign types."
        )

    _validate_non_negative(
        df,
        [
            "campaign_cost",
            "delivered",
            "clicked",
            "redeemed",
            "revenue_7d",
            "revenue_14d",
            "revenue_30d",
            "days_to_next_purchase",
            "next_purchase_amount",
        ],
        dataset_name,
    )

    _validate_purchase_windows(
        df,
        dataset_name,
    )

    if (df["clicked"] > df["delivered"]).any():
        raise ValueError(
            f"{dataset_name} contains clicks greater than deliveries."
        )

    if (df["redeemed"] > df["delivered"]).any():
        raise ValueError(
            f"{dataset_name} contains redemptions greater than deliveries."
        )

    # When there is no next purchase within the observation window,
    # both fields should be missing.
    #
    # When a next purchase exists, both fields should be populated.
    days_missing = df["days_to_next_purchase"].isna()
    amount_missing = df["next_purchase_amount"].isna()

    if (days_missing != amount_missing).any():
        raise ValueError(
            f"{dataset_name} contains inconsistent next-purchase fields."
        )


def validate_synthetic_experiment(
    df: pd.DataFrame,
) -> None:
    """
    Validate the synthetic treatment/control experiment dataset.

    The experimental unit is customer_id + campaign_type.
    experiment_id identifies the experiment/campaign type and is
    therefore intentionally allowed to repeat across customers.
    """

    dataset_name = "synthetic_campaign_experiment"

    _validate_required_columns(
        df,
        EXPERIMENT_COLUMNS,
        dataset_name,
    )

    # Each customer can appear once for each campaign type.
    if df[
        ["customer_id", "campaign_type"]
    ].duplicated().any():
        raise ValueError(
            f"{dataset_name} contains duplicate "
            "customer/campaign-type combinations."
        )

    if not df["campaign_type"].isin(
        VALID_CAMPAIGN_TYPES
    ).all():
        raise ValueError(
            f"{dataset_name} contains invalid campaign types."
        )

    if not df["treatment_group"].isin(
        VALID_TREATMENT_GROUPS
    ).all():
        raise ValueError(
            f"{dataset_name} contains invalid treatment groups."
        )

    if not df["eligible"].eq(1).all():
        raise ValueError(
            f"{dataset_name} contains ineligible experiment rows."
        )

    expected_exposure = (
        df["treatment_group"]
        .eq("treatment")
        .astype(int)
    )

    if not df["campaign_exposed"].equals(
        expected_exposure
    ):
        raise ValueError(
            f"{dataset_name} has inconsistent campaign exposure."
        )

    control_rows = df["treatment_group"].eq("control")

    if not df.loc[
        control_rows,
        "synthetic_campaign_cost",
    ].eq(0).all():
        raise ValueError(
            f"{dataset_name} has campaign costs assigned "
            "to control rows."
        )

    probability = df[
        "baseline_30d_purchase_probability"
    ]

    if (
        (probability < 0)
        | (probability > 1)
    ).any():
        raise ValueError(
            f"{dataset_name} contains invalid baseline "
            "purchase probabilities."
        )

    _validate_non_negative(
        df,
        [
            "revenue_7d",
            "revenue_14d",
            "revenue_30d",
            "synthetic_campaign_cost",
        ],
        dataset_name,
    )

    _validate_purchase_windows(
        df,
        dataset_name,
    )


def load_analytics_data(
    base_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and validate both Level 2 analytical datasets.

    Parameters
    ----------
    base_path:
        Path to the data/analytics directory.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        campaign_outcomes,
        synthetic_campaign_experiment
    """

    base_path = Path(base_path)

    campaign_outcomes_path = (
        base_path / "campaign_outcomes.csv"
    )

    experiment_path = (
        base_path / "synthetic_campaign_experiment.csv"
    )

    if not campaign_outcomes_path.exists():
        raise FileNotFoundError(
            "Campaign outcomes file not found: "
            f"{campaign_outcomes_path}"
        )

    if not experiment_path.exists():
        raise FileNotFoundError(
            "Synthetic experiment file not found: "
            f"{experiment_path}"
        )

    campaign_outcomes = pd.read_csv(
        campaign_outcomes_path
    )

    synthetic_experiment = pd.read_csv(
        experiment_path
    )

    campaign_outcomes = _prepare_campaign_outcomes(
        campaign_outcomes
    )

    synthetic_experiment = _prepare_experiment(
        synthetic_experiment
    )

    validate_campaign_outcomes(
        campaign_outcomes
    )

    validate_synthetic_experiment(
        synthetic_experiment
    )

    return (
        campaign_outcomes,
        synthetic_experiment,
    )