from __future__ import annotations

import pandas as pd


def calculate_retention_cost(
    df: pd.DataFrame,
    campaign_cost_column: str = "campaign_cost",
    incentive_cost_column: str | None = None,
    other_cost_column: str | None = None,
) -> pd.DataFrame:
    """
    Calculate total retention cost for each campaign/customer record.

    Total Retention Cost =
        Campaign Cost
        + Incentive Cost
        + Other Retention Cost

    Parameters
    ----------
    df:
        Input DataFrame containing campaign-level records.

    campaign_cost_column:
        Column containing the base campaign cost.

    incentive_cost_column:
        Optional column containing incentive/reward cost.

    other_cost_column:
        Optional column containing additional retention cost.

    Returns
    -------
    pd.DataFrame
        Copy of the input DataFrame with:
        - incentive_cost
        - other_retention_cost
        - retention_cost
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if df.empty:
        result = df.copy()
        result["incentive_cost"] = 0.0
        result["other_retention_cost"] = 0.0
        result["retention_cost"] = 0.0
        return result

    required_columns = [campaign_cost_column]

    if incentive_cost_column is not None:
        required_columns.append(incentive_cost_column)

    if other_cost_column is not None:
        required_columns.append(other_cost_column)

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required cost columns: {missing_columns}"
        )

    result = df.copy()

    for column in required_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="raise",
        )

        if result[column].isna().any():
            raise ValueError(
                f"Cost column '{column}' contains missing values."
            )

        if (result[column] < 0).any():
            raise ValueError(
                f"Cost column '{column}' contains negative values."
            )

    if incentive_cost_column is None:
        result["incentive_cost"] = 0.0
    else:
        result["incentive_cost"] = result[
            incentive_cost_column
        ]

    if other_cost_column is None:
        result["other_retention_cost"] = 0.0
    else:
        result["other_retention_cost"] = result[
            other_cost_column
        ]

    result["retention_cost"] = (
        result[campaign_cost_column]
        + result["incentive_cost"]
        + result["other_retention_cost"]
    )

    return result


def calculate_total_retention_cost(
    df: pd.DataFrame,
    cost_column: str = "retention_cost",
) -> float:
    """
    Calculate total retention cost across all records.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if cost_column not in df.columns:
        raise ValueError(
            f"Column '{cost_column}' not found in DataFrame."
        )

    values = pd.to_numeric(
        df[cost_column],
        errors="raise",
    )

    if values.isna().any():
        raise ValueError(
            f"Column '{cost_column}' contains missing values."
        )

    if (values < 0).any():
        raise ValueError(
            f"Column '{cost_column}' contains negative values."
        )

    return float(values.sum())


def calculate_average_retention_cost(
    df: pd.DataFrame,
    cost_column: str = "retention_cost",
) -> float:
    """
    Calculate average retention cost per record.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    if cost_column not in df.columns:
        raise ValueError(
            f"Column '{cost_column}' not found in DataFrame."
        )

    values = pd.to_numeric(
        df[cost_column],
        errors="raise",
    )

    if values.isna().any():
        raise ValueError(
            f"Column '{cost_column}' contains missing values."
        )

    if (values < 0).any():
        raise ValueError(
            f"Column '{cost_column}' contains negative values."
        )

    if len(values) == 0:
        return 0.0

    return float(values.mean())


def aggregate_retention_cost_by_campaign_type(
    df: pd.DataFrame,
    campaign_type_column: str = "campaign_type",
    cost_column: str = "retention_cost",
) -> pd.DataFrame:
    """
    Aggregate retention cost by campaign type.

    Returns one row per campaign type with:
        - campaign_type
        - campaign_count
        - total_retention_cost
        - average_retention_cost
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required_columns = [
        campaign_type_column,
        cost_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[campaign_type_column].isna().any():
        raise ValueError(
            f"Column '{campaign_type_column}' contains missing values."
        )

    values = pd.to_numeric(
        df[cost_column],
        errors="raise",
    )

    if values.isna().any():
        raise ValueError(
            f"Column '{cost_column}' contains missing values."
        )

    if (values < 0).any():
        raise ValueError(
            f"Column '{cost_column}' contains negative values."
        )

    working = df[
        [campaign_type_column, cost_column]
    ].copy()

    result = (
        working
        .groupby(
            campaign_type_column,
            as_index=False,
        )
        .agg(
            campaign_count=(
                campaign_type_column,
                "size",
            ),
            total_retention_cost=(
                cost_column,
                "sum",
            ),
            average_retention_cost=(
                cost_column,
                "mean",
            ),
        )
    )

    return result


def calculate_customer_retention_cost(
    df: pd.DataFrame,
    customer_id_column: str = "customer_id",
    cost_column: str = "retention_cost",
) -> pd.DataFrame:
    """
    Aggregate retention cost at customer level.

    Returns one row per customer.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    required_columns = [
        customer_id_column,
        cost_column,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df[customer_id_column].isna().any():
        raise ValueError(
            f"Column '{customer_id_column}' contains missing values."
        )

    values = pd.to_numeric(
        df[cost_column],
        errors="raise",
    )

    if values.isna().any():
        raise ValueError(
            f"Column '{cost_column}' contains missing values."
        )

    if (values < 0).any():
        raise ValueError(
            f"Column '{cost_column}' contains negative values."
        )

    working = df[
        [customer_id_column, cost_column]
    ].copy()

    result = (
        working
        .groupby(
            customer_id_column,
            as_index=False,
        )
        .agg(
            campaign_count=(
                customer_id_column,
                "size",
            ),
            total_retention_cost=(
                cost_column,
                "sum",
            ),
            average_retention_cost=(
                cost_column,
                "mean",
            ),
        )
    )

    return result