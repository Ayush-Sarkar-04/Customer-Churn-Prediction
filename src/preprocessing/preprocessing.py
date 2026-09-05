import pandas as pd


def clean_column_names(df):
    """
    Standardize column names by removing leading/trailing
    spaces and converting them to lowercase.
    """

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
    )

    return df

def convert_dates(df, date_columns):
    """
    Convert specified columns to datetime format.
    Invalid dates are converted to NaT.
    """

    df = df.copy()

    for column in date_columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce"
        )

    return df

def convert_numeric_columns(df, numeric_columns):
    """
    Convert specified columns to numeric format.
    Invalid numeric values are converted to NaN.
    """

    df = df.copy()

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    return df

def check_missing_values(df):
    """
    Return the number of missing values in each column.
    """

    missing_values = df.isna().sum()

    return missing_values

def remove_empty_rows(df):
    """
    Remove rows where all columns contain missing values.
    """

    df = df.copy()

    df = df.dropna(
        how="all"
    ).reset_index(drop=True)

    return df