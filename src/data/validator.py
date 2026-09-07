import pandas as pd


# =========================================================
# Date Parsing Helper
# =========================================================

def parse_project_dates(series):
    """
    Parse dates using the date formats supported by the project.

    Supported formats:
    - DD-MM-YYYY
    - YYYY-MM-DD

    Invalid values are returned as NaT.
    """

    values = series.astype("string").str.strip()

    # First try the project's primary format.
    parsed = pd.to_datetime(
        values,
        format="%d-%m-%Y",
        errors="coerce"
    )

    # Try ISO format for values that were not parsed.
    remaining = parsed.isna() & values.notna()

    if remaining.any():
        parsed.loc[remaining] = pd.to_datetime(
            values.loc[remaining],
            format="%Y-%m-%d",
            errors="coerce"
        )

    return parsed


# =========================================================
# Column Validation
# =========================================================

def validate_columns(df, expected_columns):
    actual_columns = set(df.columns)
    expected_columns = set(expected_columns)

    missing = expected_columns - actual_columns
    extra = actual_columns - expected_columns

    if missing or extra:
        return {
            "valid": False,
            "missing": sorted(missing),
            "extra": sorted(extra)
        }

    return {
        "valid": True,
        "missing": [],
        "extra": []
    }


# =========================================================
# Data Type Validation
# =========================================================

def validate_data_types(df, dataset_type):
    errors = []

    if dataset_type == "customer":

        if not pd.api.types.is_numeric_dtype(df["age"]):
            errors.append(
                "age must be numeric"
            )

    elif dataset_type == "transaction":

        if not pd.api.types.is_numeric_dtype(
            df["bill_amount"]
        ):
            errors.append(
                "bill_amount must be numeric"
            )

    elif dataset_type == "campaign":

        numeric_columns = [
            "reward_value",
            "campaign_cost",
            "sent",
            "delivered",
            "clicked",
            "redeemed"
        ]

        for column in numeric_columns:

            if not pd.api.types.is_numeric_dtype(
                df[column]
            ):
                errors.append(
                    f"{column} must be numeric"
                )

    else:

        errors.append(
            "Unknown dataset type"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# =========================================================
# Date Validation
# =========================================================

def validate_dates(df, dataset_type):
    errors = []

    # -----------------------------------------------------
    # Customers
    # -----------------------------------------------------
    if dataset_type == "customer":

        parsed_dates = parse_project_dates(
            df["registration_date"]
        )

        invalid_dates = parsed_dates.isna()

        if invalid_dates.any():

            errors.append(
                "registration_date contains invalid dates"
            )

    # -----------------------------------------------------
    # Transactions
    # -----------------------------------------------------
    elif dataset_type == "transaction":

        parsed_dates = parse_project_dates(
            df["transaction_date"]
        )

        invalid_dates = parsed_dates.isna()

        if invalid_dates.any():

            errors.append(
                "transaction_date contains invalid dates"
            )

    # -----------------------------------------------------
    # Campaigns
    # -----------------------------------------------------
    elif dataset_type == "campaign":

        campaign_dates = parse_project_dates(
            df["campaign_date"]
        )

        invalid_campaign_dates = (
            campaign_dates.isna()
        )

        if invalid_campaign_dates.any():

            errors.append(
                "campaign_date contains invalid dates"
            )

        # -------------------------------------------------
        # Redemption date validation
        #
        # redeemed = 0
        #     redemption_date may be blank
        #
        # redeemed = 1
        #     redemption_date must be valid
        # -------------------------------------------------

        redemption_dates = parse_project_dates(
            df["redemption_date"]
        )

        redeemed_rows = (
            df["redeemed"] == 1
        )

        invalid_redeemed_dates = (
            redeemed_rows
            & redemption_dates.isna()
        )

        if invalid_redeemed_dates.any():

            errors.append(
                "redemption_date contains invalid or missing "
                "dates for redeemed campaigns"
            )

        # -------------------------------------------------
        # Redemption cannot occur before campaign
        # -------------------------------------------------

        comparable_rows = (
            redeemed_rows
            & campaign_dates.notna()
            & redemption_dates.notna()
        )

        invalid_date_order = (
            comparable_rows
            & (
                redemption_dates
                < campaign_dates
            )
        )

        if invalid_date_order.any():

            errors.append(
                "redemption_date cannot be earlier than campaign_date"
            )

    # -----------------------------------------------------
    # Unknown dataset
    # -----------------------------------------------------
    else:

        return {
            "valid": False,
            "errors": [
                "Unknown dataset type"
            ]
        }

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# =========================================================
# Missing Value Validation
# =========================================================

def validate_missing_values(
    df,
    required_columns,
    optional_columns=None
):
    errors = []

    if optional_columns is None:
        optional_columns = []

    for column in required_columns:

        if column in optional_columns:
            continue

        if df[column].isna().any():

            errors.append(
                f"{column} contains missing values"
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# =========================================================
# Duplicate Validation
# =========================================================

def validate_duplicates(df, id_column):
    errors = []

    duplicate_count = (
        df[id_column]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:

        errors.append(
            f"{id_column} contains "
            f"{duplicate_count} duplicate values"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# =========================================================
# ID Validation
# =========================================================

def validate_ids(df, id_column):
    errors = []

    if df[id_column].isna().any():

        errors.append(
            f"{id_column} contains missing IDs"
        )

    if (
        df[id_column]
        .astype(str)
        .str.strip()
        .eq("")
        .any()
    ):

        errors.append(
            f"{id_column} contains empty IDs"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# =========================================================
# Cross-File Customer Reference Validation
# =========================================================

def validate_customer_references(
    customers_df,
    transactions_df,
    campaigns_df
):
    errors = []

    customer_ids = set(
        customers_df["customer_id"]
    )

    transaction_customer_ids = set(
        transactions_df["customer_id"]
    )

    campaign_customer_ids = set(
        campaigns_df["customer_id"]
    )

    invalid_transaction_ids = (
        transaction_customer_ids
        - customer_ids
    )

    invalid_campaign_ids = (
        campaign_customer_ids
        - customer_ids
    )

    if invalid_transaction_ids:

        errors.append(
            f"Transactions contain "
            f"{len(invalid_transaction_ids)} "
            "customer IDs not found in customers"
        )

    if invalid_campaign_ids:

        errors.append(
            f"Campaigns contain "
            f"{len(invalid_campaign_ids)} "
            "customer IDs not found in customers"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# =========================================================
# Complete Dataset Validation
# =========================================================

def validate_dataset(
    df,
    dataset_type,
    expected_columns,
    id_column
):
    errors = []

    # -----------------------------------------------------
    # 1. Schema / Column Structure
    # -----------------------------------------------------

    column_result = validate_columns(
        df,
        expected_columns
    )

    if not column_result["valid"]:

        errors.append({
            "type": "columns",
            "missing": column_result["missing"],
            "extra": column_result["extra"]
        })

        # Stop here because the remaining validators
        # may fail if required columns are missing.
        return {
            "valid": False,
            "errors": errors
        }

    # -----------------------------------------------------
    # 2. Data Types
    # -----------------------------------------------------

    type_result = validate_data_types(
        df,
        dataset_type
    )

    if not type_result["valid"]:

        errors.extend(
            type_result["errors"]
        )

    # -----------------------------------------------------
    # 3. Dates
    # -----------------------------------------------------

    date_result = validate_dates(
        df,
        dataset_type
    )

    if not date_result["valid"]:

        errors.extend(
            date_result["errors"]
        )

    # -----------------------------------------------------
    # 4. Missing Values
    # -----------------------------------------------------
    #
    # redemption_date is optional at the general
    # missing-value level because non-redeemed campaigns
    # may legitimately have no redemption date.
    #
    # The conditional rule for redeemed campaigns is
    # handled separately in validate_dates().
    # -----------------------------------------------------

    optional_columns = []

    if dataset_type == "campaign":

        optional_columns = [
            "redemption_date"
        ]

    missing_result = validate_missing_values(
        df,
        expected_columns,
        optional_columns
    )

    if not missing_result["valid"]:

        errors.extend(
            missing_result["errors"]
        )

    # -----------------------------------------------------
    # 5. Duplicate IDs
    # -----------------------------------------------------

    duplicate_result = validate_duplicates(
        df,
        id_column
    )

    if not duplicate_result["valid"]:

        errors.extend(
            duplicate_result["errors"]
        )

    # -----------------------------------------------------
    # 6. Empty / Missing IDs
    # -----------------------------------------------------

    id_result = validate_ids(
        df,
        id_column
    )

    if not id_result["valid"]:

        errors.extend(
            id_result["errors"]
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# =========================================================
# File Size Validation
# =========================================================

def validate_file_size(
    file_size_bytes,
    max_file_size_bytes
):
    errors = []

    if file_size_bytes > max_file_size_bytes:

        errors.append(
            "File size exceeds the maximum allowed size of "
            f"{max_file_size_bytes / (1024 * 1024):.0f} MB"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# =========================================================
# Row Count Validation
# =========================================================

def validate_row_count(df, max_rows):
    errors = []

    row_count = len(df)

    if row_count > max_rows:

        errors.append(
            f"File contains {row_count} rows, "
            f"which exceeds the maximum allowed limit "
            f"of {max_rows} rows"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# =========================================================
# File Count Validation
# =========================================================

def validate_file_count(
    file_count,
    max_files
):
    errors = []

    if file_count > max_files:

        errors.append(
            f"Upload contains {file_count} files, "
            f"which exceeds the maximum allowed limit "
            f"of {max_files} files"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# =========================================================
# Uploaded Dataset Validation
# =========================================================

def validate_uploaded_datasets(
    customers_df,
    transactions_df,
    campaigns_df
):
    errors = []

    # -----------------------------------------------------
    # Expected schemas
    # -----------------------------------------------------

    from src.data.schema import (
        CUSTOMER_COLUMNS,
        TRANSACTION_COLUMNS,
        CAMPAIGN_COLUMNS
    )

    # -----------------------------------------------------
    # 1. Customers
    # -----------------------------------------------------

    customer_result = validate_dataset(
        customers_df,
        "customer",
        CUSTOMER_COLUMNS,
        "customer_id"
    )

    if not customer_result["valid"]:

        errors.append({
            "dataset": "customers",
            "errors": customer_result["errors"]
        })

    # -----------------------------------------------------
    # 2. Transactions
    # -----------------------------------------------------

    transaction_result = validate_dataset(
        transactions_df,
        "transaction",
        TRANSACTION_COLUMNS,
        "transaction_id"
    )

    if not transaction_result["valid"]:

        errors.append({
            "dataset": "transactions",
            "errors": transaction_result["errors"]
        })

    # -----------------------------------------------------
    # 3. Campaigns
    # -----------------------------------------------------

    campaign_result = validate_dataset(
        campaigns_df,
        "campaign",
        CAMPAIGN_COLUMNS,
        "campaign_id"
    )

    if not campaign_result["valid"]:

        errors.append({
            "dataset": "campaigns",
            "errors": campaign_result["errors"]
        })

    # -----------------------------------------------------
    # 4. Cross-file References
    # -----------------------------------------------------

    reference_result = validate_customer_references(
        customers_df,
        transactions_df,
        campaigns_df
    )

    if not reference_result["valid"]:

        errors.append({
            "dataset": "cross_reference",
            "errors": reference_result["errors"]
        })

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }