import pandas as pd


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


def validate_data_types(df, dataset_type):
    errors = []

    if dataset_type == "customer":
        if not pd.api.types.is_numeric_dtype(df["age"]):
            errors.append("age must be numeric")

    elif dataset_type == "transaction":
        if not pd.api.types.is_numeric_dtype(df["bill_amount"]):
            errors.append("bill_amount must be numeric")

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
            if not pd.api.types.is_numeric_dtype(df[column]):
                errors.append(f"{column} must be numeric")

    else:
        errors.append("Unknown dataset type")

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def validate_dates(df, dataset_type):
    errors = []

    if dataset_type == "customer":
        date_columns = [
            "registration_date"
        ]

    elif dataset_type == "transaction":
        date_columns = [
            "transaction_date"
        ]

    elif dataset_type == "campaign":
        date_columns = [
            "campaign_date",
            "redemption_date"
        ]

    else:
        return {
            "valid": False,
            "errors": ["Unknown dataset type"]
        }

    for column in date_columns:

        # redemption_date is allowed to be empty
        if dataset_type == "campaign" and column == "redemption_date":
            values_to_check = df[column].dropna()
        else:
            values_to_check = df[column]

        invalid_dates = pd.to_datetime(
            values_to_check,
            errors="coerce"
        ).isna()

        if invalid_dates.any():
            errors.append(
                f"{column} contains invalid dates"
            )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def validate_missing_values(df, required_columns, optional_columns=None):
    errors = []

    if optional_columns is None:
        optional_columns = []

    for column in required_columns:

        # Skip optional columns
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


def validate_duplicates(df, id_column):
    errors = []

    duplicate_count = df[id_column].duplicated().sum()

    if duplicate_count > 0:
        errors.append(
            f"{id_column} contains {duplicate_count} duplicate values"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def validate_ids(df, id_column):
    errors = []

    if df[id_column].isna().any():
        errors.append(
            f"{id_column} contains missing IDs"
        )

    if (df[id_column].astype(str).str.strip() == "").any():
        errors.append(
            f"{id_column} contains empty IDs"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


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
        transaction_customer_ids - customer_ids
    )

    invalid_campaign_ids = (
        campaign_customer_ids - customer_ids
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


def validate_dataset(
    df,
    dataset_type,
    expected_columns,
    id_column
):
    errors = []

    # 1. Check column structure
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

        # Stop here because other validators may
        # fail if required columns are missing.
        return {
            "valid": False,
            "errors": errors
        }

    # 2. Check data types
    type_result = validate_data_types(
        df,
        dataset_type
    )

    if not type_result["valid"]:
        errors.extend(
            type_result["errors"]
        )

    # 3. Check dates
    date_result = validate_dates(
        df,
        dataset_type
    )

    if not date_result["valid"]:
        errors.extend(
            date_result["errors"]
        )

    # 4. Check missing values
    #
    # redemption_date is optional because
    # campaigns may not have been redeemed.
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

    # 5. Check duplicate IDs
    duplicate_result = validate_duplicates(
        df,
        id_column
    )

    if not duplicate_result["valid"]:
        errors.extend(
            duplicate_result["errors"]
        )

    # 6. Check empty/missing IDs
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

def validate_file_size(file_size_bytes, max_file_size_bytes):
    errors = []

    if file_size_bytes > max_file_size_bytes:
        errors.append(
            f"File size exceeds the maximum allowed size of "
            f"{max_file_size_bytes / (1024 * 1024):.0f} MB"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def validate_row_count(df, max_rows):
    errors = []

    row_count = len(df)

    if row_count > max_rows:
        errors.append(
            f"File contains {row_count} rows, "
            f"which exceeds the maximum allowed limit of {max_rows} rows"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def validate_file_count(file_count, max_files):
    errors = []

    if file_count > max_files:
        errors.append(
            f"Upload contains {file_count} files, "
            f"which exceeds the maximum allowed limit of {max_files} files"
        )

    return {
        "valid": len(errors) == 0,
        "errors": errors
    }

def validate_uploaded_datasets(
    customers_df,
    transactions_df,
    campaigns_df
):
    errors = []

    # Expected columns
    from src.data.schema import (
        CUSTOMER_COLUMNS,
        TRANSACTION_COLUMNS,
        CAMPAIGN_COLUMNS
    )

    # 1. Validate customers dataset
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

    # 2. Validate transactions dataset
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

    # 3. Validate campaigns dataset
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

    # 4. Validate customer references
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