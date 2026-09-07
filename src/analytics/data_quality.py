import pandas as pd

from src.data.validator import (
    validate_columns,
    validate_data_types,
    validate_dates,
    validate_missing_values,
    validate_duplicates,
    validate_ids,
    validate_customer_references,
)

from src.data.schema import (
    CUSTOMER_COLUMNS,
    TRANSACTION_COLUMNS,
    CAMPAIGN_COLUMNS,
)


def validate_dataset(
    df,
    dataset_type,
    expected_columns,
    id_column
):
    """
    Run all applicable quality checks for one dataset.

    Returns
    -------
    dict
        Structured validation results for dashboard use.
    """

    checks = {}

    # 1. Schema / column validation
    checks["schema"] = validate_columns(
        df,
        expected_columns
    )

    # Stop further checks if required columns are missing.
    if not checks["schema"]["valid"]:
        checks["data_types"] = {
            "valid": False,
            "errors": [
                "Skipped because schema validation failed"
            ]
        }

        checks["dates"] = {
            "valid": False,
            "errors": [
                "Skipped because schema validation failed"
            ]
        }

        checks["missing_values"] = {
            "valid": False,
            "errors": [
                "Skipped because schema validation failed"
            ]
        }

        checks["duplicates"] = {
            "valid": False,
            "errors": [
                "Skipped because schema validation failed"
            ]
        }

        checks["ids"] = {
            "valid": False,
            "errors": [
                "Skipped because schema validation failed"
            ]
        }

        return checks

    # 2. Data-type validation
    checks["data_types"] = validate_data_types(
        df,
        dataset_type
    )

    # 3. Date validation
    checks["dates"] = validate_dates(
        df,
        dataset_type
    )

    # 4. Missing-value validation
    #
    # redemption_date is optional for campaigns because
    # non-redeemed campaigns may legitimately have no date.
    optional_columns = []

    if dataset_type == "campaign":
        optional_columns = [
            "redemption_date"
        ]

    checks["missing_values"] = validate_missing_values(
        df,
        expected_columns,
        optional_columns
    )

    # 5. Duplicate-ID validation
    checks["duplicates"] = validate_duplicates(
        df,
        id_column
    )

    # 6. ID validation
    checks["ids"] = validate_ids(
        df,
        id_column
    )

    return checks


def build_data_quality_report(
    customers_df,
    transactions_df,
    campaigns_df
):
    """
    Build a complete data-quality report for
    Customers, Transactions and Campaigns.
    """

    datasets = {
        "Customers": {
            "df": customers_df,
            "dataset_type": "customer",
            "expected_columns": CUSTOMER_COLUMNS,
            "id_column": "customer_id",
        },
        "Transactions": {
            "df": transactions_df,
            "dataset_type": "transaction",
            "expected_columns": TRANSACTION_COLUMNS,
            "id_column": "transaction_id",
        },
        "Campaigns": {
            "df": campaigns_df,
            "dataset_type": "campaign",
            "expected_columns": CAMPAIGN_COLUMNS,
            "id_column": "campaign_id",
        },
    }

    report = {}

    for dataset_name, config in datasets.items():

        report[dataset_name] = {
            "row_count": len(config["df"]),
            "column_count": len(config["df"].columns),
            "checks": validate_dataset(
                config["df"],
                config["dataset_type"],
                config["expected_columns"],
                config["id_column"],
            ),
        }

    # Cross-file customer-reference validation
    reference_result = validate_customer_references(
        customers_df,
        transactions_df,
        campaigns_df
    )

    report["Cross-file References"] = {
        "checks": {
            "customer_references": reference_result
        }
    }

    return report


def get_quality_summary(report):
    """
    Convert the detailed validation report into
    a dashboard-friendly summary.
    """

    summary_rows = []

    for dataset_name, dataset_report in report.items():

        if dataset_name == "Cross-file References":
            continue

        checks = dataset_report["checks"]

        for check_name, result in checks.items():

            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "check": check_name,
                    "status": (
                        "PASS"
                        if result["valid"]
                        else "FAIL"
                    ),
                }
            )

    # Cross-file references
    reference_result = report[
        "Cross-file References"
    ]["checks"]["customer_references"]

    summary_rows.append(
        {
            "dataset": "Cross-file References",
            "check": "Customer References",
            "status": (
                "PASS"
                if reference_result["valid"]
                else "FAIL"
            ),
        }
    )

    summary = pd.DataFrame(
        summary_rows
    )

    total_checks = len(summary)

    passed_checks = (
        summary["status"] == "PASS"
    ).sum()

    failed_checks = (
        summary["status"] == "FAIL"
    ).sum()

    overall_status = (
        "VALID"
        if failed_checks == 0
        else "REJECTED"
    )

    return {
        "summary": summary,
        "total_checks": total_checks,
        "passed_checks": int(passed_checks),
        "failed_checks": int(failed_checks),
        "overall_status": overall_status,
    }