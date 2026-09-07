import pandas as pd

from src.analytics.data_quality import (
    validate_dataset,
    build_data_quality_report,
    get_quality_summary,
)
from src.data.schema import CUSTOMER_COLUMNS


def create_valid_customer_data():
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "gender": ["Male", "Female", "Male"],
            "age": [25, 30, 28],
            "city": ["Delhi", "Mumbai", "Kolkata"],
            "registration_date": [
                "2026-01-10",
                "2026-02-15",
                "2026-03-20",
            ],
        }
    )


def create_valid_transaction_data():
    return pd.DataFrame(
        {
            "transaction_id": ["T001", "T002", "T003"],
            "customer_id": ["C001", "C002", "C003"],
            "transaction_date": [
                "2026-04-10",
                "2026-04-15",
                "2026-04-20",
            ],
            "bill_amount": [500.0, 750.0, 1000.0],
            "outlet": ["Delhi", "Mumbai", "Kolkata"],
        }
    )


def create_valid_campaign_data():
    return pd.DataFrame(
        {
            "campaign_id": ["CMP001", "CMP002", "CMP003"],
            "customer_id": ["C001", "C002", "C003"],
            "campaign_date": [
                "2026-04-01",
                "2026-04-05",
                "2026-04-10",
            ],
            "campaign_type": [
                "Birthday",
                "Discount",
                "Loyalty",
            ],
            "reward_type": [
                "Coupon",
                "Discount",
                "Points",
            ],
            "reward_value": [500.0, 10.0, 100.0],
            "campaign_cost": [50.0, 75.0, 40.0],
            "sent": [1, 1, 1],
            "delivered": [1, 1, 1],
            "clicked": [1, 0, 1],
            "redeemed": [1, 0, 1],
            "redemption_date": [
                "2026-04-02",
                None,
                "2026-04-12",
            ],
        }
    )


def test_validate_dataset():
    df = create_valid_customer_data()

    result = validate_dataset(
        df,
        "customer",
        CUSTOMER_COLUMNS,
        "customer_id",
    )

    assert result["schema"]["valid"]
    assert result["data_types"]["valid"]
    assert result["dates"]["valid"]
    assert result["missing_values"]["valid"]
    assert result["duplicates"]["valid"]
    assert result["ids"]["valid"]


def test_build_data_quality_report():
    customers = create_valid_customer_data()
    transactions = create_valid_transaction_data()
    campaigns = create_valid_campaign_data()

    report = build_data_quality_report(
        customers,
        transactions,
        campaigns,
    )

    assert "Customers" in report
    assert "Transactions" in report
    assert "Campaigns" in report
    assert "Cross-file References" in report

    assert report["Customers"]["row_count"] == 3
    assert report["Transactions"]["row_count"] == 3
    assert report["Campaigns"]["row_count"] == 3


def test_quality_summary():
    customers = create_valid_customer_data()
    transactions = create_valid_transaction_data()
    campaigns = create_valid_campaign_data()

    report = build_data_quality_report(
        customers,
        transactions,
        campaigns,
    )

    result = get_quality_summary(report)

    # Print the detailed summary so any failing check
    # is immediately visible during development.
    print("\nQUALITY SUMMARY:")
    print(result["summary"].to_string(index=False))

    assert result["total_checks"] == 19
    assert result["failed_checks"] == 0
    assert result["passed_checks"] == 19
    assert result["overall_status"] == "VALID"

    assert len(result["summary"]) == 19
    assert set(result["summary"]["status"]) == {"PASS"}


def test_invalid_customer_data():
    df = create_valid_customer_data()

    # Convert the age column to object/string first so
    # assigning an invalid value does not create a pandas
    # incompatible-dtype warning.
    df["age"] = df["age"].astype(object)
    df.loc[1, "age"] = "invalid"

    result = validate_dataset(
        df,
        "customer",
        CUSTOMER_COLUMNS,
        "customer_id",
    )

    assert not result["data_types"]["valid"]