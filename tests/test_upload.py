from io import BytesIO

import pandas as pd
import pytest

from src.data.upload import (
    REQUIRED_DATASETS,
    load_uploaded_datasets,
    upload_result_is_valid,
)


CUSTOMERS_CSV = """customer_id,gender,age,city,registration_date
C001,Male,25,Kolkata,01-01-2024
C002,Female,30,Delhi,05-01-2024
"""

TRANSACTIONS_CSV = """transaction_id,customer_id,transaction_date,bill_amount,outlet
T001,C001,10-01-2024,500.0,Outlet 1
T002,C002,12-01-2024,750.0,Outlet 2
"""

CAMPAIGNS_CSV = """campaign_id,customer_id,campaign_date,campaign_type,reward_type,reward_value,campaign_cost,sent,delivered,clicked,redeemed,redemption_date
CMP001,C001,15-01-2024,Birthday,Discount,100,20,1,1,1,1,16-01-2024
CMP002,C002,18-01-2024,Festival,Discount,150,25,1,1,0,0,
"""


def make_uploaded_files():
    return {
        "customers": BytesIO(CUSTOMERS_CSV.encode("utf-8")),
        "transactions": BytesIO(TRANSACTIONS_CSV.encode("utf-8")),
        "campaigns": BytesIO(CAMPAIGNS_CSV.encode("utf-8")),
    }


def test_required_dataset_names_are_correct():
    assert REQUIRED_DATASETS == {
        "customers",
        "transactions",
        "campaigns",
    }


def test_valid_three_file_upload_passes():
    result = load_uploaded_datasets(
        make_uploaded_files()
    )

    assert result["valid"] is True
    assert result["errors"] == []

    assert set(result["data"].keys()) == {
        "customers",
        "transactions",
        "campaigns",
    }


def test_valid_upload_returns_dataframes():
    result = load_uploaded_datasets(
        make_uploaded_files()
    )

    assert isinstance(
        result["data"]["customers"],
        pd.DataFrame,
    )

    assert isinstance(
        result["data"]["transactions"],
        pd.DataFrame,
    )

    assert isinstance(
        result["data"]["campaigns"],
        pd.DataFrame,
    )


def test_valid_upload_preserves_row_counts():
    result = load_uploaded_datasets(
        make_uploaded_files()
    )

    assert len(result["data"]["customers"]) == 2
    assert len(result["data"]["transactions"]) == 2
    assert len(result["data"]["campaigns"]) == 2


def test_missing_customers_file_is_rejected():
    files = make_uploaded_files()
    del files["customers"]

    result = load_uploaded_datasets(files)

    assert result["valid"] is False
    assert result["data"] is None
    assert any(
        "customers" in error.lower()
        for error in result["errors"]
    )


def test_missing_transactions_file_is_rejected():
    files = make_uploaded_files()
    del files["transactions"]

    result = load_uploaded_datasets(files)

    assert result["valid"] is False
    assert result["data"] is None
    assert any(
        "transactions" in error.lower()
        for error in result["errors"]
    )


def test_missing_campaigns_file_is_rejected():
    files = make_uploaded_files()
    del files["campaigns"]

    result = load_uploaded_datasets(files)

    assert result["valid"] is False
    assert result["data"] is None
    assert any(
        "campaigns" in error.lower()
        for error in result["errors"]
    )


def test_extra_dataset_is_rejected():
    files = make_uploaded_files()
    files["customer_features"] = BytesIO(
        b"customer_id,churn\nC001,0\n"
    )

    result = load_uploaded_datasets(files)

    assert result["valid"] is False
    assert result["data"] is None
    assert any(
        "unexpected" in error.lower()
        for error in result["errors"]
    )


def test_too_many_files_are_rejected():
    files = make_uploaded_files()

    files["extra_1"] = BytesIO(b"a,b\n1,2\n")
    files["extra_2"] = BytesIO(b"a,b\n1,2\n")

    result = load_uploaded_datasets(files)

    assert result["valid"] is False
    assert result["data"] is None
    assert any(
        "too many files" in error.lower()
        for error in result["errors"]
    )


def test_none_file_is_rejected():
    files = make_uploaded_files()
    files["customers"] = None

    result = load_uploaded_datasets(files)

    assert result["valid"] is False
    assert result["data"] is None
    assert any(
        "no file" in error.lower()
        for error in result["errors"]
    )


def test_oversized_file_is_rejected(monkeypatch):
    import src.data.upload as upload

    monkeypatch.setattr(
        upload,
        "MAX_FILE_SIZE_BYTES",
        10,
    )

    files = make_uploaded_files()

    result = load_uploaded_datasets(files)

    assert result["valid"] is False
    assert result["data"] is None
    assert any(
        "file size" in error.lower()
        for error in result["errors"]
    )


def test_too_many_rows_are_rejected(monkeypatch):
    import src.data.upload as upload

    monkeypatch.setattr(
        upload,
        "MAX_ROWS",
        1,
    )

    files = make_uploaded_files()

    result = load_uploaded_datasets(files)

    assert result["valid"] is False
    assert result["data"] is None
    assert any(
        "rows" in error.lower()
        for error in result["errors"]
    )


def test_malformed_csv_is_rejected():
    files = make_uploaded_files()

    files["customers"] = BytesIO(
        b"\xff\xfe\xfa\x00"
    )

    result = load_uploaded_datasets(files)

    assert result["valid"] is False
    assert result["data"] is None


def test_extra_customer_column_is_rejected():
    customers = """customer_id,gender,age,city,registration_date,extra
C001,Male,25,Kolkata,01-01-2024,test
"""

    files = make_uploaded_files()
    files["customers"] = BytesIO(
        customers.encode("utf-8")
    )

    result = load_uploaded_datasets(files)

    assert result["valid"] is False
    assert result["data"] is None


def test_invalid_cross_file_customer_reference_is_rejected():
    transactions = """transaction_id,customer_id,transaction_date,bill_amount,outlet
T001,C999,10-01-2024,500.0,Outlet 1
"""

    files = make_uploaded_files()
    files["transactions"] = BytesIO(
        transactions.encode("utf-8")
    )

    result = load_uploaded_datasets(files)

    assert result["valid"] is False
    assert result["data"] is None


def test_upload_result_helper_returns_true_for_valid_result():
    result = load_uploaded_datasets(
        make_uploaded_files()
    )

    assert upload_result_is_valid(result) is True


def test_upload_result_helper_returns_false_for_invalid_result():
    result = {
        "valid": False,
        "errors": ["Invalid data"],
        "data": None,
    }

    assert upload_result_is_valid(result) is False