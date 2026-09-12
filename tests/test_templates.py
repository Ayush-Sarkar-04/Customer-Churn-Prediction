import io

import pandas as pd
import pytest

from src.data.templates import (
    CAMPAIGNS_COLUMNS,
    CUSTOMERS_COLUMNS,
    TRANSACTIONS_COLUMNS,
    create_blank_template,
    get_template_columns,
    get_template_filename,
    template_csv_bytes,
    template_csv_text,
)


# ---------------------------------------------------------------------------
# Expected schemas
# ---------------------------------------------------------------------------

EXPECTED_SCHEMAS = {
    "customers": [
        "customer_id",
        "gender",
        "age",
        "city",
        "registration_date",
    ],
    "transactions": [
        "transaction_id",
        "customer_id",
        "transaction_date",
        "bill_amount",
        "outlet",
    ],
    "campaigns": [
        "campaign_id",
        "customer_id",
        "campaign_date",
        "campaign_type",
        "reward_type",
        "reward_value",
        "campaign_cost",
        "sent",
        "delivered",
        "clicked",
        "redeemed",
        "redemption_date",
    ],
}


# ---------------------------------------------------------------------------
# Schema-definition tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset_name", ["customers", "transactions", "campaigns"])
def test_template_columns_match_expected_schema(dataset_name):
    """
    Each template must contain exactly the project's required columns
    and preserve their order.
    """
    assert get_template_columns(dataset_name) == EXPECTED_SCHEMAS[dataset_name]


def test_customers_columns_constant():
    assert CUSTOMERS_COLUMNS == EXPECTED_SCHEMAS["customers"]


def test_transactions_columns_constant():
    assert TRANSACTIONS_COLUMNS == EXPECTED_SCHEMAS["transactions"]


def test_campaigns_columns_constant():
    assert CAMPAIGNS_COLUMNS == EXPECTED_SCHEMAS["campaigns"]


# ---------------------------------------------------------------------------
# Blank DataFrame tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset_name", ["customers", "transactions", "campaigns"])
def test_blank_template_has_zero_rows(dataset_name):
    template = create_blank_template(dataset_name)

    assert len(template) == 0


@pytest.mark.parametrize("dataset_name", ["customers", "transactions", "campaigns"])
def test_blank_template_has_exact_columns(dataset_name):
    template = create_blank_template(dataset_name)

    assert list(template.columns) == EXPECTED_SCHEMAS[dataset_name]


@pytest.mark.parametrize("dataset_name", ["customers", "transactions", "campaigns"])
def test_blank_template_contains_no_unexpected_columns(dataset_name):
    template = create_blank_template(dataset_name)

    actual_columns = set(template.columns)
    expected_columns = set(EXPECTED_SCHEMAS[dataset_name])

    assert actual_columns == expected_columns


# ---------------------------------------------------------------------------
# CSV generation tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dataset_name", ["customers", "transactions", "campaigns"])
def test_template_csv_text_contains_headers_only(dataset_name):
    csv_text = template_csv_text(dataset_name)

    lines = csv_text.strip().splitlines()

    assert len(lines) == 1
    assert lines[0] == ",".join(EXPECTED_SCHEMAS[dataset_name])


@pytest.mark.parametrize("dataset_name", ["customers", "transactions", "campaigns"])
def test_template_csv_can_be_read_by_pandas(dataset_name):
    csv_bytes = template_csv_bytes(dataset_name)

    dataframe = pd.read_csv(io.BytesIO(csv_bytes))

    assert list(dataframe.columns) == EXPECTED_SCHEMAS[dataset_name]
    assert len(dataframe) == 0


@pytest.mark.parametrize("dataset_name", ["customers", "transactions", "campaigns"])
def test_template_csv_bytes_are_utf8(dataset_name):
    csv_bytes = template_csv_bytes(dataset_name)

    decoded = csv_bytes.decode("utf-8")

    assert decoded == template_csv_text(dataset_name)


# ---------------------------------------------------------------------------
# Filename tests
# ---------------------------------------------------------------------------

def test_template_filenames():
    assert get_template_filename("customers") == "customers_template.csv"
    assert get_template_filename("transactions") == "transactions_template.csv"
    assert get_template_filename("campaigns") == "campaigns_template.csv"


# ---------------------------------------------------------------------------
# Invalid dataset tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "function_name",
    [
        "get_template_columns",
        "create_blank_template",
        "template_csv_text",
        "template_csv_bytes",
        "get_template_filename",
    ],
)
def test_invalid_dataset_name_is_rejected(function_name):
    from src.data import templates

    function = getattr(templates, function_name)

    with pytest.raises(ValueError):
        function("invalid_dataset")