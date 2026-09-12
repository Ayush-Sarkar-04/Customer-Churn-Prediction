"""
CSV template generation for custom customer churn data uploads.

The application accepts three raw datasets:
1. Customers
2. Transactions
3. Campaigns

Templates contain headers only and no sample data.
"""

from io import StringIO

import pandas as pd


# ---------------------------------------------------------------------------
# Fixed raw upload schemas
# ---------------------------------------------------------------------------

CUSTOMERS_COLUMNS = [
    "customer_id",
    "gender",
    "age",
    "city",
    "registration_date",
]

TRANSACTIONS_COLUMNS = [
    "transaction_id",
    "customer_id",
    "transaction_date",
    "bill_amount",
    "outlet",
]

CAMPAIGNS_COLUMNS = [
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
]


TEMPLATE_COLUMNS = {
    "customers": CUSTOMERS_COLUMNS,
    "transactions": TRANSACTIONS_COLUMNS,
    "campaigns": CAMPAIGNS_COLUMNS,
}


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------

def get_template_columns(dataset_name):
    """
    Return the expected columns for a raw upload dataset.

    Parameters
    ----------
    dataset_name : str
        One of:
        - customers
        - transactions
        - campaigns

    Returns
    -------
    list[str]
        Expected columns in the correct order.

    Raises
    ------
    ValueError
        If dataset_name is not supported.
    """
    if dataset_name not in TEMPLATE_COLUMNS:
        valid_names = ", ".join(TEMPLATE_COLUMNS.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Expected one of: {valid_names}"
        )

    # Return a copy so callers cannot accidentally modify
    # the global schema definition.
    return TEMPLATE_COLUMNS[dataset_name].copy()


def create_blank_template(dataset_name):
    """
    Create an empty DataFrame containing only the required headers.

    Parameters
    ----------
    dataset_name : str
        One of:
        - customers
        - transactions
        - campaigns

    Returns
    -------
    pandas.DataFrame
        Empty DataFrame with the exact required columns.
    """
    columns = get_template_columns(dataset_name)

    return pd.DataFrame(columns=columns)


def template_csv_text(dataset_name):
    """
    Generate a blank CSV template as UTF-8 text.

    Parameters
    ----------
    dataset_name : str
        One of:
        - customers
        - transactions
        - campaigns

    Returns
    -------
    str
        CSV text containing headers only.
    """
    template = create_blank_template(dataset_name)

    buffer = StringIO()
    template.to_csv(buffer, index=False)

    return buffer.getvalue()


def template_csv_bytes(dataset_name):
    """
    Generate a blank CSV template as UTF-8 encoded bytes.

    This format can be passed directly to Streamlit's
    st.download_button().
    """
    return template_csv_text(dataset_name).encode("utf-8")


def get_template_filename(dataset_name):
    """
    Return the recommended filename for a dataset template.
    """
    filenames = {
        "customers": "customers_template.csv",
        "transactions": "transactions_template.csv",
        "campaigns": "campaigns_template.csv",
    }

    if dataset_name not in filenames:
        valid_names = ", ".join(filenames.keys())
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Expected one of: {valid_names}"
        )

    return filenames[dataset_name]