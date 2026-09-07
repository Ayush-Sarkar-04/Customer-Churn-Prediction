import pandas as pd


def search_customers(
    customer_analytics,
    search_term="",
):
    """
    Search customers using customer_id.

    Supports:
    - Exact customer ID search
    - Partial customer ID search

    Parameters
    ----------
    customer_analytics : pandas.DataFrame
        Customer-level analytics data containing customer_id.

    search_term : str
        Customer ID or partial customer ID.

    Returns
    -------
    pandas.DataFrame
        Matching customer records.
    """

    if "customer_id" not in customer_analytics.columns:
        raise ValueError(
            "Missing required column: customer_id"
        )

    result = customer_analytics.copy()

    search_term = str(search_term).strip()

    # Empty search returns all customers.
    if not search_term:
        return result

    customer_ids = (
        result["customer_id"]
        .astype(str)
        .str.strip()
    )

    matches = customer_ids.str.contains(
        search_term,
        case=False,
        na=False,
        regex=False,
    )

    return result.loc[matches].copy()


def get_customer_profile(
    customer_analytics,
    customer_id,
):
    """
    Retrieve a single customer's profile.

    Parameters
    ----------
    customer_analytics : pandas.DataFrame
        Customer-level analytics data.

    customer_id : str
        Exact customer ID.

    Returns
    -------
    pandas.Series
        Customer profile.

    Raises
    ------
    ValueError
        If customer_id is missing or not found.
    """

    if "customer_id" not in customer_analytics.columns:
        raise ValueError(
            "Missing required column: customer_id"
        )

    customer_id = str(customer_id).strip()

    if not customer_id:
        raise ValueError(
            "Customer ID cannot be empty"
        )

    customer_ids = (
        customer_analytics["customer_id"]
        .astype(str)
        .str.strip()
    )

    matches = customer_analytics.loc[
        customer_ids == customer_id
    ]

    if matches.empty:
        raise ValueError(
            f"Customer ID '{customer_id}' was not found"
        )

    return matches.iloc[0]