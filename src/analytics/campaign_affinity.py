import pandas as pd


REQUIRED_COLUMNS = {
    "customer_id",
    "campaign_type",
    "sent",
    "delivered",
    "clicked",
}


def _validate_columns(df):
    """
    Validate that the campaign DataFrame contains
    all columns required for campaign affinity analysis.
    """

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )


def _prepare_campaign_data(campaigns):
    """
    Prepare campaign data for affinity analysis.

    Converts engagement fields to numeric values and
    validates the campaign funnel relationships.
    """

    _validate_columns(campaigns)

    result = campaigns.copy()

    numeric_columns = [
        "sent",
        "delivered",
        "clicked",
    ]

    for column in numeric_columns:
        result[column] = pd.to_numeric(
            result[column],
            errors="coerce",
        )

    if result[numeric_columns].isna().any().any():
        raise ValueError(
            "Campaign engagement fields must contain valid numeric values."
        )

    if (result["sent"] < 0).any():
        raise ValueError("sent cannot contain negative values.")

    if (result["delivered"] < 0).any():
        raise ValueError("delivered cannot contain negative values.")

    if (result["clicked"] < 0).any():
        raise ValueError("clicked cannot contain negative values.")

    if (result["delivered"] > result["sent"]).any():
        raise ValueError(
            "delivered cannot be greater than sent."
        )

    if (result["clicked"] > result["delivered"]).any():
        raise ValueError(
            "clicked cannot be greater than delivered."
        )

    return result


def calculate_campaign_affinity(campaigns):
    """
    Calculate customer-level campaign affinity.

    For each customer and campaign type, calculate:

    - campaigns_sent
    - campaigns_delivered
    - campaigns_clicked
    - campaigns_not_clicked
    - click_rate

    Click rate is calculated as:

        clicked / delivered

    Undelivered campaigns are not treated as opportunities
    for interaction.

    Returns
    -------
    pandas.DataFrame
        One row per customer and campaign type.
    """

    result = _prepare_campaign_data(campaigns)

    affinity = (
        result
        .groupby(
            [
                "customer_id",
                "campaign_type",
            ],
            as_index=False,
        )
        .agg(
            campaigns_sent=("sent", "sum"),
            campaigns_delivered=("delivered", "sum"),
            campaigns_clicked=("clicked", "sum"),
        )
    )

    affinity["campaigns_not_clicked"] = (
        affinity["campaigns_delivered"]
        - affinity["campaigns_clicked"]
    )

    affinity["click_rate"] = (
        affinity["campaigns_clicked"]
        / affinity["campaigns_delivered"]
    )

    affinity.loc[
        affinity["campaigns_delivered"] == 0,
        "click_rate",
    ] = pd.NA

    affinity = affinity.sort_values(
        [
            "customer_id",
            "click_rate",
            "campaigns_delivered",
        ],
        ascending=[
            True,
            False,
            False,
        ],
        na_position="last",
    ).reset_index(drop=True)

    return affinity


def get_customer_campaign_affinity(
    campaigns,
    customer_id,
):
    """
    Return campaign affinity for one customer.

    Parameters
    ----------
    campaigns : pandas.DataFrame
        Raw campaign dataset.

    customer_id : str
        Customer identifier.

    Returns
    -------
    pandas.DataFrame
        Campaign affinity by campaign type for the
        requested customer.
    """

    if customer_id is None:
        raise ValueError("customer_id cannot be None.")

    customer_id = str(customer_id).strip()

    if not customer_id:
        raise ValueError("customer_id cannot be empty.")

    affinity = calculate_campaign_affinity(campaigns)

    result = affinity[
        affinity["customer_id"].astype(str).str.upper()
        == customer_id.upper()
    ].copy()

    return result.reset_index(drop=True)


def calculate_campaign_type_affinity(campaigns):
    """
    Calculate overall campaign affinity by campaign type.

    This provides the campaign-type benchmark used by
    the Campaign Affinity dashboard.

    Returns
    -------
    pandas.DataFrame
        One row per campaign type.
    """

    result = _prepare_campaign_data(campaigns)

    affinity = (
        result
        .groupby(
            "campaign_type",
            as_index=False,
        )
        .agg(
            campaigns_sent=("sent", "sum"),
            campaigns_delivered=("delivered", "sum"),
            campaigns_clicked=("clicked", "sum"),
        )
    )

    affinity["campaigns_not_clicked"] = (
        affinity["campaigns_delivered"]
        - affinity["campaigns_clicked"]
    )

    affinity["click_rate"] = (
        affinity["campaigns_clicked"]
        / affinity["campaigns_delivered"]
    )

    affinity.loc[
        affinity["campaigns_delivered"] == 0,
        "click_rate",
    ] = pd.NA

    affinity = affinity.sort_values(
        [
            "click_rate",
            "campaigns_delivered",
        ],
        ascending=[
            False,
            False,
        ],
        na_position="last",
    ).reset_index(drop=True)

    return affinity


def search_campaign_customers(
    campaigns,
    search_term,
):
    """
    Search customer IDs using a case-insensitive partial match.

    Example:
        'C00' -> C0001, C0002, C0003, ...

    Returns
    -------
    list
        Sorted matching customer IDs.
    """

    if search_term is None:
        return []

    search_term = str(search_term).strip().upper()

    if not search_term:
        return []

    _validate_columns(campaigns)

    customer_ids = (
        campaigns["customer_id"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
    )

    matches = customer_ids[
        customer_ids.str.upper().str.contains(
            search_term,
            regex=False,
        )
    ]

    return sorted(matches.tolist())

def get_best_campaign_type(customer_affinity):
    """
    Identify the campaign type with the strongest observed
    click response for a customer.

    Best campaign type is determined by:
        1. Highest click rate
        2. Highest number of delivered campaigns
        3. Highest number of clicks

    Click rate is calculated from delivered campaigns only.

    Campaign types with zero delivered campaigns are excluded.

    If the customer has delivered campaigns but no clicks,
    no demonstrated campaign preference is returned.
    """

    required_columns = {
        "campaign_type",
        "campaigns_delivered",
        "campaigns_clicked",
        "click_rate",
    }

    missing_columns = required_columns - set(customer_affinity.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    result = customer_affinity.copy()

    result["campaigns_delivered"] = pd.to_numeric(
        result["campaigns_delivered"],
        errors="coerce",
    )

    result["campaigns_clicked"] = pd.to_numeric(
        result["campaigns_clicked"],
        errors="coerce",
    )

    result["click_rate"] = pd.to_numeric(
        result["click_rate"],
        errors="coerce",
    )

    # Only delivered campaigns create a valid click opportunity.
    result = result[
        result["campaigns_delivered"].fillna(0) > 0
    ].copy()

    if result.empty:
        return None

    # Remove invalid click-rate records.
    result = result[
        result["click_rate"].notna()
    ].copy()

    if result.empty:
        return None

    # If there are no clicks at all, there is no demonstrated
    # customer preference to report.
    if result["campaigns_clicked"].fillna(0).sum() <= 0:
        return None

    result = result.sort_values(
        [
            "click_rate",
            "campaigns_delivered",
            "campaigns_clicked",
            "campaign_type",
        ],
        ascending=[
            False,
            False,
            False,
            True,
        ],
    )

    best = result.iloc[0]

    return {
        "campaign_type": best["campaign_type"],
        "click_rate": float(best["click_rate"]),
        "campaigns_delivered": int(best["campaigns_delivered"]),
        "campaigns_clicked": int(best["campaigns_clicked"]),
        "campaigns_not_clicked": int(
            best["campaigns_delivered"]
            - best["campaigns_clicked"]
        ),
    }