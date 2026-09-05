import pandas as pd


def _parse_dates(series):
    """
    Safely parse dates that may appear in either:
    - YYYY-MM-DD
    - DD-MM-YYYY

    Invalid dates are converted to NaT.
    """

    # First try the standard YYYY-MM-DD format
    result = pd.to_datetime(
        series,
        format="%Y-%m-%d",
        errors="coerce"
    )

    # For values that were not parsed, try DD-MM-YYYY
    missing = result.isna()

    if missing.any():
        result.loc[missing] = pd.to_datetime(
            series.loc[missing],
            format="%d-%m-%Y",
            errors="coerce"
        )

    return result


def calculate_customer_tenure(df):
    """
    Calculate customer tenure in days.
    """

    result = df.copy()

    result["registration_date"] = _parse_dates(
        result["registration_date"]
    )

    result["observation_date"] = _parse_dates(
        result["observation_date"]
    )

    result["customer_tenure"] = (
        result["observation_date"]
        - result["registration_date"]
    ).dt.days

    return result


def calculate_rfm_features(transactions_df, observation_date):
    """
    Calculate customer-level RFM features.

    Recency:
        Number of days since the customer's most recent purchase.

    Frequency:
        Number of purchases made on or before the observation date.

    Monetary:
        Total amount spent on or before the observation date.

    Average Bill:
        Average amount spent per purchase.

    Average Purchase Gap:
        Average number of days between consecutive purchases.
    """

    transactions = transactions_df.copy()

    # Convert transaction dates
    transactions["transaction_date"] = _parse_dates(
        transactions["transaction_date"]
    )

    # Convert observation date
    observation_date = pd.to_datetime(observation_date)

    # Prevent future-data leakage
    transactions = transactions[
        transactions["transaction_date"] <= observation_date
    ].copy()

    # Sort transactions chronologically for each customer
    transactions = transactions.sort_values(
        ["customer_id", "transaction_date"]
    )

    # Calculate gap between consecutive purchases
    transactions["purchase_gap"] = (
        transactions
        .groupby("customer_id")["transaction_date"]
        .diff()
        .dt.days
    )

    # Calculate customer-level RFM values
    rfm = (
        transactions
        .groupby("customer_id")
        .agg(
            last_purchase_date=("transaction_date", "max"),
            frequency=("transaction_id", "count"),
            monetary=("bill_amount", "sum"),
            average_purchase_gap=("purchase_gap", "mean")
        )
        .reset_index()
    )

    # Calculate recency
    rfm["recency"] = (
        observation_date - rfm["last_purchase_date"]
    ).dt.days

    # Calculate average bill
    rfm["average_bill"] = (
        rfm["monetary"] / rfm["frequency"]
    )

    # Keep required customer-level features
    rfm = rfm[
        [
            "customer_id",
            "recency",
            "frequency",
            "monetary",
            "average_bill",
            "average_purchase_gap"
        ]
    ]

    return rfm


def calculate_campaign_features(campaigns_df, observation_date):
    """
    Calculate customer-level campaign engagement features.

    Only campaigns occurring on or before the observation date
    are included to prevent future-data leakage.
    """

    campaigns = campaigns_df.copy()

    # Convert campaign dates
    campaigns["campaign_date"] = _parse_dates(
        campaigns["campaign_date"]
    )

    # Convert observation date
    observation_date = pd.to_datetime(observation_date)

    # Keep only campaigns available at the observation date
    campaigns = campaigns[
        campaigns["campaign_date"] <= observation_date
    ].copy()

    # Calculate campaign activity for each customer
    engagement = (
        campaigns
        .groupby("customer_id")
        .agg(
            campaigns_sent=("sent", "sum"),
            campaigns_delivered=("delivered", "sum"),
            campaigns_clicked=("clicked", "sum"),
            campaigns_redeemed=("redeemed", "sum")
        )
        .reset_index()
    )

    # Calculate delivery rate
    engagement["delivery_rate"] = (
        engagement["campaigns_delivered"]
        / engagement["campaigns_sent"]
    )

    # Calculate click rate
    engagement["click_rate"] = (
        engagement["campaigns_clicked"]
        / engagement["campaigns_delivered"]
    )

    # Calculate redemption rate
    engagement["redemption_rate"] = (
        engagement["campaigns_redeemed"]
        / engagement["campaigns_delivered"]
    )

    # Replace infinite values caused by division by zero
    engagement = engagement.replace(
        [float("inf"), float("-inf")],
        0
    )

    # Replace missing rates with zero
    engagement[
        [
            "delivery_rate",
            "click_rate",
            "redemption_rate"
        ]
    ] = engagement[
        [
            "delivery_rate",
            "click_rate",
            "redemption_rate"
        ]
    ].fillna(0)

    return engagement


def calculate_previous_redemptions(campaigns_df, observation_date):
    """
    Calculate the number of campaigns actually redeemed
    by each customer on or before the observation date.
    """

    campaigns = campaigns_df.copy()

    # Convert campaign and redemption dates
    campaigns["campaign_date"] = _parse_dates(
        campaigns["campaign_date"]
    )

    campaigns["redemption_date"] = _parse_dates(
        campaigns["redemption_date"]
    )

    # Convert observation date
    observation_date = pd.to_datetime(observation_date)

    # Keep only actual redemptions that happened
    # on or before the observation date
    campaigns = campaigns[
        (campaigns["redeemed"] == 1)
        & (campaigns["redemption_date"] <= observation_date)
    ].copy()

    # Count previous redemptions per customer
    result = (
        campaigns
        .groupby("customer_id")
        .size()
        .reset_index(name="previous_redemptions")
    )

    return result


def build_customer_features(
    customers_df,
    transactions_df,
    campaigns_df,
    observation_date
):
    """
    Build a complete customer-level feature table.

    Each row represents one customer and contains:

    - customer profile information
    - customer tenure
    - RFM features
    - campaign engagement features
    - previous redemptions
    """

    # Make copies so original data is not modified
    customers = customers_df.copy()

    # Convert observation date
    observation_date = pd.to_datetime(observation_date)

    # --------------------------------------------------
    # 1. CUSTOMER TENURE
    # --------------------------------------------------

    customers["registration_date"] = _parse_dates(
        customers["registration_date"]
    )

    customers["customer_tenure"] = (
        observation_date
        - customers["registration_date"]
    ).dt.days

    customers["observation_date"] = observation_date

    # --------------------------------------------------
    # 2. RFM FEATURES
    # --------------------------------------------------

    rfm = calculate_rfm_features(
        transactions_df,
        observation_date
    )

    # --------------------------------------------------
    # 3. CAMPAIGN FEATURES
    # --------------------------------------------------

    campaign_features = calculate_campaign_features(
        campaigns_df,
        observation_date
    )

    # --------------------------------------------------
    # 4. PREVIOUS REDEMPTIONS
    # --------------------------------------------------

    previous_redemptions = calculate_previous_redemptions(
        campaigns_df,
        observation_date
    )

    # --------------------------------------------------
    # 5. MERGE ALL FEATURES
    # --------------------------------------------------

    result = customers.merge(
        rfm,
        on="customer_id",
        how="left"
    )

    result = result.merge(
        campaign_features,
        on="customer_id",
        how="left"
    )

    result = result.merge(
        previous_redemptions,
        on="customer_id",
        how="left"
    )

    # --------------------------------------------------
    # 6. FILL FEATURES FOR CUSTOMERS WITH NO ACTIVITY
    # --------------------------------------------------

    numeric_columns = [
        "recency",
        "frequency",
        "monetary",
        "average_bill",
        "average_purchase_gap",
        "campaigns_sent",
        "campaigns_delivered",
        "campaigns_clicked",
        "campaigns_redeemed",
        "delivery_rate",
        "click_rate",
        "redemption_rate",
        "previous_redemptions"
    ]

    for column in numeric_columns:
        if column in result.columns:
            result[column] = result[column].fillna(0)

    # --------------------------------------------------
    # 7. RETURN FINAL FEATURE TABLE
    # --------------------------------------------------

    return result