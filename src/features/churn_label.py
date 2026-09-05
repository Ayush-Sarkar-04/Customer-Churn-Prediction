import pandas as pd


def calculate_future_90_day_purchases(
    customers_df,
    transactions_df,
    observation_date
):
    """
    Calculate the number of purchases made by each customer
    during the 90 days after the observation date.
    """

    customers = customers_df.copy()
    transactions = transactions_df.copy()

    # Convert transaction dates
    transactions["transaction_date"] = pd.to_datetime(
        transactions["transaction_date"],
        errors="coerce"
    )

    # Convert observation date
    observation_date = pd.to_datetime(observation_date)

    # Define the 90-day future period
    future_end_date = observation_date + pd.Timedelta(days=90)

    # Keep transactions within the future 90-day period
    future_transactions = transactions[
        (transactions["transaction_date"] > observation_date)
        & (transactions["transaction_date"] <= future_end_date)
    ].copy()

    # Count future purchases
    purchase_counts = (
        future_transactions
        .groupby("customer_id")
        .size()
        .reset_index(name="future_90_day_purchases")
    )

    # Start with ALL customers
    result = customers[["customer_id"]].copy()

    # Add future purchase counts
    result = result.merge(
        purchase_counts,
        on="customer_id",
        how="left"
    )

    # Customers with no future purchases get 0
    result["future_90_day_purchases"] = (
        result["future_90_day_purchases"]
        .fillna(0)
        .astype(int)
    )

    return result


def calculate_churn_label(
    customers_df,
    transactions_df,
    observation_date
):
    """
    Calculate churn labels for every customer.

    Churn:
        1 = No purchase in the following 90 days
        0 = At least one purchase in the following 90 days
    """

    result = calculate_future_90_day_purchases(
        customers_df,
        transactions_df,
        observation_date
    )

    # Create churn label
    result["churn"] = (
        result["future_90_day_purchases"] == 0
    ).astype(int)

    return result