import pandas as pd

from src.data.validator import validate_customer_references


# Valid customer data
customers_df = pd.DataFrame({
    "customer_id": [
        "C001",
        "C002",
        "C003"
    ]
})


# Transactions reference existing customers
transactions_df = pd.DataFrame({
    "customer_id": [
        "C001",
        "C002",
        "C999"
    ]
})


# Campaigns reference existing customers
campaigns_df = pd.DataFrame({
    "customer_id": [
        "C001",
        "C002",
        "C003"
    ]
})


result = validate_customer_references(
    customers_df,
    transactions_df,
    campaigns_df
)

print(result)