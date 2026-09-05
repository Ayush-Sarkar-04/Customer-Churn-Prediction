import pandas as pd

from src.features.churn_label import calculate_churn_label


# Customer data
customers_df = pd.DataFrame({
    "customer_id": [
        "C001",
        "C002",
        "C003",
        "C004"
    ]
})


# Transaction data
transactions_df = pd.DataFrame({
    "transaction_id": [
        "T001",
        "T002",
        "T003",
        "T004",
        "T005"
    ],
    "customer_id": [
        "C001",
        "C001",
        "C002",
        "C003",
        "C003"
    ],
    "transaction_date": [
        "2026-01-10",
        "2026-03-15",
        "2026-02-20",
        "2026-02-15",
        "2026-05-01"
    ],
    "bill_amount": [
        500,
        750,
        1000,
        600,
        800
    ]
})


# Observation date
observation_date = "2026-02-10"


# Calculate churn labels
result = calculate_churn_label(
    customers_df,
    transactions_df,
    observation_date
)


print("CHURN LABEL RESULT:")
print(result)