import pandas as pd

from src.features.feature_engineering import calculate_rfm_features


# Test transaction data
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
        "C002",
        "C001"
    ],

    "transaction_date": [
        "2026-01-01",
        "2026-01-10",
        "2026-01-05",
        "2026-01-15",
        "2026-02-01"
    ],

    "bill_amount": [
        500,
        750,
        1000,
        250,
        300
    ]
})


# Observation date
observation_date = "2026-02-10"


# Calculate RFM
result = calculate_rfm_features(
    transactions_df,
    observation_date
)


print("RFM FEATURES:")
print(result)