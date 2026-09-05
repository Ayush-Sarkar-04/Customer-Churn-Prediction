import pandas as pd

from src.features.feature_engineering import calculate_customer_tenure


# Test data
df = pd.DataFrame({
    "customer_id": [
        "C001",
        "C002",
        "C003"
    ],

    "registration_date": [
        "2024-01-10",
        "2024-06-15",
        "2025-01-01"
    ],

    "observation_date": [
        "2025-01-10",
        "2025-06-15",
        "2025-07-01"
    ]
})


# Calculate customer tenure
result = calculate_customer_tenure(df)


print("Customer tenure:")
print(
    result[
        [
            "customer_id",
            "registration_date",
            "observation_date",
            "customer_tenure"
        ]
    ]
)