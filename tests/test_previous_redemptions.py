import pandas as pd

from src.features.feature_engineering import calculate_previous_redemptions


# Sample campaign data
campaigns_df = pd.DataFrame({
    "customer_id": [
        "C001",
        "C001",
        "C001",
        "C002",
        "C002",
        "C003"
    ],

    "campaign_date": [
        "2025-10-01",
        "2025-11-15",
        "2026-02-15",
        "2025-12-01",
        "2026-01-20",
        "2025-09-10"
    ],

    "redeemed": [
        1,
        1,
        1,
        1,
        0,
        0
    ],

    "redemption_date": [
        "2025-10-05",
        "2025-11-20",
        "2026-02-20",
        "2025-12-05",
        None,
        None
    ]
})


# Observation date
observation_date = "2026-02-10"


# Calculate previous redemptions
result = calculate_previous_redemptions(
    campaigns_df,
    observation_date
)


print("PREVIOUS REDEMPTIONS:")
print(result)