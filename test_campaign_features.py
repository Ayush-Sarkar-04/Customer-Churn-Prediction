import pandas as pd

from src.features.feature_engineering import calculate_campaign_features


# Test campaign data
campaigns_df = pd.DataFrame({
    "campaign_id": [
        "CP001",
        "CP002",
        "CP003",
        "CP004",
        "CP005"
    ],

    "customer_id": [
        "C001",
        "C001",
        "C002",
        "C002",
        "C001"
    ],

    "campaign_date": [
        "2026-01-01",
        "2026-01-10",
        "2026-01-05",
        "2026-01-15",
        "2026-03-01"
    ],

    "sent": [
        1,
        1,
        1,
        1,
        1
    ],

    "delivered": [
        1,
        1,
        1,
        0,
        1
    ],

    "clicked": [
        1,
        0,
        1,
        0,
        1
    ],

    "redeemed": [
        1,
        0,
        0,
        0,
        1
    ]
})


# Observation date
observation_date = "2026-02-10"


# Calculate campaign features
result = calculate_campaign_features(
    campaigns_df,
    observation_date
)


print("CAMPAIGN FEATURES:")
print(result)