import pandas as pd

from src.features.feature_engineering import calculate_campaign_features

from src.analytics.campaign import (
    calculate_campaign_funnel,
    calculate_campaign_performance
)


# --------------------------------------------------
# Test campaign data
# --------------------------------------------------

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


# --------------------------------------------------
# Observation date
# --------------------------------------------------

observation_date = "2026-02-10"


# --------------------------------------------------
# Test campaign features
# --------------------------------------------------

result = calculate_campaign_features(
    campaigns_df,
    observation_date
)


print("CAMPAIGN FEATURES:")
print("==================")
print(result)


# --------------------------------------------------
# Test campaign analytics
# --------------------------------------------------

def test_campaign_analytics():

    campaigns = pd.read_csv(
        "data/training/campaigns.csv"
    )

    # Calculate overall campaign funnel
    funnel = calculate_campaign_funnel(
        campaigns
    )

    print("\nCAMPAIGN FUNNEL:")
    print("================")

    print(f"Sent:       {funnel['sent']}")
    print(f"Delivered:  {funnel['delivered']}")
    print(f"Clicked:    {funnel['clicked']}")
    print(f"Redeemed:   {funnel['redeemed']}")

    print(
        f"Delivery Rate:   "
        f"{funnel['delivery_rate']:.4f}"
    )

    print(
        f"Click Rate:      "
        f"{funnel['click_rate']:.4f}"
    )

    print(
        f"Redemption Rate: "
        f"{funnel['redemption_rate']:.4f}"
    )

    # Calculate campaign-type performance
    performance = calculate_campaign_performance(
        campaigns
    )

    print("\nCAMPAIGN PERFORMANCE:")
    print("=====================")
    print(performance)

    # Verify funnel values
    assert funnel["sent"] >= 0
    assert funnel["delivered"] >= 0
    assert funnel["clicked"] >= 0
    assert funnel["redeemed"] >= 0

    assert 0 <= funnel["delivery_rate"] <= 1
    assert 0 <= funnel["click_rate"] <= 1
    assert 0 <= funnel["redemption_rate"] <= 1

    # Verify performance table
    assert len(performance) > 0

    required_columns = [
        "campaign_type",
        "campaigns_sent",
        "campaigns_delivered",
        "campaigns_clicked",
        "campaigns_redeemed",
        "total_cost",
        "total_reward_value",
        "delivery_rate",
        "click_rate",
        "redemption_rate"
    ]

    for column in required_columns:
        assert column in performance.columns