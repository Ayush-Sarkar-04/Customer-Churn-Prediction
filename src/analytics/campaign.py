import pandas as pd


def calculate_campaign_funnel(campaigns_df):
    """
    Calculate overall campaign funnel metrics.

    Funnel:
    Sent → Delivered → Clicked → Redeemed
    """

    campaigns = campaigns_df.copy()

    sent = campaigns["sent"].sum()
    delivered = campaigns["delivered"].sum()
    clicked = campaigns["clicked"].sum()
    redeemed = campaigns["redeemed"].sum()

    delivery_rate = (
        delivered / sent
        if sent > 0
        else 0
    )

    click_rate = (
        clicked / delivered
        if delivered > 0
        else 0
    )

    redemption_rate = (
        redeemed / delivered
        if delivered > 0
        else 0
    )

    return {
        "sent": sent,
        "delivered": delivered,
        "clicked": clicked,
        "redeemed": redeemed,
        "delivery_rate": delivery_rate,
        "click_rate": click_rate,
        "redemption_rate": redemption_rate
    }


def calculate_campaign_performance(campaigns_df):
    """
    Calculate performance metrics for each campaign type.
    """

    campaigns = campaigns_df.copy()

    performance = (
        campaigns
        .groupby("campaign_type")
        .agg(
            campaigns_sent=("sent", "sum"),
            campaigns_delivered=("delivered", "sum"),
            campaigns_clicked=("clicked", "sum"),
            campaigns_redeemed=("redeemed", "sum"),
            total_cost=("campaign_cost", "sum"),
            total_reward_value=("reward_value", "sum")
        )
        .reset_index()
    )

    performance["delivery_rate"] = (
        performance["campaigns_delivered"]
        / performance["campaigns_sent"]
    )

    performance["click_rate"] = (
        performance["campaigns_clicked"]
        / performance["campaigns_delivered"]
    )

    performance["redemption_rate"] = (
        performance["campaigns_redeemed"]
        / performance["campaigns_delivered"]
    )

    performance = performance.replace(
        [float("inf"), float("-inf")],
        0
    )

    rate_columns = [
        "delivery_rate",
        "click_rate",
        "redemption_rate"
    ]

    performance[rate_columns] = (
        performance[rate_columns].fillna(0)
    )

    return performance