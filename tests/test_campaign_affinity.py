import pandas as pd
import pytest

from src.analytics.campaign_affinity import (
    calculate_campaign_affinity,
    calculate_campaign_type_affinity,
    get_best_campaign_type,
)


def test_calculate_campaign_affinity():
    campaigns = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C001",
                "C001",
                "C002",
            ],
            "campaign_type": [
                "Birthday",
                "Birthday",
                "Loyalty",
                "Discount",
            ],
            "sent": [
                1,
                1,
                1,
                1,
            ],
            "delivered": [
                1,
                1,
                1,
                0,
            ],
            "clicked": [
                1,
                0,
                1,
                0,
            ],
        }
    )

    result = calculate_campaign_affinity(campaigns)

    assert not result.empty

    assert {
        "customer_id",
        "campaign_type",
        "campaigns_sent",
        "campaigns_delivered",
        "campaigns_clicked",
        "campaigns_not_clicked",
        "click_rate",
    }.issubset(result.columns)


def test_campaign_affinity_click_rate_uses_delivered():
    campaigns = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C001",
                "C001",
            ],
            "campaign_type": [
                "Birthday",
                "Birthday",
                "Birthday",
            ],
            "sent": [
                1,
                1,
                1,
            ],
            "delivered": [
                1,
                1,
                0,
            ],
            "clicked": [
                1,
                0,
                0,
            ],
        }
    )

    result = calculate_campaign_affinity(campaigns)

    birthday = result[
        (result["customer_id"] == "C001")
        & (result["campaign_type"] == "Birthday")
    ].iloc[0]

    assert birthday["campaigns_sent"] == 3
    assert birthday["campaigns_delivered"] == 2
    assert birthday["campaigns_clicked"] == 1
    assert birthday["campaigns_not_clicked"] == 1
    assert birthday["click_rate"] == 0.50


def test_campaign_affinity_undelivered_not_counted_as_not_clicked():
    campaigns = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C001",
            ],
            "campaign_type": [
                "Loyalty",
                "Loyalty",
            ],
            "sent": [
                1,
                1,
            ],
            "delivered": [
                1,
                0,
            ],
            "clicked": [
                0,
                0,
            ],
        }
    )

    result = calculate_campaign_affinity(campaigns)

    loyalty = result[
        (result["customer_id"] == "C001")
        & (result["campaign_type"] == "Loyalty")
    ].iloc[0]

    assert loyalty["campaigns_sent"] == 2
    assert loyalty["campaigns_delivered"] == 1
    assert loyalty["campaigns_clicked"] == 0
    assert loyalty["campaigns_not_clicked"] == 1
    assert loyalty["click_rate"] == 0.0


def test_calculate_campaign_type_affinity():
    campaigns = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
            ],
            "campaign_type": [
                "Birthday",
                "Birthday",
                "Loyalty",
            ],
            "sent": [
                1,
                1,
                1,
            ],
            "delivered": [
                1,
                1,
                1,
            ],
            "clicked": [
                1,
                0,
                1,
            ],
        }
    )

    result = calculate_campaign_type_affinity(campaigns)

    assert not result.empty

    assert {
        "campaign_type",
        "campaigns_sent",
        "campaigns_delivered",
        "campaigns_clicked",
        "campaigns_not_clicked",
        "click_rate",
    }.issubset(result.columns)


def test_campaign_type_affinity_click_rate():
    campaigns = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
            ],
            "campaign_type": [
                "Birthday",
                "Birthday",
                "Birthday",
                "Birthday",
            ],
            "sent": [
                1,
                1,
                1,
                1,
            ],
            "delivered": [
                1,
                1,
                1,
                0,
            ],
            "clicked": [
                1,
                0,
                0,
                0,
            ],
        }
    )

    result = calculate_campaign_type_affinity(campaigns)

    birthday = result[
        result["campaign_type"] == "Birthday"
    ].iloc[0]

    assert birthday["campaigns_sent"] == 4
    assert birthday["campaigns_delivered"] == 3
    assert birthday["campaigns_clicked"] == 1
    assert birthday["campaigns_not_clicked"] == 2
    assert birthday["click_rate"] == pytest.approx(1 / 3)


def test_best_campaign_type():
    customer_affinity = pd.DataFrame(
        {
            "campaign_type": [
                "Birthday",
                "Loyalty",
                "Discount",
            ],
            "campaigns_delivered": [
                4,
                8,
                2,
            ],
            "campaigns_clicked": [
                2,
                2,
                0,
            ],
            "click_rate": [
                0.50,
                0.25,
                0.00,
            ],
        }
    )

    result = get_best_campaign_type(
        customer_affinity
    )

    assert result["campaign_type"] == "Birthday"
    assert result["click_rate"] == 0.50
    assert result["campaigns_delivered"] == 4
    assert result["campaigns_clicked"] == 2
    assert result["campaigns_not_clicked"] == 2


def test_best_campaign_type_ignores_undelivered_campaigns():
    customer_affinity = pd.DataFrame(
        {
            "campaign_type": [
                "Birthday",
                "Loyalty",
            ],
            "campaigns_delivered": [
                0,
                4,
            ],
            "campaigns_clicked": [
                0,
                1,
            ],
            "click_rate": [
                0.00,
                0.25,
            ],
        }
    )

    result = get_best_campaign_type(
        customer_affinity
    )

    assert result["campaign_type"] == "Loyalty"


def test_best_campaign_type_uses_exposure_as_tiebreaker():
    customer_affinity = pd.DataFrame(
        {
            "campaign_type": [
                "Birthday",
                "Loyalty",
            ],
            "campaigns_delivered": [
                2,
                10,
            ],
            "campaigns_clicked": [
                1,
                5,
            ],
            "click_rate": [
                0.50,
                0.50,
            ],
        }
    )

    result = get_best_campaign_type(
        customer_affinity
    )

    assert result["campaign_type"] == "Loyalty"
    assert result["campaigns_delivered"] == 10
    assert result["campaigns_clicked"] == 5


def test_best_campaign_type_returns_none_without_clicks():
    customer_affinity = pd.DataFrame(
        {
            "campaign_type": [
                "Birthday",
                "Loyalty",
            ],
            "campaigns_delivered": [
                4,
                6,
            ],
            "campaigns_clicked": [
                0,
                0,
            ],
            "click_rate": [
                0.00,
                0.00,
            ],
        }
    )

    result = get_best_campaign_type(
        customer_affinity
    )

    assert result is None


def test_best_campaign_type_requires_columns():
    customer_affinity = pd.DataFrame(
        {
            "campaign_type": [
                "Birthday",
            ],
            "campaigns_delivered": [
                2,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        get_best_campaign_type(
            customer_affinity
        )


def test_best_campaign_type_handles_no_delivered_campaigns():
    customer_affinity = pd.DataFrame(
        {
            "campaign_type": [
                "Birthday",
                "Loyalty",
            ],
            "campaigns_delivered": [
                0,
                0,
            ],
            "campaigns_clicked": [
                0,
                0,
            ],
            "click_rate": [
                0.00,
                0.00,
            ],
        }
    )

    result = get_best_campaign_type(
        customer_affinity
    )

    assert result is None