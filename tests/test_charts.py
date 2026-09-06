import pandas as pd

from src.visualization.charts import (
    create_segment_distribution_chart,
    create_risk_distribution_chart,
    create_segment_risk_chart,
)


def test_segment_distribution_chart():
    data = pd.DataFrame(
        {
            "customer_segment": [
                "Champions",
                "Loyal Customers",
                "Potential Loyalists",
                "Regular Customers",
                "At Risk",
                "Inactive",
                "Lost",
            ],
            "customer_count": [
                914,
                613,
                785,
                1792,
                376,
                353,
                182,
            ],
        }
    )

    fig = create_segment_distribution_chart(data)

    assert fig is not None
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == 7
    assert len(fig.data[0].y) == 7


def test_segment_distribution_chart_missing_columns():
    data = pd.DataFrame(
        {
            "customer_segment": ["Champions"],
        }
    )

    try:
        create_segment_distribution_chart(data)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "customer_count" in str(exc)


def test_risk_distribution_chart():
    data = pd.DataFrame(
        {
            "risk_level": [
                "Low",
                "Medium",
                "High",
                "Very High",
                "Churned",
            ],
            "customer_count": [
                1000,
                1200,
                900,
                600,
                315,
            ],
        }
    )

    fig = create_risk_distribution_chart(data)

    assert fig is not None
    assert len(fig.data) == 1
    assert len(fig.data[0].x) == 5
    assert len(fig.data[0].y) == 5


def test_risk_distribution_chart_missing_columns():
    data = pd.DataFrame(
        {
            "risk_level": ["Low"],
        }
    )

    try:
        create_risk_distribution_chart(data)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "customer_count" in str(exc)


def test_segment_risk_chart():
    data = pd.DataFrame(
        {
            "customer_segment": [
                "Champions",
                "Champions",
                "Loyal Customers",
                "Loyal Customers",
                "At Risk",
                "At Risk",
                "Lost",
                "Lost",
            ],
            "risk_level": [
                "Low",
                "Churned",
                "Low",
                "Churned",
                "Medium",
                "Churned",
                "Low",
                "Churned",
            ],
            "customer_count": [
                700,
                214,
                500,
                113,
                100,
                276,
                20,
                162,
            ],
        }
    )

    fig = create_segment_risk_chart(data)

    assert fig is not None
    assert len(fig.data) > 0


def test_segment_risk_chart_missing_columns():
    data = pd.DataFrame(
        {
            "customer_segment": ["Champions"],
            "risk_level": ["Low"],
        }
    )

    try:
        create_segment_risk_chart(data)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "customer_count" in str(exc)