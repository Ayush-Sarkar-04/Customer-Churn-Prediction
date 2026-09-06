import pandas as pd

from src.analytics.segmentation import (
    calculate_rfm_scores,
    assign_customer_segments,
    get_segment_summary,
    get_segment_counts,
)


def test_rfm_scores():
    df = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "recency": [10, 30, 60, 100, 200],
            "frequency": [20, 15, 10, 5, 1],
            "monetary": [5000, 4000, 3000, 1500, 500],
        }
    )

    result = calculate_rfm_scores(df)

    assert len(result) == 5
    assert "recency_score" in result.columns
    assert "frequency_score" in result.columns
    assert "monetary_score" in result.columns
    assert "rfm_score" in result.columns


def test_customer_segments():
    df = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "recency": [10, 30, 60, 100, 200],
            "frequency": [20, 15, 10, 5, 1],
            "monetary": [5000, 4000, 3000, 1500, 500],
        }
    )

    result = assign_customer_segments(df)

    assert len(result) == 5
    assert "customer_segment" in result.columns

    valid_segments = {
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "Regular Customers",
        "At Risk",
        "Inactive",
        "Lost",
    }

    assert set(result["customer_segment"]).issubset(valid_segments)


def test_segment_summary():
    df = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "recency": [10, 100, 200],
            "frequency": [20, 5, 1],
            "monetary": [5000, 1500, 500],
        }
    )

    segmented = assign_customer_segments(df)
    summary = get_segment_summary(segmented)

    assert "customer_segment" in summary.columns
    assert "customer_count" in summary.columns
    assert "average_recency" in summary.columns
    assert "average_frequency" in summary.columns
    assert "average_monetary" in summary.columns


def test_segment_counts():
    df = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004"],
            "recency": [10, 100, 200, 30],
            "frequency": [20, 5, 1, 15],
            "monetary": [5000, 1500, 500, 4000],
        }
    )

    segmented = assign_customer_segments(df)
    counts = get_segment_counts(segmented)

    assert "customer_segment" in counts.columns
    assert "customer_count" in counts.columns
    assert "percentage" in counts.columns

    assert counts["customer_count"].sum() == 4
    assert round(counts["percentage"].sum(), 2) == 100.00