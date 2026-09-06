import pandas as pd


SEGMENTS = [
    "Champions",
    "Loyal Customers",
    "Potential Loyalists",
    "Regular Customers",
    "At Risk",
    "Inactive",
    "Lost",
]


def calculate_rfm_scores(rfm_df):
    """
    Calculate RFM scores from recency, frequency, and monetary values.

    Recency:
        Lower is better, so the score is reversed.

    Frequency:
        Higher is better.

    Monetary:
        Higher is better.
    """

    result = rfm_df.copy()

    result["recency_score"] = pd.qcut(
        result["recency"].rank(method="first"),
        q=5,
        labels=[5, 4, 3, 2, 1],
    ).astype(int)

    result["frequency_score"] = pd.qcut(
        result["frequency"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    result["monetary_score"] = pd.qcut(
        result["monetary"].rank(method="first"),
        q=5,
        labels=[1, 2, 3, 4, 5],
    ).astype(int)

    result["rfm_score"] = (
        result["recency_score"]
        + result["frequency_score"]
        + result["monetary_score"]
    )

    return result


def assign_customer_segment(row):
    """
    Assign a customer segment using RFM scores and recency.
    """

    recency = row["recency"]
    rfm_score = row["rfm_score"]
    frequency_score = row["frequency_score"]
    monetary_score = row["monetary_score"]

    if recency >= 180:
        return "Lost"

    if recency >= 90:
        return "Inactive"

    if recency >= 60:
        return "At Risk"

    if (
        rfm_score >= 13
        and frequency_score >= 4
        and monetary_score >= 4
    ):
        return "Champions"

    if (
        rfm_score >= 11
        and frequency_score >= 4
    ):
        return "Loyal Customers"

    if (
        rfm_score >= 9
        and frequency_score >= 3
    ):
        return "Potential Loyalists"

    return "Regular Customers"


def assign_customer_segments(rfm_df):
    """
    Calculate RFM scores and assign a segment to each customer.
    """

    result = calculate_rfm_scores(rfm_df)

    result["customer_segment"] = result.apply(
        assign_customer_segment,
        axis=1,
    )

    return result


def get_segment_summary(segmented_df):
    """
    Return customer counts and average RFM metrics by segment.
    """

    summary = (
        segmented_df.groupby("customer_segment")
        .agg(
            customer_count=("customer_id", "count"),
            average_recency=("recency", "mean"),
            average_frequency=("frequency", "mean"),
            average_monetary=("monetary", "mean"),
        )
        .reset_index()
    )

    segment_order = {segment: i for i, segment in enumerate(SEGMENTS)}

    summary["segment_order"] = summary["customer_segment"].map(
        segment_order
    )

    summary = (
        summary.sort_values("segment_order")
        .drop(columns="segment_order")
        .reset_index(drop=True)
    )

    return summary

def get_segment_counts(segmented_df):
    """
    Return customer counts and percentages by segment.
    """

    counts = (
        segmented_df["customer_segment"]
        .value_counts()
        .reindex(SEGMENTS, fill_value=0)
        .reset_index()
    )

    counts.columns = ["customer_segment", "customer_count"]

    counts["percentage"] = (
        counts["customer_count"]
        / counts["customer_count"].sum()
        * 100
    ).round(2)

    return counts