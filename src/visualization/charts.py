import pandas as pd
import plotly.express as px


def create_segment_distribution_chart(segment_counts):
    """
    Create a bar chart showing customer distribution by segment.

    Parameters
    ----------
    segment_counts : pandas.DataFrame
        DataFrame containing:
        - customer_segment
        - customer_count

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly bar chart.
    """

    required_columns = {
        "customer_segment",
        "customer_count",
    }

    missing_columns = required_columns - set(segment_counts.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    chart_data = segment_counts.copy()

    fig = px.bar(
        chart_data,
        x="customer_segment",
        y="customer_count",
        title="Customer Distribution by Segment",
        labels={
            "customer_segment": "Customer Segment",
            "customer_count": "Number of Customers",
        },
        text="customer_count",
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        xaxis_tickangle=-25,
        showlegend=False,
        margin={
            "l": 40,
            "r": 40,
            "t": 70,
            "b": 100,
        },
    )

    return fig

def create_risk_distribution_chart(risk_counts):
    """
    Create a bar chart showing customer distribution by risk level.
    """

    required_columns = {
        "risk_level",
        "customer_count",
    }

    missing_columns = required_columns - set(risk_counts.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    chart_data = risk_counts.copy()

    fig = px.bar(
        chart_data,
        x="risk_level",
        y="customer_count",
        title="Customer Distribution by Churn Risk",
        labels={
            "risk_level": "Risk Level",
            "customer_count": "Number of Customers",
        },
        text="customer_count",
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        showlegend=False,
        margin={
            "l": 40,
            "r": 40,
            "t": 70,
            "b": 70,
        },
    )

    return fig

def create_segment_risk_chart(segment_risk_counts):
    """
    Create a grouped bar chart showing churn risk
    distribution across customer segments.
    """

    required_columns = {
        "customer_segment",
        "risk_level",
        "customer_count",
    }

    missing_columns = required_columns - set(segment_risk_counts.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    chart_data = segment_risk_counts.copy()

    fig = px.bar(
        chart_data,
        x="customer_segment",
        y="customer_count",
        color="risk_level",
        barmode="group",
        title="Churn Risk by Customer Segment",
        labels={
            "customer_segment": "Customer Segment",
            "customer_count": "Number of Customers",
            "risk_level": "Risk Level",
        },
    )

    fig.update_layout(
        xaxis_tickangle=-25,
        margin={
            "l": 40,
            "r": 40,
            "t": 70,
            "b": 100,
        },
    )

    return fig