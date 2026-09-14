from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    COLOR_BACKGROUND,
    COLOR_SURFACE,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_BORDER,
    COLOR_GRID,
    COLOR_PRIMARY,
    COLOR_TERTIARY,
    COLOR_ALERT,
    COLOR_NEUTRAL,
    RETENTION_PRIORITY_COLORS,
)
from src.analytics.customer_value import (
    calculate_customer_value,
    calculate_customer_value_summary,
)
from src.analytics.retention_cost import (
    calculate_retention_cost,
    calculate_total_retention_cost,
    calculate_average_retention_cost,
    calculate_customer_retention_cost,
    aggregate_retention_cost_by_campaign_type,
)
from src.analytics.economic_value import (
    calculate_economic_value_at_risk,
    calculate_economic_value_summary,
)
from src.analytics.economic_priority import (
    calculate_economic_priority,
    rank_economic_priority,
    summarize_economic_priority,
)


CHART_FONT = "Arial"


def _apply_chart_style(fig):
    fig.update_layout(
        font={"family": CHART_FONT, "color": COLOR_TEXT},
        paper_bgcolor=COLOR_BACKGROUND,
        plot_bgcolor=COLOR_BACKGROUND,
        margin={"l": 45, "r": 25, "t": 30, "b": 45},
        hoverlabel={
            "bgcolor": COLOR_SURFACE,
            "font": {"color": COLOR_TEXT, "family": CHART_FONT},
            "bordercolor": COLOR_BORDER,
        },
    )
    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=COLOR_BORDER,
        tickfont={"color": COLOR_TEXT_MUTED, "family": CHART_FONT},
        title_font={"color": COLOR_TEXT_MUTED, "family": CHART_FONT},
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLOR_GRID,
        zeroline=False,
        linecolor=COLOR_BORDER,
        tickfont={"color": COLOR_TEXT_MUTED, "family": CHART_FONT},
        title_font={"color": COLOR_TEXT_MUTED, "family": CHART_FONT},
    )
    return fig


def _currency(value):
    return f"₹{float(value):,.0f}"


def _percent(value):
    return f"{float(value):.1%}"


def _kpi(label, value, description, accent=COLOR_PRIMARY):
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="border-top:2px solid {accent}; padding-top:8px;">
                <div style="color:{COLOR_TEXT_MUTED};font-size:10px;font-weight:700;letter-spacing:.08em;">
                    {label}
                </div>
                <div style="color:{COLOR_TEXT};font-size:22px;font-weight:700;margin-top:7px;">
                    {value}
                </div>
                <div style="color:{COLOR_TEXT_MUTED};font-size:11px;margin-top:6px;">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_economic_dataset(customer_analytics, transactions, campaigns):
    """
    Build the customer-level economic dataset used by both Level 2 pages.

    Customer value is historical revenue. Retention cost is the customer's
    historical campaign-cost total. These are analytical inputs, not causal
    estimates of intervention cost or revenue saved.
    """
    # The transaction source uses DD-MM-YYYY dates (for example,
    # 16-07-2024), while the Customer Value backend expects ISO-style
    # YYYY-MM-DD input. Normalize the date at the page integration
    # boundary so the backend contract remains unchanged.
    transactions_for_value = transactions.copy()
    if "transaction_date" not in transactions_for_value.columns:
        raise ValueError("Transactions must contain transaction_date.")

    parsed_transaction_dates = pd.to_datetime(
        transactions_for_value["transaction_date"],
        dayfirst=True,
        errors="raise",
    )
    transactions_for_value["transaction_date"] = parsed_transaction_dates.dt.strftime(
        "%Y-%m-%d"
    )

    customer_value = calculate_customer_value(transactions_for_value)

    campaign_cost = calculate_retention_cost(campaigns)
    customer_cost = calculate_customer_retention_cost(campaign_cost)
    customer_cost = customer_cost.rename(
        columns={"total_retention_cost": "retention_cost"}
    )

    current = customer_analytics.copy()
    if "customer_id" not in current.columns:
        raise ValueError("Customer analytics must contain customer_id.")

    current["customer_id"] = current["customer_id"].astype(str).str.strip()
    customer_value["customer_id"] = customer_value["customer_id"].astype(str).str.strip()
    customer_cost["customer_id"] = customer_cost["customer_id"].astype(str).str.strip()

    economic = current.merge(
        customer_value[
            [
                "customer_id",
                "total_historical_revenue",
                "purchase_count",
                "average_order_value",
                "active_days",
                "annualized_revenue",
                "customer_value",
            ]
        ],
        on="customer_id",
        how="left",
    )
    economic = economic.merge(
        customer_cost[["customer_id", "retention_cost", "campaign_count", "average_retention_cost"]],
        on="customer_id",
        how="left",
    )

    for column in [
        "customer_value",
        "total_historical_revenue",
        "purchase_count",
        "average_order_value",
        "active_days",
        "annualized_revenue",
        "retention_cost",
        "campaign_count",
        "average_retention_cost",
    ]:
        if column in economic.columns:
            economic[column] = pd.to_numeric(economic[column], errors="coerce").fillna(0)

    economic = calculate_economic_value_at_risk(
        economic,
        churn_probability_column="churn_probability",
        customer_value_column="customer_value",
        retention_cost_column="retention_cost",
    )
    return economic


def render_economic_intelligence_page(customer_analytics, transactions, campaigns):
    st.title("Economic Intelligence")
    st.caption(
        "Measure customer value, retention cost and the economic exposure associated with predicted churn."
    )

    economic = build_economic_dataset(customer_analytics, transactions, campaigns)
    if economic.empty:
        st.info("No customer-level economic data is available.")
        return

    value_summary = calculate_customer_value_summary(
        economic[
            ["customer_value", "annualized_revenue"]
        ]
    )
    economic_summary = calculate_economic_value_summary(economic)
    campaign_cost = calculate_retention_cost(campaigns)
    total_cost = calculate_total_retention_cost(
        economic[["retention_cost"]]
    )
    average_cost = calculate_average_retention_cost(
        economic[["retention_cost"]]
    )

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        _kpi("CUSTOMERS ANALYZED", f"{len(economic):,}", "Current customer profiles")
    with c2:
        _kpi("TOTAL CUSTOMER VALUE", _currency(value_summary["total_customer_value"]), "Historical customer value")
    with c3:
        _kpi("TOTAL RETENTION COST", _currency(total_cost), "Historical campaign-cost proxy", COLOR_TERTIARY)
    with c4:
        _kpi("EXPECTED VALUE AT RISK", _currency(economic_summary["total_expected_value_at_risk"]), "Churn probability × customer value", COLOR_ALERT)

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        _kpi("NET VALUE AT RISK", _currency(economic_summary["total_net_value_at_risk"]), "EVaR after retention cost", COLOR_ALERT)
    with c2:
        _kpi("AVG. CUSTOMER VALUE", _currency(value_summary["average_customer_value"]), "Mean historical customer value")
    with c3:
        _kpi("AVG. RETENTION COST", _currency(average_cost), "Average historical campaign cost", COLOR_TERTIARY)
    with c4:
        _kpi("AVG. EVaR", _currency(economic_summary["average_expected_value_at_risk"]), "Average economic exposure", COLOR_ALERT)

    st.divider()

    st.subheader("Economic Exposure")
    st.caption("Customers with the highest net economic exposure after accounting for historical retention cost.")

    top = (
        economic.sort_values("net_value_at_risk", ascending=False)
        .head(15)
        .sort_values("net_value_at_risk", ascending=True)
    )
    fig = px.bar(
        top,
        x="net_value_at_risk",
        y="customer_id",
        orientation="h",
        text="net_value_at_risk",
        color_discrete_sequence=[COLOR_PRIMARY],
    )
    fig = _apply_chart_style(fig)
    fig.update_layout(
        xaxis_title="Net Value at Risk",
        yaxis_title=None,
        showlegend=False,
        height=460,
        xaxis_tickprefix="₹",
    )
    fig.update_traces(
        texttemplate="₹%{text:,.0f}",
        textposition="outside",
        marker_line_width=0,
        textfont_color=COLOR_TEXT,
    )
    st.plotly_chart(fig, width="stretch")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.subheader("Customer Value vs Churn Probability")
        st.caption("Higher-value customers with higher predicted churn contribute more to EVaR.")
        scatter = economic.copy()
        fig = px.scatter(
            scatter,
            x="customer_value",
            y="churn_probability",
            size="expected_value_at_risk",
            color="economic_priority_tier",
            color_discrete_map=RETENTION_PRIORITY_COLORS,
            hover_name="customer_id",
            hover_data={
                "customer_value": ":,.0f",
                "churn_probability": ":.1%",
                "expected_value_at_risk": ":,.0f",
                "economic_priority_tier": True,
            },
        )
        fig = _apply_chart_style(fig)
        fig.update_layout(
            xaxis_title="Customer Value",
            yaxis_title="Churn Probability",
            yaxis_tickformat=".0%",
            height=420,
            legend_title="Economic Priority",
        )
        st.plotly_chart(fig, width="stretch")

    with col2:
        st.subheader("Retention Cost by Campaign Type")
        st.caption("Historical campaign-cost distribution used as the retention-cost input.")
        cost_by_type = aggregate_retention_cost_by_campaign_type(campaign_cost)
        cost_by_type = cost_by_type.sort_values("total_retention_cost", ascending=True)
        fig = px.bar(
            cost_by_type,
            x="total_retention_cost",
            y="campaign_type",
            orientation="h",
            text="total_retention_cost",
            color_discrete_sequence=[COLOR_TERTIARY],
        )
        fig = _apply_chart_style(fig)
        fig.update_layout(
            xaxis_title="Retention Cost",
            yaxis_title=None,
            showlegend=False,
            height=420,
            xaxis_tickprefix="₹",
        )
        fig.update_traces(
            texttemplate="₹%{text:,.0f}",
            textposition="outside",
            marker_line_width=0,
            textfont_color=COLOR_TEXT,
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Customer Economic Profile")
    st.caption("Top customers ranked by expected economic exposure.")
    table = (
        economic.sort_values("expected_value_at_risk", ascending=False)
        .head(20)[
            [
                "customer_id",
                "churn_probability",
                "customer_value",
                "retention_cost",
                "expected_value_at_risk",
                "net_value_at_risk",
            ]
        ]
        .copy()
    )
    table.columns = [
        "Customer ID",
        "Churn Probability",
        "Customer Value",
        "Retention Cost",
        "Expected Value at Risk",
        "Net Value at Risk",
    ]
    table["Churn Probability"] = table["Churn Probability"].map(_percent)
    for column in table.columns[2:]:
        table[column] = table[column].map(_currency)
    st.dataframe(table, width="stretch", hide_index=True, height=520)

    st.caption(
        "Interpretation note: EVaR is an analytical exposure metric, not guaranteed revenue loss, revenue saved, ROI, or treatment effect."
    )


def render_retention_planning_page(customer_analytics, transactions, campaigns):
    st.title("Retention Planning")
    st.caption(
        "Prioritize customers using economic exposure after accounting for historical retention cost."
    )

    economic = build_economic_dataset(customer_analytics, transactions, campaigns)
    if economic.empty:
        st.info("No customer-level economic data is available.")
        return

    economic = calculate_economic_priority(economic)
    summary = summarize_economic_priority(economic)
    ranked = rank_economic_priority(economic)

    c1, c2, c3, c4 = st.columns(4, gap="medium")
    with c1:
        _kpi("CRITICAL", f"{summary['critical_customers']:,}", "Highest economic priority", COLOR_ALERT)
    with c2:
        _kpi("HIGH + CRITICAL", f"{summary['high_customers'] + summary['critical_customers']:,}", "Customers requiring priority action", COLOR_TERTIARY)
    with c3:
        _kpi("HIGH + CRITICAL VALUE", _currency(summary["high_or_critical_net_value_at_risk"]), "Net value at risk in priority tiers", COLOR_ALERT)
    with c4:
        _kpi("AVG. PRIORITY SCORE", f"{economic['economic_priority_score'].mean():.1f}", "Relative economic priority")

    st.divider()
    st.subheader("Economic Priority Distribution")
    st.caption("Relative priority is based on percentile rank of Net Value at Risk within the current customer population.")

    tier_order = ["Low", "Medium", "High", "Critical"]
    counts = (
        economic["economic_priority_tier"]
        .value_counts()
        .reindex(tier_order, fill_value=0)
        .rename_axis("Priority")
        .reset_index(name="Customers")
    )
    fig = px.bar(
        counts,
        x="Customers",
        y="Priority",
        orientation="h",
        text="Customers",
        color="Priority",
        category_orders={"Priority": tier_order},
        color_discrete_map=RETENTION_PRIORITY_COLORS,
    )
    fig = _apply_chart_style(fig)
    fig.update_layout(
        xaxis_title="Customers",
        yaxis_title=None,
        showlegend=False,
        height=390,
    )
    fig.update_traces(
        textposition="outside",
        marker_line_width=0,
        textfont_color=COLOR_TEXT,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Priority Customer Ranking")
    st.caption("Customers are ranked by Economic Priority Score, then Net Value at Risk, then customer ID.")

    table = ranked.head(50)[
        [
            "economic_priority_rank",
            "customer_id",
            "economic_priority_score",
            "economic_priority_tier",
            "churn_probability",
            "customer_value",
            "retention_cost",
            "expected_value_at_risk",
            "net_value_at_risk",
        ]
    ].copy()
    table.columns = [
        "Rank",
        "Customer ID",
        "Priority Score",
        "Priority",
        "Churn Probability",
        "Customer Value",
        "Retention Cost",
        "Expected Value at Risk",
        "Net Value at Risk",
    ]
    table["Priority Score"] = table["Priority Score"].map(lambda x: f"{float(x):.1f}")
    table["Churn Probability"] = table["Churn Probability"].map(_percent)
    for column in [
        "Customer Value",
        "Retention Cost",
        "Expected Value at Risk",
        "Net Value at Risk",
    ]:
        table[column] = table[column].map(_currency)

    st.dataframe(table, width="stretch", hide_index=True, height=650)

    st.caption(
        "Economic Priority is a relative prioritization metric. It does not represent guaranteed revenue loss, revenue saved, campaign ROI, or incremental treatment effect."
    )
