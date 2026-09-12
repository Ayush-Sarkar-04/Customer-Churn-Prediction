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
)
from src.analytics.campaign_affinity import (
    calculate_campaign_affinity,
    calculate_campaign_type_affinity,
    get_customer_campaign_affinity,
    search_campaign_customers,
)
# =========================================================
# FORMATTING HELPERS
# =========================================================
def _format_number(value):
    if pd.isna(value):
        return "—"
    return f"{int(value):,}"
def _format_percent(value):
    if pd.isna(value):
        return "—"

    return f"{float(value):.2%}"
# =========================================================
# CARD
# =========================================================
def _render_card(
    label,
    value,
    description,
):
    st.markdown(
        f"""
        <div style="
            background-color: {COLOR_SURFACE};
            border: 1px solid {COLOR_BORDER};
            border-top: 2px solid #8C9299;
            border-radius: 8px;
            padding: 18px;
            min-height: 125px;
        ">
            <div style="
                color: {COLOR_TEXT_MUTED};
                font-size: 11px;
                font-weight: 600;
                letter-spacing: 0.3px;
                text-transform: uppercase;
                margin-bottom: 10px;
            ">
                {label}
            </div>
            <div style="
                color: {COLOR_TEXT};
                font-size: 23px;
                font-weight: 600;
                line-height: 1.15;
                margin-bottom: 8px;
            ">
                {value}
            </div>
            <div style="
                color: {COLOR_TEXT_MUTED};
                font-size: 12px;
                line-height: 1.4;
            ">
                {description}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
# =========================================================
# CHART STYLE
# =========================================================
def _style_affinity_chart(fig):
    fig.update_layout(
        font={
            "family": "Arial",
            "color": COLOR_TEXT,
        },
        paper_bgcolor=COLOR_BACKGROUND,
        plot_bgcolor=COLOR_BACKGROUND,
        margin={
            "l": 30,
            "r": 30,
            "t": 20,
            "b": 45,
        },
        showlegend=False,
        hoverlabel={
            "bgcolor": COLOR_SURFACE,
            "font": {
                "color": COLOR_TEXT,
                "family": "Arial",
            },
            "bordercolor": COLOR_BORDER,
        },
    )
    fig.update_traces(
        marker_color="#A7ADB4",
        marker_line_color="#A7ADB4",
        textposition="outside",
        textfont={
            "color": COLOR_TEXT,
            "family": "Arial",
        },
        hovertemplate=(
            "<b>%{y}</b>"
            "<br>Click rate: %{x:.2%}"
            "<br>Delivered: %{customdata[0]:,}"
            "<br>Clicked: %{customdata[1]:,}"
            "<extra></extra>"
        ),
    )
    fig.update_xaxes(
        tickformat=".0%",
        showgrid=False,
        zeroline=False,
        linecolor=COLOR_BORDER,
        tickfont={
            "color": COLOR_TEXT_MUTED,
            "family": "Arial",
        },
        title="Click Rate",
        title_font={
            "color": COLOR_TEXT_MUTED,
            "family": "Arial",
        },
        range=[
            0,
            max(
                0.10,
                float(fig.data[0].x.max()) * 1.15
            ),
        ],
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLOR_GRID,
        zeroline=False,
        linecolor=COLOR_BORDER,
        tickfont={
            "color": COLOR_TEXT_MUTED,
            "family": "Arial",
        },
        title=None,
    )
    return fig
# =========================================================
# CAMPAIGN AFFINITY PAGE
# =========================================================
def render_campaign_affinity_page(campaigns):
    """
    Render the Campaign Affinity page.
    The page answers:
    Which campaign types does a customer engage with?
    Affinity is based on delivered campaigns and clicks.
    Undelivered campaigns are not treated as interaction
    opportunities.
    """
    if campaigns is None or campaigns.empty:
        st.info("No campaign data is available.")
        return
    # =====================================================
    # PREPARE AFFINITY DATA
    # =====================================================
    affinity = calculate_campaign_affinity(campaigns)
    campaign_type_affinity = calculate_campaign_type_affinity(
        campaigns
    )
    customer_ids = sorted(
        affinity["customer_id"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    campaign_types = sorted(
        affinity["campaign_type"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    # =====================================================
    # HEADER
    # =====================================================
    st.title("Campaign Affinity")
    st.caption(
        "Understand which campaign types each customer engages "
        "with based on delivered messages and clicks."
    )
    st.divider()
    # =====================================================
    # OVERVIEW CARDS
    # =====================================================
    total_customers = len(customer_ids)
    total_campaign_types = len(campaign_types)
    strongest_campaign = (
        campaign_type_affinity
        .sort_values(
            [
                "click_rate",
                "campaigns_delivered",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .iloc[0]
    )
    total_delivered = int(
        campaign_type_affinity[
            "campaigns_delivered"
        ].sum()
    )
    total_clicked = int(
        campaign_type_affinity[
            "campaigns_clicked"
        ].sum()
    )
    overall_click_rate = (
        total_clicked / total_delivered
        if total_delivered > 0
        else 0
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _render_card(
            "Customers Analyzed",
            f"{total_customers:,}",
            "Customers with campaign history",
        )
    with col2:
        _render_card(
            "Campaign Types",
            f"{total_campaign_types:,}",
            "Distinct campaign categories",
        )
    with col3:
        _render_card(
            "Strongest Campaign Type",
            strongest_campaign["campaign_type"],
            (
                f"{strongest_campaign['click_rate']:.2%} "
                "overall click rate"
            ),
        )
    with col4:
        _render_card(
            "Overall Click Rate",
            f"{overall_click_rate:.2%}",
            (
                f"{total_clicked:,} clicks from "
                f"{total_delivered:,} delivered"
            ),
        )
    # =====================================================
    # CUSTOMER SEARCH
    # =====================================================
    st.divider()
    st.subheader("Customer Affinity")
    st.caption(
        "Search by customer ID to compare engagement "
        "across campaign types."
    )
    search_query = st.text_input(
        "Search Customer ID",
        placeholder="Type a customer ID, e.g. C00 or C0036",
        key="campaign_affinity_search",
    ).strip()
    if not search_query:
        st.info(
            "Enter a customer ID or partial ID to begin. "
            "For example, C00 will show matching customers."
        )
        return
    matching_ids = search_campaign_customers(
        campaigns,
        search_query,
    )
    if not matching_ids:
        st.warning(
            f"No customer IDs found matching '{search_query}'."
        )
        return
    # =====================================================
    # CUSTOMER SELECTION
    # =====================================================
    query_upper = search_query.upper()
    exact_matches = [
        customer_id
        for customer_id in matching_ids
        if customer_id.upper() == query_upper
    ]
    if exact_matches:
        selected_customer_id = exact_matches[0]
    elif len(matching_ids) == 1:
        selected_customer_id = matching_ids[0]
    else:
        st.caption(
            f"{len(matching_ids):,} customer IDs match "
            f"'{search_query}'. Select a customer."
        )
        selected_customer_id = st.selectbox(
            "Matching Customers",
            matching_ids,
            key="campaign_affinity_customer",
        )
    # =====================================================
    # SELECTED CUSTOMER DATA
    # =====================================================
    customer_affinity = get_customer_campaign_affinity(
        campaigns,
        selected_customer_id,
    )
    if customer_affinity.empty:
        st.info(
            f"No campaign affinity data is available "
            f"for {selected_customer_id}."
        )
        return
    # =====================================================
    # CUSTOMER SUMMARY
    # =====================================================
    valid_rates = customer_affinity[
        customer_affinity["click_rate"].notna()
    ].copy()
    if not valid_rates.empty:
        strongest_customer_campaign = (
            valid_rates
            .sort_values(
                [
                    "click_rate",
                    "campaigns_delivered",
                ],
                ascending=[
                    False,
                    False,
                ],
            )
            .iloc[0]
        )
        strongest_customer_type = (
            strongest_customer_campaign["campaign_type"]
        )
    else:
        strongest_customer_type = "—"
    customer_delivered = int(
        customer_affinity[
            "campaigns_delivered"
        ].sum()
    )
    customer_clicked = int(
        customer_affinity[
            "campaigns_clicked"
        ].sum()
    )
    customer_click_rate = (
        customer_clicked / customer_delivered
        if customer_delivered > 0
        else 0
    )
    st.markdown(
        f"### Customer Profile — {selected_customer_id}"
    )
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        _render_card(
            "Customer",
            selected_customer_id,
            "Selected customer profile",
        )
    with col2:
        _render_card(
            "Strongest Affinity",
            strongest_customer_type,
            "Highest click rate with delivered exposure",
        )
    with col3:
        _render_card(
            "Delivered",
            f"{customer_delivered:,}",
            "Campaign messages received",
        )
    with col4:
        _render_card(
            "Overall Click Rate",
            f"{customer_click_rate:.2%}",
            f"{customer_clicked:,} campaign interactions",
        )
    # =====================================================
    # CUSTOMER CHART
    # =====================================================
    st.divider()
    st.subheader("Engagement by Campaign Type")
    st.caption(
        "Click rate is calculated as clicks divided by "
        "delivered campaigns. Exposure is retained alongside "
        "the rate for context."
    )
    chart_data = customer_affinity.copy()
    chart_data["click_rate"] = (
        pd.to_numeric(
            chart_data["click_rate"],
            errors="coerce",
        )
        .fillna(0)
    )
    chart_data = chart_data.sort_values(
        "click_rate",
        ascending=True,
    )
    fig = px.bar(
        chart_data,
        x="click_rate",
        y="campaign_type",
        orientation="h",
        text="click_rate",
        custom_data=[
            "campaigns_delivered",
            "campaigns_clicked",
        ],
    )
    fig.update_traces(
        texttemplate="%{x:.1%}",
    )
    fig = _style_affinity_chart(fig)
    fig.update_layout(
        height=350,
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False,
        },
    )
    # =====================================================
    # DETAILED TABLE
    # =====================================================
    st.subheader("Campaign Affinity Detail")
    st.caption(
        "Campaign exposure and interaction history by "
        "campaign type."
    )
    display_data = customer_affinity.copy()
    display_data["campaigns_sent"] = (
        display_data["campaigns_sent"]
        .map(_format_number)
    )
    display_data["campaigns_delivered"] = (
        display_data["campaigns_delivered"]
        .map(_format_number)
    )
    display_data["campaigns_clicked"] = (
        display_data["campaigns_clicked"]
        .map(_format_number)
    )
    display_data["campaigns_not_clicked"] = (
        display_data["campaigns_not_clicked"]
        .map(_format_number)
    )
    display_data["click_rate"] = (
        display_data["click_rate"]
        .map(_format_percent)
    )
    display_data.columns = [
        "Customer ID",
        "Campaign Type",
        "Sent",
        "Delivered",
        "Clicked",
        "Not Clicked",
        "Click Rate",
    ]
    st.dataframe(
        display_data[
            [
                "Campaign Type",
                "Sent",
                "Delivered",
                "Clicked",
                "Not Clicked",
                "Click Rate",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )