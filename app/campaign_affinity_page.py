import pandas as pd
import plotly.express as px
import streamlit as st

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
    """
    Format numeric values as whole numbers.
    """
    if pd.isna(value):
        return "—"

    return f"{int(value):,}"


def _format_percent(value):
    """
    Format percentage values safely.

    Missing/undefined rates are displayed as an em dash.
    This is important when delivered = 0 because no valid
    click-rate denominator exists.
    """
    if pd.isna(value):
        return "—"

    return f"{float(value):.2%}"


# =========================================================
# CHART HELPERS
# =========================================================

def _style_affinity_chart(fig):
    """
    Apply the common styling used by campaign affinity charts.
    """
    fig.update_layout(
        height=390,
        margin=dict(
            l=20,
            r=20,
            t=45,
            b=20,
        ),
        xaxis_title=None,
        yaxis_title=None,
        legend_title=None,
    )

    return fig


# =========================================================
# OVERVIEW
# =========================================================

def render_affinity_overview(affinity_df):
    """
    Render the overall campaign affinity overview.
    """
    if affinity_df.empty:
        st.info("No campaign affinity data available.")
        return

    total_sent = affinity_df["sent"].sum()
    total_delivered = affinity_df["delivered"].sum()
    total_clicked = affinity_df["clicked"].sum()

    overall_delivery_rate = (
        total_delivered / total_sent
        if total_sent > 0
        else pd.NA
    )

    overall_click_rate = (
        total_clicked / total_delivered
        if total_delivered > 0
        else pd.NA
    )

    strongest_campaign = None

    valid_affinity = affinity_df[
        affinity_df["delivered"] > 0
    ].copy()

    if not valid_affinity.empty:
        strongest_campaign = (
            valid_affinity
            .sort_values(
                by=[
                    "click_rate",
                    "delivered",
                    "clicked",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
            )
            .iloc[0]
        )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Customers",
            f"{affinity_df['customer_id'].nunique():,}",
        )

    with col2:
        st.metric(
            "Campaign Types",
            f"{affinity_df['campaign_type'].nunique():,}",
        )

    with col3:
        st.metric(
            "Overall Click Rate",
            _format_percent(overall_click_rate),
        )

    with col4:
        if strongest_campaign is not None:
            st.metric(
                "Strongest Campaign",
                str(strongest_campaign["campaign_type"]),
                help=(
                    "Campaign type with the highest click rate. "
                    "Delivered exposure is used as the denominator."
                ),
            )
        else:
            st.metric(
                "Strongest Campaign",
                "—",
            )

    if strongest_campaign is not None:
        st.caption(
            f"Highest click rate: "
            f"{_format_percent(strongest_campaign['click_rate'])}"
        )


# =========================================================
# CAMPAIGN TYPE AFFINITY
# =========================================================

def render_campaign_type_affinity(affinity_df):
    """
    Render campaign-type-level affinity analytics.
    """
    st.subheader("Campaign Type Affinity")

    if affinity_df.empty:
        st.info("No campaign type affinity data available.")
        return

    campaign_type_df = (
        calculate_campaign_type_affinity(affinity_df)
    )

    if campaign_type_df.empty:
        st.info("No campaign type affinity data available.")
        return

    display_df = campaign_type_df.copy()

    for column in [
        "sent",
        "delivered",
        "clicked",
    ]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                _format_number
            )

    if "click_rate" in display_df.columns:
        display_df["click_rate"] = display_df[
            "click_rate"
        ].map(_format_percent)

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )

    chart_df = campaign_type_df.copy()

    chart_df["chart_click_rate"] = (
        chart_df["click_rate"].fillna(0)
    )

    fig = px.bar(
        chart_df,
        x="campaign_type",
        y="chart_click_rate",
        text="chart_click_rate",
        title="Click Rate by Campaign Type",
    )

    fig.update_traces(
        texttemplate="%{y:.2%}",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Click Rate: %{y:.2%}"
            "<extra></extra>"
        ),
    )

    fig = _style_affinity_chart(fig)

    st.plotly_chart(
        fig,
        width="stretch",
    )


# =========================================================
# CUSTOMER SEARCH
# =========================================================

def render_customer_search(affinity_df):
    """
    Render customer-level campaign affinity search.
    """
    st.subheader("Customer Campaign Affinity")

    if affinity_df.empty:
        st.info("No campaign affinity data available.")
        return

    matches = search_campaign_customers(
        affinity_df
    )

    if not matches:
        st.info("No customers available.")
        return

    selected_customer = st.selectbox(
        "Customer",
        options=matches,
        key="campaign_affinity_customer_selector",
    )

    customer_affinity = get_customer_campaign_affinity(
        affinity_df,
        selected_customer,
    )

    if customer_affinity.empty:
        st.warning(
            f"No campaign affinity data found for customer "
            f"{selected_customer}."
        )
        return

    total_sent = customer_affinity["sent"].sum()
    total_delivered = customer_affinity["delivered"].sum()
    total_clicked = customer_affinity["clicked"].sum()

    customer_click_rate = (
        total_clicked / total_delivered
        if total_delivered > 0
        else pd.NA
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Campaign Types",
            f"{customer_affinity['campaign_type'].nunique():,}",
        )

    with col2:
        st.metric(
            "Sent",
            _format_number(total_sent),
        )

    with col3:
        st.metric(
            "Delivered",
            _format_number(total_delivered),
        )

    with col4:
        st.metric(
            "Click Rate",
            _format_percent(customer_click_rate),
        )

    display_df = customer_affinity.copy()

    for column in [
        "sent",
        "delivered",
        "clicked",
    ]:
        if column in display_df.columns:
            display_df[column] = display_df[column].map(
                _format_number
            )

    if "click_rate" in display_df.columns:
        display_df["click_rate"] = display_df[
            "click_rate"
        ].map(_format_percent)

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
    )

    chart_df = customer_affinity.copy()

    chart_df["chart_click_rate"] = (
        chart_df["click_rate"].fillna(0)
    )

    fig = px.bar(
        chart_df,
        x="campaign_type",
        y="chart_click_rate",
        text="chart_click_rate",
        title=f"Campaign Affinity — {selected_customer}",
    )

    fig.update_traces(
        texttemplate="%{y:.2%}",
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Click Rate: %{y:.2%}"
            "<extra></extra>"
        ),
    )

    fig = _style_affinity_chart(fig)

    st.plotly_chart(
        fig,
        width="stretch",
    )


# =========================================================
# MAIN PAGE
# =========================================================

def render_campaign_affinity_page(affinity_df=None):
    """
    Render the complete Campaign Affinity page.

    The page supports:
    - Overall campaign affinity
    - Campaign-type affinity
    - Customer-level campaign affinity
    """

    st.title("Campaign Affinity")

    st.caption(
        "Understand which campaign types generate the strongest "
        "customer engagement."
    )

    if affinity_df is None:
        st.warning(
            "Campaign affinity data is not available."
        )
        return

    affinity_df = affinity_df.copy()

    if affinity_df.empty:
        st.info(
            "No campaign affinity data available."
        )
        return

    # -----------------------------------------------------
    # OVERVIEW
    # -----------------------------------------------------

    render_affinity_overview(
        affinity_df
    )

    st.divider()

    # -----------------------------------------------------
    # CAMPAIGN TYPE AFFINITY
    # -----------------------------------------------------

    render_campaign_type_affinity(
        affinity_df
    )

    st.divider()

    # -----------------------------------------------------
    # CUSTOMER SEARCH
    # -----------------------------------------------------

    render_customer_search(
        affinity_df
    )