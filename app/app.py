import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    COLOR_BACKGROUND,
    COLOR_SURFACE,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    COLOR_TEXT_DARK,
    COLOR_BORDER,
    COLOR_GRID,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    COLOR_ALERT,
    COLOR_NEUTRAL,
    RISK_COLORS,
    RETENTION_PRIORITY_COLORS,
    CHART_PALETTE,
)

from data_quality_page import render_data_quality_page
from customer_search_page import render_customer_search_page

from src.analytics.customer import build_customer_analytics



# =========================================================
# CHART STYLE
# =========================================================

CHART_FONT = "Arial"

CHART_LAYOUT = {
    "font": {
        "family": CHART_FONT,
        "color": COLOR_TEXT,
    },
    "paper_bgcolor": COLOR_BACKGROUND,
    "plot_bgcolor": COLOR_BACKGROUND,
    "margin": {
        "l": 40,
        "r": 30,
        "t": 50,
        "b": 50,
    },
    "hoverlabel": {
        "bgcolor": COLOR_SURFACE,
        "font": {
            "color": COLOR_TEXT,
            "family": CHART_FONT,
        },
        "bordercolor": COLOR_BORDER,
    },
}


def apply_chart_style(fig):
    """
    Apply the application's global chart styling.
    """

    fig.update_layout(**CHART_LAYOUT)

    fig.update_xaxes(
        showgrid=False,
        zeroline=False,
        linecolor=COLOR_BORDER,
        tickfont={
            "color": COLOR_TEXT_MUTED,
            "family": CHART_FONT,
        },
        title_font={
            "color": COLOR_TEXT_MUTED,
            "family": CHART_FONT,
        },
    )

    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLOR_GRID,
        zeroline=False,
        linecolor=COLOR_BORDER,
        tickfont={
            "color": COLOR_TEXT_MUTED,
            "family": CHART_FONT,
        },
        title_font={
            "color": COLOR_TEXT_MUTED,
            "family": CHART_FONT,
        },
    )

    return fig

# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Intelligence System",
    page_icon="📊",
    layout="wide",
)


# =========================================================
# DATASET PATHS
# =========================================================

CUSTOMERS_PATH = "data/training/customers.csv"
TRANSACTIONS_PATH = "data/training/transactions.csv"
CAMPAIGNS_PATH = "data/training/campaigns.csv"
FEATURES_PATH = "data/training/customer_features.csv"
MODEL_PATH = "models/random_forest.joblib"


# =========================================================
# LOAD DEMO DATA
# =========================================================

@st.cache_data
def load_demo_data():
    """
    Load the fixed demo datasets used by the application.
    """

    customers = pd.read_csv(CUSTOMERS_PATH)
    transactions = pd.read_csv(TRANSACTIONS_PATH)
    campaigns = pd.read_csv(CAMPAIGNS_PATH)

    return customers, transactions, campaigns


# =========================================================
# LOAD CUSTOMER ANALYTICS
# =========================================================

@st.cache_data
def load_customer_analytics():
    """
    Build the integrated customer analytics dataset.

    This combines:
    - RFM features
    - customer segmentation
    - churn prediction
    - churn probability
    - risk classification
    - campaign engagement
    """

    features = pd.read_csv(FEATURES_PATH)

    analytics = build_customer_analytics(
        features,
        model_path=MODEL_PATH,
    )

    return analytics


# =========================================================
# PREPARE CUSTOMER-LEVEL DATA
# =========================================================

def prepare_customer_search_data(customer_analytics):
    """
    Keep only the latest observation for each customer.

    The feature dataset contains multiple historical
    observations for the same customer. The application
    therefore uses the latest observation when displaying
    current customer-level intelligence.
    """

    result = customer_analytics.copy()

    if "observation_date" not in result.columns:
        raise ValueError(
            "Customer analytics must contain observation_date"
        )

    result["observation_date"] = pd.to_datetime(
        result["observation_date"],
        errors="coerce",
        dayfirst=True,
    )

    if result["observation_date"].isna().all():
        raise ValueError(
            "No valid observation dates found"
        )

    latest_date = result["observation_date"].max()

    result = result[
        result["observation_date"] == latest_date
    ].copy()

    result = (
        result
        .drop_duplicates(
            subset=["customer_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return result


# =========================================================
# CAMPAIGN METRICS
# =========================================================

def calculate_campaign_metrics(campaigns):
    """
    Calculate the overall campaign funnel.
    """

    sent = int(campaigns["sent"].sum())
    delivered = int(campaigns["delivered"].sum())
    clicked = int(campaigns["clicked"].sum())
    redeemed = int(campaigns["redeemed"].sum())

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
        "redemption_rate": redemption_rate,
    }


def calculate_campaign_type_performance(campaigns):
    """
    Calculate campaign performance by campaign type.
    """

    grouped = (
        campaigns
        .groupby("campaign_type")
        .agg(
            campaigns_sent=("sent", "sum"),
            campaigns_delivered=("delivered", "sum"),
            campaigns_clicked=("clicked", "sum"),
            campaigns_redeemed=("redeemed", "sum"),
            total_reward_value=("reward_value", "sum"),
        )
        .reset_index()
    )

    grouped["delivery_rate"] = (
        grouped["campaigns_delivered"]
        / grouped["campaigns_sent"]
    )

    grouped["click_rate"] = (
        grouped["campaigns_clicked"]
        / grouped["campaigns_delivered"]
    )

    grouped["redemption_rate"] = (
        grouped["campaigns_redeemed"]
        / grouped["campaigns_delivered"]
    )

    grouped = grouped.replace(
        [float("inf"), float("-inf")],
        0,
    )

    grouped[
        [
            "delivery_rate",
            "click_rate",
            "redemption_rate",
        ]
    ] = grouped[
        [
            "delivery_rate",
            "click_rate",
            "redemption_rate",
        ]
    ].fillna(0)

    return grouped



# =========================================================
# OVERVIEW
# =========================================================

def render_overview(
    customers,
    transactions,
    campaigns,
    customer_analytics,
):
    """
    Render the executive customer intelligence dashboard.
    """

    # -----------------------------------------------------
    # Latest customer-level observations
    # -----------------------------------------------------

    customer_data = prepare_customer_search_data(
        customer_analytics
    )

    # -----------------------------------------------------
    # Business metrics from source files
    # -----------------------------------------------------

    total_customers = len(customers)
    total_transactions = len(transactions)
    total_campaigns = len(campaigns)

    total_revenue = pd.to_numeric(
        transactions["bill_amount"],
        errors="coerce",
    ).sum()

    average_bill = pd.to_numeric(
        transactions["bill_amount"],
        errors="coerce",
    ).mean()

    # -----------------------------------------------------
    # Customer health from churn analytics
    # -----------------------------------------------------

    churned_customers = int(
        (
            customer_data["churn_prediction"]
            == 1
        ).sum()
    )

    very_high_risk = int(
        (
            customer_data["risk_level"]
            == "Very High"
        ).sum()
    )

    high_risk = int(
        (
            customer_data["risk_level"]
            == "High"
        ).sum()
    )

    medium_risk = int(
        (
            customer_data["risk_level"]
            == "Medium"
        ).sum()
    )

    overall_churn_probability = (
        customer_data["churn_probability"].mean()
    )

    # =====================================================
    # HEADER
    # =====================================================

    st.title(
        "Customer Churn Prediction & Campaign Analytics"
    )

    st.write(
        "An integrated view of customer behavior, "
        "segmentation, churn risk and campaign performance."
    )

    st.divider()


    # =========================================================
    # REUSABLE KPI CARD
    # =========================================================

    def render_kpi_card(
        title,
        value,
        description,
        accent_color,
    ):
        """Render a compact KPI card using native Streamlit components."""

        with st.container(border=True):
            st.markdown(
                f"""
                <div style="
                    border-top: 2px solid {accent_color};
                    margin: -1rem -1rem 0 -1rem;
                    padding-top: 2px;
                "></div>
                """,
                unsafe_allow_html=True,
            )

            st.caption(title)
            st.markdown(f"### {value}")
            st.caption(description)


    # =========================================================
    # BUSINESS OVERVIEW
    # =========================================================

    st.subheader("Business Overview")

    col1, col2, col3, col4 = st.columns(4, gap="small")

    with col1:
        render_kpi_card(
            "CUSTOMERS",
            f"{total_customers:,}",
            "Registered customers",
            COLOR_PRIMARY,
        )

    with col2:
        render_kpi_card(
            "TRANSACTIONS",
            f"{total_transactions:,}",
            "Recorded transactions",
            COLOR_PRIMARY,
        )

    with col3:
        render_kpi_card(
            "CAMPAIGNS",
            f"{total_campaigns:,}",
            "Campaign records",
            COLOR_PRIMARY,
        )

    with col4:
        render_kpi_card(
            "TOTAL REVENUE",
            f"₹{total_revenue:,.0f}",
            "Revenue from transactions",
            COLOR_PRIMARY,
        )


    # =========================================================
    # CUSTOMER HEALTH
    # =========================================================

    st.markdown(
        "<div style='height: 24px;'></div>",
        unsafe_allow_html=True,
    )

    st.subheader("Customer Health")

    col1, col2, col3, col4 = st.columns(4, gap="small")

    with col1:
        render_kpi_card(
            "CHURNED CUSTOMERS",
            f"{churned_customers:,}",
            "Predicted churn customers",
            COLOR_ALERT,
        )

    with col2:
        render_kpi_card(
            "VERY HIGH RISK",
            f"{very_high_risk:,}",
            "Highest churn probability tier",
            COLOR_ALERT,
        )

    with col3:
        render_kpi_card(
            "HIGH RISK",
            f"{high_risk:,}",
            "High churn probability tier",
            COLOR_TERTIARY,
        )

    with col4:
        render_kpi_card(
            "AVG. CHURN PROBABILITY",
            f"{overall_churn_probability:.1%}",
            "Average model probability",
            COLOR_PRIMARY,
        )


    # =====================================================
    # CUSTOMER SEGMENT + RISK DISTRIBUTION
    # =====================================================

    st.divider()

    st.subheader("Customer Intelligence")

    col1, col2 = st.columns(2)

    # -----------------------------------------------------
    # Segment Distribution
    # -----------------------------------------------------

    with col1:

        st.markdown("#### Customer Segments")

        if "customer_segment" in customer_data.columns:

            segment_counts = (
                customer_data["customer_segment"]
                .value_counts()
                .reset_index()
            )

            segment_counts.columns = [
                "Segment",
                "Customers",
            ]

            fig = px.bar(
                segment_counts,
                x="Segment",
                y="Customers",
                text="Customers",
                color_discrete_sequence=[COLOR_PRIMARY],
            )

            fig = apply_chart_style(fig)

            fig.update_layout(
                xaxis_title=None,
                yaxis_title="Customers",
                showlegend=False,
                height=380,
            )

            fig.update_traces(
                textposition="outside",
                marker_line_width=0,
                textfont_color=COLOR_TEXT,
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
            )

        else:
            st.warning(
                "Customer segmentation data is unavailable."
            )

    # -----------------------------------------------------
    # Risk Distribution
    # -----------------------------------------------------

    with col2:

        st.markdown("#### Churn Risk Distribution")

        risk_counts = (
            customer_data["risk_level"]
            .value_counts()
            .reindex(
                [
                    "Low",
                    "Medium",
                    "High",
                    "Very High",
                ],
                fill_value=0,
            )
            .reset_index()
        )

        risk_counts.columns = [
            "Risk Level",
            "Customers",
        ]

        fig = px.bar(
            risk_counts,
            x="Risk Level",
            y="Customers",
            text="Customers",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
        )

        fig = apply_chart_style(fig)

        fig.update_layout(
            xaxis_title=None,
            yaxis_title="Customers",
            showlegend=False,
            height=380,
        )

        fig.update_traces(
            textposition="outside",
            marker_line_width=0,
            textfont_color=COLOR_TEXT,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # =====================================================
    # SEGMENT × RISK
    # =====================================================

    if "customer_segment" in customer_data.columns:

        st.divider()

        st.subheader(
            "Segment × Churn Risk"
        )

        segment_risk = (
            customer_data
            .groupby(
                [
                    "customer_segment",
                    "risk_level",
                ]
            )
            .size()
            .reset_index(
                name="Customers"
            )
        )

        fig = px.bar(
            segment_risk,
            x="customer_segment",
            y="Customers",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            barmode="stack",
        )

        fig.update_traces(
            marker_line_width=0,
        )

        fig = apply_chart_style(fig)

        fig.update_layout(
            xaxis_title="Customer Segment",
            yaxis_title="Customers",
            legend_title="Risk Level",
            height=430,
        )

        st.plotly_chart(
            fig,
            use_container_width=True,
        )

    # =====================================================
    # CAMPAIGN PERFORMANCE
    # =====================================================

    st.divider()

    st.subheader(
        "Campaign Performance"
    )

    campaign_metrics = calculate_campaign_metrics(
        campaigns
    )

    # -----------------------------------------------------
    # Funnel metrics
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Sent",
            f"{campaign_metrics['sent']:,}",
        )

    with col2:
        st.metric(
            "Delivered",
            f"{campaign_metrics['delivered']:,}",
            f"{campaign_metrics['delivery_rate']:.2%}",
        )

    with col3:
        st.metric(
            "Clicked",
            f"{campaign_metrics['clicked']:,}",
            f"{campaign_metrics['click_rate']:.2%}",
        )

    with col4:
        st.metric(
            "Redeemed",
            f"{campaign_metrics['redeemed']:,}",
            f"{campaign_metrics['redemption_rate']:.2%}",
        )

    # -----------------------------------------------------
    # Campaign funnel chart
    # -----------------------------------------------------

    funnel_data = pd.DataFrame(
        {
            "Stage": [
                "Sent",
                "Delivered",
                "Clicked",
                "Redeemed",
            ],
            "Customers": [
                campaign_metrics["sent"],
                campaign_metrics["delivered"],
                campaign_metrics["clicked"],
                campaign_metrics["redeemed"],
            ],
        }
    )

    fig = px.funnel(
        funnel_data,
        x="Customers",
        y="Stage",
    )

    fig.update_traces(
        marker_color=COLOR_PRIMARY,
        marker_line_width=0,
        textfont_color=COLOR_TEXT_DARK,
    )

    fig = apply_chart_style(fig)

    fig.update_layout(
        height=420,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # =========================================================
    # CAMPAIGN TYPE PERFORMANCE
    # =========================================================

    st.divider()

    st.subheader(
        "Campaign Performance by Type"
    )

    st.caption(
        "Compare campaign reach, engagement and redemption across campaign types."
    )

    campaign_type_data = (
        calculate_campaign_type_performance(
            campaigns
        )
    )

    display_campaign_data = (
        campaign_type_data.copy()
    )

    # ---------------------------------------------------------
    # Format display values
    # ---------------------------------------------------------

    display_campaign_data[
        "delivery_rate"
    ] = (
        display_campaign_data[
            "delivery_rate"
        ].map(
            lambda x: f"{x:.2%}"
        )
    )

    display_campaign_data[
        "click_rate"
    ] = (
        display_campaign_data[
            "click_rate"
        ].map(
            lambda x: f"{x:.2%}"
        )
    )

    display_campaign_data[
        "redemption_rate"
    ] = (
        display_campaign_data[
            "redemption_rate"
        ].map(
            lambda x: f"{x:.2%}"
        )
    )

    display_campaign_data[
        "total_reward_value"
    ] = (
        display_campaign_data[
            "total_reward_value"
        ].map(
            lambda x: f"₹{x:,.0f}"
        )
    )

    display_campaign_data.columns = [
        "Campaign Type",
        "Sent",
        "Delivered",
        "Clicked",
        "Redeemed",
        "Reward Value",
        "Delivery Rate",
        "Click Rate",
        "Redemption Rate",
    ]

    # ---------------------------------------------------------
    # Keep the full table, but make it feel like a dashboard
    # component rather than a raw dataframe.
    # ---------------------------------------------------------

    styled_campaign_data = (
        display_campaign_data
        .style
        .set_properties(
            **{
                "background-color": COLOR_SURFACE,
                "color": COLOR_TEXT,
                "border-color": COLOR_BORDER,
                "font-size": "13px",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", COLOR_BACKGROUND),
                        ("color", COLOR_TEXT_MUTED),
                        ("font-weight", "600"),
                        ("font-size", "11px"),
                        ("letter-spacing", "0.2px"),
                        ("border-bottom", f"1px solid {COLOR_BORDER}"),
                        ("padding", "10px 12px"),
                        ("text-align", "left"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("padding", "9px 12px"),
                        ("border-bottom", f"1px solid {COLOR_GRID}"),
                    ],
                },
                {
                    "selector": "tbody tr:nth-child(even) td",
                    "props": [
                        ("background-color", COLOR_BACKGROUND),
                    ],
                },
                {
                    "selector": "tbody tr:hover td",
                    "props": [
                        ("background-color", COLOR_BORDER),
                    ],
                },
            ]
        )
        .set_properties(
            subset=["Campaign Type"],
            **{
                "font-weight": "600",
                "color": COLOR_TEXT,
            }
        )
        .set_properties(
            subset=[
                "Delivery Rate",
                "Click Rate",
                "Redemption Rate",
            ],
            **{
                "font-weight": "600",
                "color": COLOR_TEXT_MUTED,
            }
        )
    )

    with st.container(border=True):
        st.dataframe(
            styled_campaign_data,
            use_container_width=True,
            hide_index=True,
            height=275,
        )

    # =========================================================
    # BUSINESS INSIGHTS
    # =========================================================

    st.divider()

    st.subheader("Key Business Insights")

    # ---------------------------------------------------------
    # Calculate insights
    # ---------------------------------------------------------

    if "customer_segment" in customer_data.columns:

        segment_counts = (
            customer_data["customer_segment"]
            .value_counts()
        )

        top_segment = segment_counts.idxmax()
        top_segment_count = segment_counts.max()

    else:
        top_segment = "Unavailable"
        top_segment_count = 0


    best_campaign = (
        campaign_type_data
        .sort_values(
            "redemption_rate",
            ascending=False,
        )
        .iloc[0]
    )


    customers_requiring_attention = (
        high_risk
        + very_high_risk
        + churned_customers
    )


    # ---------------------------------------------------------
    # Insight cards — Row 1
    # ---------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            f"""
            <div style="
                background-color: {COLOR_SURFACE};
                border: 1px solid {COLOR_BORDER};
                border-top: 3px solid {COLOR_PRIMARY};
                border-radius: 8px;
                padding: 20px;
                min-height: 145px;
            ">

            <div style="
                color: {COLOR_TEXT_MUTED};
                font-size: 13px;
                font-weight: 600;
                margin-bottom: 10px;
            ">
            LARGEST CUSTOMER SEGMENT
            </div>

            <div style="
                color: {COLOR_TEXT};
                font-size: 24px;
                font-weight: 600;
                margin-bottom: 8px;
            ">
            {top_segment}
            </div>

            <div style="
                color: {COLOR_TEXT_MUTED};
                font-size: 14px;
            ">
            {top_segment_count:,} current customer profiles
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with col2:

        st.markdown(
            f"""
            <div style="
                background-color: {COLOR_SURFACE};
                border: 1px solid {COLOR_BORDER};
                border-top: 3px solid {COLOR_ALERT};
                border-radius: 8px;
                padding: 20px;
                min-height: 145px;
            ">

            <div style="
                color: {COLOR_TEXT_MUTED};
                font-size: 13px;
                font-weight: 600;
                margin-bottom: 10px;
            ">
            CUSTOMERS REQUIRING ATTENTION
            </div>

            <div style="
                color: {COLOR_TEXT};
                font-size: 24px;
                font-weight: 600;
                margin-bottom: 8px;
            ">
            {customers_requiring_attention:,}
            </div>

            <div style="
                color: {COLOR_TEXT_MUTED};
                font-size: 14px;
            ">
            High, Very High risk or predicted churn
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    # ---------------------------------------------------------
    # Insight cards — Row 2
    # ---------------------------------------------------------

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)


    with col1:

        st.markdown(
            f"""
            <div style="
                background-color: {COLOR_SURFACE};
                border: 1px solid {COLOR_BORDER};
                border-top: 3px solid {COLOR_SECONDARY};
                border-radius: 8px;
                padding: 20px;
                min-height: 145px;
            ">

            <div style="
                color: {COLOR_TEXT_MUTED};
                font-size: 13px;
                font-weight: 600;
                margin-bottom: 10px;
            ">
            BEST CAMPAIGN TYPE
            </div>

            <div style="
                color: {COLOR_TEXT};
                font-size: 24px;
                font-weight: 600;
                margin-bottom: 8px;
            ">
            {best_campaign['campaign_type']}
            </div>

            <div style="
                color: {COLOR_TEXT_MUTED};
                font-size: 14px;
            ">
            {best_campaign['redemption_rate']:.2%} redemption rate
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )


    with col2:

        st.markdown(
            f"""
            <div style="
                background-color: {COLOR_SURFACE};
                border: 1px solid {COLOR_BORDER};
                border-top: 3px solid {COLOR_NEUTRAL};
                border-radius: 8px;
                padding: 20px;
                min-height: 145px;
            ">

            <div style="
                color: {COLOR_TEXT_MUTED};
                font-size: 13px;
                font-weight: 600;
                margin-bottom: 10px;
            ">
            AVERAGE TRANSACTION VALUE
            </div>

            <div style="
                color: {COLOR_TEXT};
                font-size: 24px;
                font-weight: 600;
                margin-bottom: 8px;
            ">
            ₹{average_bill:,.2f}
            </div>

            <div style="
                color: {COLOR_TEXT_MUTED};
                font-size: 14px;
            ">
            Average value per transaction
            </div>

            </div>
            """,
            unsafe_allow_html=True,
        )   

# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    # =====================================================
    # SIDEBAR
    # =====================================================

    st.sidebar.title(
        "Customer Intelligence"
    )

    st.sidebar.caption(
        "Customer analytics and campaign intelligence"
    )

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Data Quality",
            "Customer Search",
        ],
    )

    # =====================================================
    # LOAD DATA
    # =====================================================

    customers, transactions, campaigns = (
        load_demo_data()
    )

    # =====================================================
    # OVERVIEW
    # =====================================================

    if page == "Overview":

        with st.spinner(
            "Preparing customer intelligence..."
        ):

            customer_analytics = (
                load_customer_analytics()
            )

        render_overview(
            customers,
            transactions,
            campaigns,
            customer_analytics,
        )

    # =====================================================
    # DATA QUALITY
    # =====================================================

    elif page == "Data Quality":

        render_data_quality_page(
            customers,
            transactions,
            campaigns,
        )

    # =====================================================
    # CUSTOMER SEARCH
    # =====================================================

    elif page == "Customer Search":

        with st.spinner(
            "Preparing customer analytics..."
        ):

            customer_analytics = (
                load_customer_analytics()
            )

        search_data = (
            prepare_customer_search_data(
                customer_analytics
            )
        )

        render_customer_search_page(
            search_data
        )


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
