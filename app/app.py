import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    COLOR_BACKGROUND,
    COLOR_SURFACE,
    COLOR_SURFACE_ALT,
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
    COLOR_WHITE,
    CHART_PALETTE,
    RISK_COLORS,
    RETENTION_PRIORITY_COLORS,
)

from data_quality_page import render_data_quality_page
from customer_search_page import render_customer_search_page
from data_upload_page import (
    initialize_dataset_state,
    render_data_upload_page,
)

from src.analytics.customer import build_customer_analytics
from src.analytics.campaign_affinity import (
    calculate_campaign_affinity,
    calculate_campaign_type_affinity,
)
from src.data.custom_pipeline import build_custom_customer_analytics
from src.ml.feature_importance import load_random_forest_feature_importance


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
        "l": 45,
        "r": 25,
        "t": 30,
        "b": 45,
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
    layout="wide",
)

# Keep the entire application visually grayscale, including Streamlit
# navigation controls and the selected sidebar state.
st.markdown(
    """
    <style>
        :root {
            --bg: #0E1117;
            --surface: #161A1F;
            --surface-alt: #1E2329;
            --border: #30343A;
            --text: #E8E8E8;
            --muted: #92979D;
            --blue: #71879C;
            --red: #986B70;
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stMain"] {
            background: var(--bg);
        }

        [data-testid="stHeader"] {
            background: #0E1117;
        }

        /* =====================================================
           SIDEBAR
           ===================================================== */

        [data-testid="stSidebar"] {
            background: #0E1117;
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 1.05rem;
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
            color: var(--muted);
        }

        .sidebar-section-label {
            color: var(--muted);
            font-size: 9px;
            font-weight: 700;
            letter-spacing: .13em;
            text-transform: uppercase;
            margin: 8px 0 9px 1px;
        }

        .sidebar-divider {
            height: 1px;
            background: var(--border);
            margin: 11px 0;
        }

        /* Approved render: full-size buttons, compact gaps */
        [data-testid="stSidebar"] .stButton {
            margin: 0 0 4px 0;
        }

        [data-testid="stSidebar"] .stButton > button {
            min-height: 44px !important;
            height: 44px !important;
            padding: 0.55rem 0.85rem !important;
            justify-content: center !important;
            text-align: center !important;
            background: var(--surface) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 7px !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            letter-spacing: .005em !important;
            box-shadow: none !important;
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            background: var(--surface-alt) !important;
            border-color: #30343A !important;
            color: var(--text) !important;
        }

        [data-testid="stSidebar"] .stButton > button[kind="primary"],
        [data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
            background: #1E2329 !important;
            border-color: var(--blue) !important;
            color: #E8E8E8 !important;
        }

        /* Home button */
        .sidebar-home .stButton {
            margin-bottom: 0 !important;
        }

        .sidebar-home .stButton > button {
            min-height: 46px !important;
            height: 46px !important;
            font-size: 12px !important;
            font-weight: 500 !important;
            background: #1E2329 !important;
            border-color: var(--blue) !important;
        }

        /* Slim current-dataset indicator */
        .sidebar-dataset {
            height: 31px;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            padding: 0 10px;
            border: 1px solid var(--border);
            border-radius: 6px;
            background: var(--surface);
            color: var(--muted);
            font-size: 9px;
            font-weight: 700;
            letter-spacing: .06em;
            margin: 0 0 12px 0;
        }

        .sidebar-dataset-dot {
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--blue);
            margin-right: 8px;
            flex: 0 0 auto;
        }

        /* =====================================================
           CONTENT
           ===================================================== */

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: var(--surface);
            border-color: var(--border) !important;
            border-radius: 8px;
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div,
        textarea {
            background: var(--surface) !important;
            border-color: var(--border) !important;
            color: var(--text) !important;
        }

        input, textarea {
            color: var(--text) !important;
        }

        .stButton > button,
        .stDownloadButton > button {
            background: var(--surface-alt) !important;
            color: var(--text) !important;
            border: 1px solid var(--border) !important;
            border-radius: 6px !important;
            font-weight: 600;
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--blue) !important;
            color: #E8E8E8 !important;
        }

        .stButton > button[kind="primary"],
        button[data-testid="baseButton-primary"] {
            background: var(--blue) !important;
            color: #0E1117 !important;
            border-color: var(--blue) !important;
        }

        [data-testid="stAlert"] {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
        }

        [data-testid="stMetricValue"] {
            color: var(--text) !important;
            font-weight: 700;
        }

        [data-testid="stMetricLabel"],
        .stCaption,
        [data-testid="stCaptionContainer"] {
            color: var(--muted) !important;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
        }

        button[data-baseweb="tab"] {
            color: var(--muted) !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: var(--text) !important;
            border-bottom-color: var(--blue) !important;
        }

        [data-testid="stFileUploader"] section {
            background: var(--surface) !important;
            border: 1px dashed var(--border) !important;
            border-radius: 8px;
        }

        hr {
            border-color: var(--border) !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
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
# ACTIVE DATASET
# =========================================================

def get_active_dataset():
    """Return the raw datasets currently selected by the user."""

    active_dataset = st.session_state.get(
        "active_dataset",
        "Demo Dataset",
    )

    custom_data = st.session_state.get("custom_data")

    if active_dataset == "Custom Data" and custom_data:
        return (
            custom_data["customers"],
            custom_data["transactions"],
            custom_data["campaigns"],
            "Custom Data",
        )

    customers, transactions, campaigns = load_demo_data()

    return (
        customers,
        transactions,
        campaigns,
        "Demo Dataset",
    )


@st.cache_data(show_spinner=False)
def load_custom_customer_analytics(
    customers,
    transactions,
    campaigns,
):
    """Build prediction and retention analytics for custom data."""

    custom_data = {
        "customers": customers,
        "transactions": transactions,
        "campaigns": campaigns,
    }

    return build_custom_customer_analytics(
        custom_data,
        model_path=MODEL_PATH,
    )


def load_active_customer_analytics(
    customers,
    transactions,
    campaigns,
    data_mode,
):
    """Return analytics for the currently active dataset."""

    if data_mode == "Custom Data":
        return load_custom_customer_analytics(
            customers,
            transactions,
            campaigns,
        )

    return load_customer_analytics()


# =========================================================
# LOAD MODEL FEATURE IMPORTANCE
# =========================================================

@st.cache_data
def load_feature_importance():
    """
    Load feature importance from the selected Random Forest model.
    """

    return load_random_forest_feature_importance(
        model_path=MODEL_PATH,
    )


# =========================================================
# PREPARE CUSTOMER-LEVEL DATA
# =========================================================

def prepare_customer_search_data(customer_analytics, customers=None):
    """
    Keep only the latest observation for each customer and, when
    available, attach raw customer profile fields used by Customer Search.
    """

    result = customer_analytics.copy()

    if customers is not None and "customer_id" in customers.columns and "customer_id" in result.columns:
        profile_columns = [
            column
            for column in ["gender", "age", "city", "registration_date"]
            if column in customers.columns
        ]
        if profile_columns:
            profile = customers[["customer_id"] + profile_columns].copy()
            profile["customer_id"] = profile["customer_id"].astype(str).str.strip()
            result["customer_id"] = result["customer_id"].astype(str).str.strip()
            result = result.drop(
                columns=[column for column in profile_columns if column in result.columns],
                errors="ignore",
            ).merge(profile, on="customer_id", how="left")


    # Demo analytics use observation_date. Custom prediction analytics
    # use the prediction observation date if needed.
    if "observation_date" not in result.columns:
        if "prediction_observation_date" in result.columns:
            result["observation_date"] = result[
                "prediction_observation_date"
            ]
        else:
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
# REUSABLE UI COMPONENTS
# =========================================================

def render_kpi_card(title, value, description, accent_color=COLOR_PRIMARY):
    """Render a compact product-style KPI card."""
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="height:3px;background:{accent_color};margin:-1rem -1rem 0 -1rem;
                        border-radius:8px 8px 0 0;"></div>
            <div style="padding-top:8px;">
                <div style="font-size:11px;font-weight:700;letter-spacing:.08em;
                            color:{COLOR_TEXT_MUTED};">{title}</div>
                <div style="font-size:25px;font-weight:700;line-height:1.25;
                            color:{COLOR_WHITE};margin-top:5px;">{value}</div>
                <div style="font-size:11px;color:{COLOR_TEXT_MUTED};margin-top:5px;">
                    {description}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_page_header(title, description):
    """Render the consistent product header used across analytical pages."""
    active_dataset = st.session_state.get("active_dataset", "Demo Dataset")
    badge_color = COLOR_PRIMARY if active_dataset == "Custom Data" else COLOR_TERTIARY

    left, right = st.columns([5.5, 1.5], vertical_alignment="center")
    with left:
        st.markdown(
            f'<div style="font-size:31px;font-weight:750;letter-spacing:-.03em;">{title}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-size:13px;color:{COLOR_TEXT_MUTED};margin-top:4px;">{description}</div>',
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            f"""<div style="text-align:right;margin-top:4px;">
                <span style="display:inline-block;padding:5px 10px;border-radius:14px;
                background:{COLOR_SURFACE_ALT};border:1px solid {COLOR_BORDER};
                color:{badge_color};font-size:11px;font-weight:700;">
                {active_dataset.upper()}
                </span>
            </div>""",
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div style="height:1px;background:{COLOR_BORDER};margin:16px 0 20px;"></div>',
        unsafe_allow_html=True,
    )


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

def render_executive_overview(
    customers,
    transactions,
    campaigns,
    customer_analytics,
    data_mode,
):
    """Render the executive landing page with two clean KPI sections."""

    customer_data = prepare_customer_search_data(customer_analytics, customers)

    # =====================================================
    # SOURCE-FILE BUSINESS METRICS
    # =====================================================

    total_customers = len(customers)
    total_transactions = len(transactions)
    total_campaigns = len(campaigns)

    total_revenue = pd.to_numeric(
        transactions["bill_amount"],
        errors="coerce",
    ).sum()

    # =====================================================
    # CUSTOMER HEALTH METRICS
    # =====================================================

    churned_customers = int(
        (customer_data["churn_prediction"] == 1).sum()
    )
    churn_label = "PREDICTED CHURN" if data_mode == "Custom Data" else "CHURNED CUSTOMERS"
    churn_description = (
        "Customers predicted to churn by the model"
        if data_mode == "Custom Data"
        else "Customers classified as churned"
    )

    very_high_risk = int(
        (customer_data["risk_level"] == "Very High").sum()
    )

    high_risk = int(
        (customer_data["risk_level"] == "High").sum()
    )

    overall_churn_probability = customer_data["churn_probability"].mean()

    # =====================================================
    # HEADER
    # =====================================================

    render_page_header(
        "Customer Churn Prediction & Campaign Analytics",
        "An integrated view of customer behavior, segmentation, churn risk and campaign performance.",
    )

    # =====================================================
    # BUSINESS OVERVIEW
    # =====================================================

    st.subheader("Business Overview")
    st.caption("Key metrics from the registered customer, transaction and campaign records.")

    col1, col2, col3, col4 = st.columns(4, gap="medium")

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

    # Deliberate separation between source metrics and ML metrics.
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    # =====================================================
    # CUSTOMER HEALTH
    # =====================================================

    st.subheader("Customer Health")
    st.caption("Churn prediction and probability-based risk indicators from the selected Random Forest model.")

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        render_kpi_card(
            churn_label,
            f"{churned_customers:,}",
            churn_description,
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
            COLOR_SECONDARY,
        )

    with col4:
        render_kpi_card(
            "AVG. CHURN PROBABILITY",
            f"{overall_churn_probability:.1%}",
            "Average model probability",
            COLOR_PRIMARY,
        )


# =========================================================
# CUSTOMER INTELLIGENCE
# =========================================================

def render_customer_segmentation(customer_analytics, customers, data_mode):
    """Render customer segmentation and churn-risk intelligence."""

    customer_data = prepare_customer_search_data(
        customer_analytics
    )

    render_page_header(
        "Customer Intelligence",
        "Understand customer segments, churn exposure, retention priority and the model's strongest signals.",
    )

    st.caption(
        f"Data source: {data_mode}."
    )
    st.caption(
        f"Analysis uses the latest available observation for each customer profile ({len(customer_data):,} current profiles)."
    )

    # =====================================================
    # CUSTOMER SEGMENTS + CHURN RISK
    # =====================================================

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("Customer Segments")

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
            segment_counts.sort_values(
                "Customers",
                ascending=True,
            ),
            x="Customers",
            y="Segment",
            orientation="h",
            text="Customers",
            color_discrete_sequence=[COLOR_PRIMARY],
        )

        fig = apply_chart_style(fig)
        fig.update_layout(
            xaxis_title="Customers",
            yaxis_title=None,
            showlegend=False,
            height=360,
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

    with col2:
        st.subheader("Churn Risk Distribution")

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
            risk_counts.sort_values(
                "Customers",
                ascending=True,
            ),
            x="Customers",
            y="Risk Level",
            orientation="h",
            text="Customers",
            color="Risk Level",
            color_discrete_map=RISK_COLORS,
        )

        fig = apply_chart_style(fig)
        fig.update_layout(
            xaxis_title="Customers",
            yaxis_title=None,
            showlegend=False,
            height=360,
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
    # SEGMENT × CHURN RISK
    # =====================================================

    st.divider()
    st.subheader("Segment × Churn Risk")
    st.caption(
        "Composition of each customer segment across the four predicted churn-risk tiers."
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
        .reset_index(name="Customers")
    )

    segment_order = [
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "Regular Customers",
        "At Risk",
        "Inactive",
        "Lost",
    ]

    segment_risk["customer_segment"] = pd.Categorical(
        segment_risk["customer_segment"],
        categories=segment_order,
        ordered=True,
    )

    risk_order = [
        "Low",
        "Medium",
        "High",
        "Very High",
    ]

    segment_risk["risk_level"] = pd.Categorical(
        segment_risk["risk_level"],
        categories=risk_order,
        ordered=True,
    )

    fig = px.bar(
        segment_risk.sort_values(
            ["customer_segment", "risk_level"]
        ),
        x="Customers",
        y="customer_segment",
        color="risk_level",
        color_discrete_map=RISK_COLORS,
        orientation="h",
        barmode="stack",
    )

    fig = apply_chart_style(fig)
    fig.update_layout(
        xaxis_title="Customers",
        yaxis_title=None,
        legend_title="Risk Level",
        height=390,
    )
    fig.update_traces(
        marker_line_width=0,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

# =========================================================
# RETENTION & MODEL INSIGHTS
# =========================================================

def render_retention_model_insights(customer_analytics, customers, data_mode):
    """Render model drivers followed by retention-priority analytics."""

    customer_data = prepare_customer_search_data(customer_analytics, customers)

    render_page_header(
        "Retention & Model Insights",
        "Prioritize customers for retention and understand the strongest signals used by the churn model.",
    )

    st.caption(f"Data source: {data_mode}.")

    # =====================================================
    # CHURN DRIVERS — FULL WIDTH
    # =====================================================

    st.subheader("Churn Drivers")
    st.caption(
        "Random Forest feature importance — model signal strength, not causal impact."
    )

    feature_importance = load_feature_importance().copy()

    top_features = (
        feature_importance
        .head(10)
        .sort_values("importance", ascending=True)
        .copy()
    )

    top_features["importance_pct"] = top_features["importance"] * 100

    fig = px.bar(
        top_features,
        x="importance_pct",
        y="feature",
        orientation="h",
        text="importance_pct",
        color_discrete_sequence=[COLOR_PRIMARY],
    )

    fig = apply_chart_style(fig)
    fig.update_layout(
        xaxis_title="Importance (%)",
        yaxis_title=None,
        showlegend=False,
        height=500,
        margin={"l": 45, "r": 50, "t": 25, "b": 55},
    )
    fig.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        marker_line_width=0,
        textfont_color=COLOR_TEXT,
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("View complete feature ranking"):
        feature_table = feature_importance.copy()
        feature_table["importance"] = feature_table["importance"].map(
            lambda x: f"{x:.2%}"
        )
        feature_table.columns = ["Rank", "Feature", "Importance"]
        st.dataframe(
            feature_table,
            use_container_width=True,
            hide_index=True,
        )

    # =====================================================
    # RETENTION PRIORITY — FULL WIDTH
    # =====================================================

    st.divider()
    st.subheader("Retention Priority")
    st.caption(
        "Prioritizes customers using predicted churn probability and relative customer value."
    )

    required_priority_columns = {
        "retention_priority",
        "retention_priority_score",
        "expected_revenue_at_risk",
    }

    if not required_priority_columns.issubset(customer_data.columns):
        st.info("Retention priority analytics are unavailable.")
        return

    priority_order = ["Critical", "High", "Medium", "Low"]

    priority_counts = (
        customer_data["retention_priority"]
        .value_counts()
        .reindex(priority_order, fill_value=0)
        .reset_index()
    )
    priority_counts.columns = ["Retention Priority", "Customers"]

    critical_count = int(
        (customer_data["retention_priority"] == "Critical").sum()
    )

    high_priority_count = int(
        customer_data["retention_priority"].isin(["Critical", "High"]).sum()
    )

    revenue_at_risk = pd.to_numeric(
        customer_data["expected_revenue_at_risk"],
        errors="coerce",
    ).fillna(0).sum()

    average_priority_score = pd.to_numeric(
        customer_data["retention_priority_score"],
        errors="coerce",
    ).mean()

    # -----------------------------------------------------
    # Four distinct retention KPIs
    # -----------------------------------------------------

    kpi_1, kpi_2, kpi_3, kpi_4 = st.columns(4, gap="medium")

    with kpi_1:
        render_kpi_card(
            "CRITICAL",
            f"{critical_count:,}",
            "Highest retention priority",
            COLOR_ALERT,
        )

    with kpi_2:
        render_kpi_card(
            "EXPECTED REVENUE AT RISK",
            f"₹{revenue_at_risk:,.0f}",
            "Probability-weighted customer value",
            COLOR_PRIMARY,
        )

    with kpi_3:
        render_kpi_card(
            "AVG. PRIORITY SCORE",
            f"{average_priority_score:.1f}",
            "Average retention priority score",
            COLOR_SECONDARY,
        )

    with kpi_4:
        render_kpi_card(
            "HIGH + CRITICAL",
            f"{high_priority_count:,}",
            "Customers requiring priority action",
            COLOR_TERTIARY,
        )

    # -----------------------------------------------------
    # Full-width priority distribution
    # -----------------------------------------------------

    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)
    st.markdown("**Priority Distribution**")

    fig = px.bar(
        priority_counts.sort_values("Customers", ascending=True),
        x="Customers",
        y="Retention Priority",
        orientation="h",
        text="Customers",
        color="Retention Priority",
        color_discrete_map=RETENTION_PRIORITY_COLORS,
    )

    fig = apply_chart_style(fig)
    fig.update_layout(
        xaxis_title="Customers",
        yaxis_title=None,
        showlegend=False,
        height=480,
        margin={"l": 35, "r": 55, "t": 25, "b": 55},
    )
    fig.update_traces(
        textposition="outside",
        marker_line_width=0,
        textfont_color=COLOR_TEXT,
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# CAMPAIGN PERFORMANCE
# =========================================================

def render_campaign_performance(campaigns):
    """Render campaign funnel and campaign-type performance table."""

    render_page_header(
        "Campaign Performance",
        "Measure campaign reach, engagement and redemption across the overall funnel and individual campaign types.",
    )

    campaign_metrics = calculate_campaign_metrics(
        campaigns
    )

    # =====================================================
    # CAMPAIGN KPIs
    # =====================================================

    col1, col2, col3, col4 = st.columns(4, gap="small")

    with col1:
        render_kpi_card(
            "SENT",
            f"{campaign_metrics['sent']:,}",
            "Campaign messages sent",
            COLOR_PRIMARY,
        )

    with col2:
        render_kpi_card(
            "DELIVERED",
            f"{campaign_metrics['delivered']:,}",
            f"Delivery rate {campaign_metrics['delivery_rate']:.2%}",
            COLOR_SECONDARY,
        )

    with col3:
        render_kpi_card(
            "CLICKED",
            f"{campaign_metrics['clicked']:,}",
            f"Click rate {campaign_metrics['click_rate']:.2%}",
            COLOR_TERTIARY,
        )

    with col4:
        render_kpi_card(
            "REDEEMED",
            f"{campaign_metrics['redeemed']:,}",
            f"Redemption rate {campaign_metrics['redemption_rate']:.2%}",
            COLOR_ALERT,
        )

    # =====================================================
    # CAMPAIGN FUNNEL
    # =====================================================

    st.divider()
    st.subheader("Campaign Funnel")
    st.caption(
        "Sequential movement from campaign delivery through interaction and redemption."
    )

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
        textfont_color=COLOR_TEXT,
    )

    fig = apply_chart_style(fig)
    fig.update_layout(
        height=390,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # =========================================================
    # CAMPAIGN TYPE PERFORMANCE
    # =========================================================

    st.divider()
    st.subheader("Campaign Performance by Type")
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

    display_campaign_data["delivery_rate"] = (
        display_campaign_data["delivery_rate"]
        .map(lambda x: f"{x:.2%}")
    )

    display_campaign_data["click_rate"] = (
        display_campaign_data["click_rate"]
        .map(lambda x: f"{x:.2%}")
    )

    display_campaign_data["redemption_rate"] = (
        display_campaign_data["redemption_rate"]
        .map(lambda x: f"{x:.2%}")
    )

    display_campaign_data["total_reward_value"] = (
        display_campaign_data["total_reward_value"]
        .map(lambda x: f"₹{x:,.0f}")
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
                        ("background-color", COLOR_SURFACE),
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
# CAMPAIGN AFFINITY
# =========================================================

def render_campaign_affinity(campaigns):
    """
    Render customer-level campaign affinity analysis.

    Campaign affinity is based on campaign delivery and click
    behaviour by customer and campaign type. Click rate is
    calculated as clicked / delivered, so undelivered campaigns
    do not create a click opportunity.
    """

    render_page_header(
        "Campaign Affinity",
        "Understand which campaign types each customer engages with based on delivered campaigns and clicks.",
    )

    affinity_data = calculate_campaign_affinity(campaigns)

    if affinity_data.empty:
        st.info("No campaign affinity data is available.")
        return

    # -----------------------------------------------------
    # Overall affinity summary
    # -----------------------------------------------------

    total_customers = affinity_data["customer_id"].nunique()
    total_campaign_types = affinity_data["campaign_type"].nunique()
    total_delivered = int(affinity_data["campaigns_delivered"].sum())
    total_clicked = int(affinity_data["campaigns_clicked"].sum())

    col1, col2, col3, col4 = st.columns(4, gap="medium")

    with col1:
        render_kpi_card(
            "CUSTOMERS ANALYZED",
            f"{total_customers:,}",
            "Customers with campaign history",
            COLOR_PRIMARY,
        )

    with col2:
        render_kpi_card(
            "CAMPAIGN TYPES",
            f"{total_campaign_types:,}",
            "Campaign types with observed activity",
            COLOR_PRIMARY,
        )

    with col3:
        render_kpi_card(
            "DELIVERED",
            f"{total_delivered:,}",
            "Delivered campaign opportunities",
            COLOR_SECONDARY,
        )

    with col4:
        render_kpi_card(
            "CLICKS",
            f"{total_clicked:,}",
            "Observed customer interactions",
            COLOR_TERTIARY,
        )

    # -----------------------------------------------------
    # Customer lookup
    # -----------------------------------------------------

    st.divider()
    st.subheader("Customer Campaign Affinity")
    st.caption(
        "Search a customer to compare campaign exposure and click behaviour across campaign types."
    )

    affinity_data = affinity_data.copy()
    affinity_data["customer_id"] = (
        affinity_data["customer_id"].astype(str).str.strip()
    )

    customer_ids = sorted(
        affinity_data["customer_id"].dropna().unique().tolist()
    )

    search_query = st.text_input(
        "Customer ID",
        placeholder="Type a customer ID, e.g. C00 or C0036",
        key="campaign_affinity_customer_search",
    ).strip()

    if not search_query:
        st.info(
            "Enter a customer ID or partial ID to inspect campaign affinity."
        )
    else:
        query_lower = search_query.lower()

        matching_ids = [
            customer_id
            for customer_id in customer_ids
            if query_lower in customer_id.lower()
        ]

        if not matching_ids:
            st.warning(
                f"No customer IDs found matching '{search_query}'."
            )
        else:
            exact_matches = [
                customer_id
                for customer_id in matching_ids
                if customer_id.lower() == query_lower
            ]

            if exact_matches:
                selected_customer_id = exact_matches[0]
            elif len(matching_ids) == 1:
                selected_customer_id = matching_ids[0]
            else:
                st.caption(
                    f"{len(matching_ids):,} customer IDs match "
                    f"'{search_query}'. Select a customer:"
                )
                selected_customer_id = st.selectbox(
                    "Matching customers",
                    matching_ids,
                    label_visibility="collapsed",
                    key="campaign_affinity_customer_result",
                )

            customer_affinity = affinity_data[
                affinity_data["customer_id"] == selected_customer_id
            ].copy()

            if customer_affinity.empty:
                st.info("No campaign affinity records are available for this customer.")
            else:
                st.markdown(
                    f"**Campaign response profile — {selected_customer_id}**"
                )

                customer_affinity = customer_affinity.sort_values(
                    ["click_rate", "campaigns_delivered", "campaigns_clicked"],
                    ascending=[False, False, False],
                )

                chart_data = customer_affinity.sort_values(
                    "click_rate",
                    ascending=True,
                ).copy()

                fig = px.bar(
                    chart_data,
                    x="click_rate",
                    y="campaign_type",
                    orientation="h",
                    text="click_rate",
                    color_discrete_sequence=[COLOR_PRIMARY],
                )

                fig = apply_chart_style(fig)
                fig.update_layout(
                    xaxis_title="Click Rate",
                    yaxis_title=None,
                    xaxis_tickformat=".0%",
                    showlegend=False,
                    height=360,
                )
                fig.update_traces(
                    texttemplate="%{text:.1%}",
                    textposition="outside",
                    marker_line_width=0,
                    textfont_color=COLOR_TEXT,
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                )

                display_affinity = customer_affinity[
                    [
                        "campaign_type",
                        "campaigns_sent",
                        "campaigns_delivered",
                        "campaigns_clicked",
                        "campaigns_not_clicked",
                        "click_rate",
                    ]
                ].copy()

                display_affinity["click_rate"] = (
                    display_affinity["click_rate"]
                    .map(lambda x: f"{x:.2%}")
                )

                display_affinity.columns = [
                    "Campaign Type",
                    "Sent",
                    "Delivered",
                    "Clicked",
                    "Not Clicked",
                    "Click Rate",
                ]

                with st.container(border=True):
                    st.dataframe(
                        display_affinity,
                        use_container_width=True,
                        hide_index=True,
                    )

    # -----------------------------------------------------
    # Campaign-type affinity benchmark
    # -----------------------------------------------------

    st.divider()
    st.subheader("Campaign Type Affinity")
    st.caption(
        "Overall customer engagement by campaign type. "
        "Click rate is based on delivered campaigns."
    )

    type_affinity = calculate_campaign_type_affinity(campaigns)

    if type_affinity.empty:
        st.info("No campaign-type affinity data is available.")
        return

    type_chart = type_affinity.sort_values(
        "click_rate",
        ascending=True,
    ).copy()

    fig = px.bar(
        type_chart,
        x="click_rate",
        y="campaign_type",
        orientation="h",
        text="click_rate",
        color_discrete_sequence=[COLOR_SECONDARY],
    )

    fig = apply_chart_style(fig)
    fig.update_layout(
        xaxis_title="Click Rate",
        yaxis_title=None,
        xaxis_tickformat=".0%",
        showlegend=False,
        height=350,
    )
    fig.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
        marker_line_width=0,
        textfont_color=COLOR_TEXT,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    display_type_affinity = type_affinity.copy()

    if "click_rate" in display_type_affinity.columns:
        display_type_affinity["click_rate"] = (
            display_type_affinity["click_rate"]
            .map(lambda x: f"{x:.2%}")
        )

    display_type_affinity.columns = [
        column.replace("_", " ").title()
        for column in display_type_affinity.columns
    ]

    with st.container(border=True):
        st.dataframe(
            display_type_affinity,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# MAIN APPLICATION
# =========================================================

def _set_page(page_name):
    """Set the active application page and rerun the Streamlit app."""
    st.session_state["current_page"] = page_name
    st.rerun()


def render_sidebar():
    """Render the finalized sidebar matching the approved UI render."""

    active_dataset = st.session_state.get("active_dataset", "Demo Dataset")
    current_page = st.session_state.get("current_page", "Home Page")

    # ---------------------------------------------------------
    # BRAND
    # ---------------------------------------------------------

    st.sidebar.markdown(
        f"""
        <div style="
            font-size:26px;
            font-weight:800;
            line-height:0.98;
            letter-spacing:-0.045em;
            color:{COLOR_WHITE};
        ">
            Customer<br>Intelligence
        </div>
        <div style="
            font-size:10px;
            font-weight:600;
            color:{COLOR_TEXT_MUTED};
            margin-top:10px;
            letter-spacing:.105em;
        ">
            CHURN &amp; CAMPAIGN ANALYTICS
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # CURRENT DATASET — FIRST CONTROL UNDER THE BRAND
    # ---------------------------------------------------------

    dataset_tone = COLOR_PRIMARY if active_dataset == "Custom Data" else COLOR_TERTIARY

    st.sidebar.markdown(
        f"""
        <div class="sidebar-dataset" style="margin-top:22px;">
            <span class="sidebar-dataset-dot"
                  style="background:{dataset_tone};"></span>
            {active_dataset.upper()}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Dataset → Home divider
    st.sidebar.markdown(
        '<div class="sidebar-divider" style="margin-top:10px;margin-bottom:12px;"></div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # HOME PAGE
    # ---------------------------------------------------------

    st.sidebar.markdown('<div class="sidebar-home">', unsafe_allow_html=True)

    if st.sidebar.button(
        "⌂ Home Page",
        use_container_width=True,
        type="primary" if current_page == "Home Page" else "secondary",
        key="sidebar_home",
    ):
        _set_page("Home Page")

    st.sidebar.markdown("</div>", unsafe_allow_html=True)

    # Home → Data divider
    st.sidebar.markdown(
        '<div class="sidebar-divider" style="margin-top:10px;margin-bottom:10px;"></div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # DATA
    # ---------------------------------------------------------

    st.sidebar.markdown(
        '<div class="sidebar-section-label">DATA</div>',
        unsafe_allow_html=True,
    )

    if st.sidebar.button(
        "▤ Custom Data",
        use_container_width=True,
        type="primary" if current_page == "Data & Upload" else "secondary",
        key="sidebar_custom_data",
    ):
        _set_page("Data & Upload")

    if st.sidebar.button(
        "▧ Data Quality",
        use_container_width=True,
        type="primary" if current_page == "Data Quality" else "secondary",
        key="sidebar_data_quality",
    ):
        _set_page("Data Quality")

    # Data → Customer divider
    st.sidebar.markdown(
        '<div class="sidebar-divider" style="margin-top:8px;margin-bottom:9px;"></div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # CUSTOMER
    # ---------------------------------------------------------

    st.sidebar.markdown(
        '<div class="sidebar-section-label">CUSTOMER</div>',
        unsafe_allow_html=True,
    )

    if st.sidebar.button(
        "⌕ Customer Search",
        use_container_width=True,
        type="primary" if current_page == "Customer Search" else "secondary",
        key="sidebar_customer_search",
    ):
        _set_page("Customer Search")

    # Customer → Analytics divider
    st.sidebar.markdown(
        '<div class="sidebar-divider" style="margin-top:8px;margin-bottom:9px;"></div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # ANALYTICS
    # ---------------------------------------------------------

    st.sidebar.markdown(
        '<div class="sidebar-section-label">ANALYTICS</div>',
        unsafe_allow_html=True,
    )

    analytics_pages = [
        ("▥ Customer Intelligence", "Customer Intelligence", "sidebar_customer_intelligence"),
        ("⌁ Retention & Model Insights", "Retention & Model Insights", "sidebar_retention"),
        ("◈ Campaign Performance", "Campaign Performance", "sidebar_campaign_performance"),
        ("⌘ Campaign Affinity", "Campaign Affinity", "sidebar_campaign_affinity"),
    ]

    for label, page_name, key in analytics_pages:
        if st.sidebar.button(
            label,
            use_container_width=True,
            type="primary" if current_page == page_name else "secondary",
            key=key,
        ):
            _set_page(page_name)


def main():

    initialize_dataset_state()

    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Home Page"

    render_sidebar()

    page = st.session_state["current_page"]

    # =====================================================
    # DATA & UPLOAD
    # =====================================================

    if page == "Data & Upload":
        render_data_upload_page()
        st.stop()

    # =====================================================
    # LOAD ACTIVE SOURCE DATA
    # =====================================================

    customers, transactions, campaigns, data_mode = get_active_dataset()

    # =====================================================
    # HOME PAGE
    # =====================================================

    if page == "Home Page":

        with st.spinner("Preparing customer intelligence..."):
            customer_analytics = load_active_customer_analytics(
                customers,
                transactions,
                campaigns,
                data_mode,
            )

        render_executive_overview(
            customers,
            transactions,
            campaigns,
            customer_analytics,
            data_mode,
        )

    # =====================================================
    # CUSTOMER INTELLIGENCE
    # =====================================================

    elif page == "Customer Intelligence":

        with st.spinner("Preparing customer intelligence..."):
            customer_analytics = load_active_customer_analytics(
                customers,
                transactions,
                campaigns,
                data_mode,
            )

        render_customer_segmentation(customer_analytics, customers, data_mode)

    # =====================================================
    # RETENTION & MODEL INSIGHTS
    # =====================================================

    elif page == "Retention & Model Insights":

        with st.spinner("Preparing retention and model insights..."):
            customer_analytics = load_active_customer_analytics(
                customers,
                transactions,
                campaigns,
                data_mode,
            )

        render_retention_model_insights(customer_analytics, customers, data_mode)

    # =====================================================
    # CAMPAIGN PERFORMANCE
    # =====================================================

    elif page == "Campaign Performance":

        render_campaign_performance(campaigns)

    # =====================================================
    # CAMPAIGN AFFINITY
    # =====================================================

    elif page == "Campaign Affinity":

        render_campaign_affinity(campaigns)

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

        with st.spinner("Preparing customer analytics..."):
            customer_analytics = load_active_customer_analytics(
                customers,
                transactions,
                campaigns,
                data_mode,
            )

        search_data = prepare_customer_search_data(customer_analytics, customers)

        render_customer_search_page(search_data)


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
