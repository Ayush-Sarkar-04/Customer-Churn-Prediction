import pandas as pd
import plotly.express as px
import streamlit as st

from config import (
    COLOR_BACKGROUND,
    COLOR_SURFACE,
    COLOR_SURFACE_LIGHT,
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
    COLOR_CARD,
    COLOR_CARD_TEXT,
    COLOR_CARD_MUTED,
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
from src.analytics.retention_cost import (
    calculate_retention_cost,
    calculate_customer_retention_cost,
)
from src.analytics.customer_value import calculate_customer_value
from src.analytics.economic_value import calculate_economic_value_at_risk


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

# Apply the global UI theme from config.py.
# Theme changes should be made in config.py only.
THEME_CSS = """
<style>
    :root {
        --bg: __COLOR_BACKGROUND__;
        --surface: __COLOR_SURFACE__;
        --surface-light: __COLOR_SURFACE_LIGHT__;
        --surface-alt: __COLOR_SURFACE_ALT__;
        --border: __COLOR_BORDER__;
        --grid: __COLOR_GRID__;
        --text: __COLOR_TEXT__;
        --text-muted: __COLOR_TEXT_MUTED__;
        --text-dark: __COLOR_TEXT_DARK__;
        --primary: __COLOR_PRIMARY__;
        --secondary: __COLOR_SECONDARY__;
        --alert: __COLOR_ALERT__;
        --card: __COLOR_CARD__;
        --card-text: __COLOR_CARD_TEXT__;
        --card-muted: __COLOR_CARD_MUTED__;
    }

    /* =====================================================
       GLOBAL
       ===================================================== */

    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: var(--bg);
    }

    [data-testid="stHeader"] {
        background: var(--bg);
    }

    h1, h2, h3, h4, h5, h6,
    [data-testid="stMarkdownContainer"] p,
    label {
        color: var(--text);
    }

    /* =====================================================
       SIDEBAR — COMPACT, NO UNNECESSARY SCROLLING
       ===================================================== */

    [data-testid="stSidebar"] {
        background: var(--bg);
        border-right: 1px solid var(--border);
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.75rem;
        padding-bottom: 0.55rem;
    }

    [data-testid="stSidebar"] .stButton {
        margin: 0 0 5px 0;
    }

    [data-testid="stSidebar"] .stButton > button {
        min-height: 39px !important;
        height: 39px !important;
        padding: 0.35rem 0.65rem !important;
        justify-content: center !important;
        text-align: center !important;
        background: var(--surface-alt) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        box-shadow: none !important;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background: var(--surface) !important;
        border-color: var(--primary) !important;
        color: var(--text-dark) !important;
    }

    [data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
        background: var(--primary) !important;
        border-color: var(--primary) !important;
        color: var(--text) !important;
    }

    .sidebar-brand {
        color: var(--text) !important;
        font-size: 24px;
        font-weight: 800;
        line-height: 0.96;
        letter-spacing: -0.045em;
    }

    .sidebar-tagline {
        color: var(--text-muted) !important;
        font-size: 9px;
        font-weight: 700;
        margin-top: 8px;
        letter-spacing: .10em;
    }

    .sidebar-dataset-button .stButton > button {
        min-height: 34px !important;
        height: 34px !important;
        font-size: 9px !important;
        letter-spacing: .07em !important;
        text-transform: uppercase !important;
        justify-content: flex-start !important;
        padding-left: 12px !important;
        background: var(--surface-alt) !important;
        border-color: var(--border) !important;
    }

    .sidebar-section-label {
        color: var(--text-muted);
        font-size: 8px;
        font-weight: 800;
        letter-spacing: .15em;
        text-transform: uppercase;
        margin: 7px 0 5px 1px;
    }

    .sidebar-divider {
        height: 1px;
        background: var(--border);
        margin: 7px 0;
    }

    /* =====================================================
       CONTENT
       ===================================================== */

    [data-testid="stVerticalBlockBorderWrapper"] {
        background: var(--card) !important;
        border-color: var(--border) !important;
        border-radius: 8px;
    }

    /* Bordered cards use the cream theme surface for visual variation. */
    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stMetricValue"],
    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stMetricLabel"],
    [data-testid="stVerticalBlockBorderWrapper"] label {
        color: var(--card-text) !important;
    }

    [data-testid="stVerticalBlockBorderWrapper"]
    .stCaption,
    [data-testid="stVerticalBlockBorderWrapper"]
    [data-testid="stCaptionContainer"] {
        color: var(--card-muted) !important;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    textarea {
        background: var(--surface-alt) !important;
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
        background: var(--surface) !important;
        border-color: var(--primary) !important;
        color: var(--text-dark) !important;
    }

    .stButton > button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        background: var(--primary) !important;
        color: var(--text) !important;
        border-color: var(--primary) !important;
    }

    [data-testid="stAlert"] {
        background: var(--surface-alt) !important;
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
        color: var(--text-muted) !important;
    }

    /* Keep native dataframe containers visually connected to the page. */
    [data-testid="stDataFrame"] {
        background: var(--surface-alt) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        padding: 0 !important;
    }

    /* -----------------------------------------------------
       HTML TABLES
       Used by Data Quality and Campaign Affinity so the
       complete table surface follows the application theme.
       ----------------------------------------------------- */
    .app-table-wrap {
        width: 100%;
        margin: 0;
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
        background: var(--surface-alt);
    }

    .app-table {
        width: 100%;
        border-collapse: collapse;
        border-spacing: 0;
        background: var(--surface-alt);
        color: var(--text);
        font-size: 12px;
        line-height: 1.35;
    }

    .app-table thead th {
        background: var(--card);
        color: var(--card-text);
        font-weight: 700;
        text-align: left;
        padding: 10px 12px;
        border-bottom: 1px solid var(--border);
        white-space: nowrap;
    }

    .app-table tbody td {
        background: var(--surface-alt);
        color: var(--text);
        padding: 9px 12px;
        border-bottom: 1px solid var(--grid);
        vertical-align: middle;
    }

    .app-table tbody tr:last-child td {
        border-bottom: 0;
        background: var(--surface-alt);
    }

    .app-table tbody tr:nth-child(even) td {
        background: var(--surface);
    }

    .app-table tbody tr:hover td {
        background: var(--surface-light);
    }

    .app-table tbody td:first-child {
        font-weight: 600;
    }

    /* Sidebar dataset control is an indicator/shortcut, not a
       primary action. Keep it on the normal surface even when
       the Data & Upload page is active. */
    .sidebar-dataset-button button[data-testid="baseButton-primary"],
    .sidebar-dataset-button button[kind="primary"] {
        background: var(--surface-alt) !important;
        color: var(--text) !important;
        border-color: var(--border) !important;
    }

    .sidebar-dataset-button button[data-testid="baseButton-primary"]:hover,
    .sidebar-dataset-button button[kind="primary"]:hover {
        background: var(--surface) !important;
        border-color: var(--primary) !important;
        color: var(--text) !important;
    }

    button[data-baseweb="tab"] {
        color: var(--text-muted) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--text) !important;
        border-bottom-color: var(--primary) !important;
    }

    [data-testid="stFileUploader"] section {
        background: var(--card) !important;
        border: 1px dashed var(--border) !important;
        border-radius: 8px;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section * {
        color: var(--card-text) !important;
    }

    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span {
        color: var(--card-muted) !important;
    }

    /* File uploader action button — keep it inside the application theme. */
    [data-testid="stFileUploader"] button {
        background: var(--surface) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        box-shadow: none !important;
        font-weight: 600 !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background: var(--surface-light) !important;
        color: var(--text) !important;
        border-color: var(--primary) !important;
    }

    /* Sidebar dataset indicator — compact, centered and not a full-size nav button. */
    [data-testid="stSidebar"] div[class*="st-key-sidebar_dataset_control"] .stButton > button {
        width: 145px !important;
        min-width: 145px !important;
        max-width: 145px !important;
        min-height: 29px !important;
        height: 29px !important;
        padding: 0.15rem 0.55rem !important;
        margin: 0 auto !important;
        justify-content: center !important;
        text-align: center !important;
        font-size: 9px !important;
        letter-spacing: .06em !important;
    }

    hr {
        border-color: var(--border) !important;
    }

    /* Radio controls */
    [data-baseweb="radio"] [role="radio"][aria-checked="true"] {
        border-color: var(--primary) !important;
        background: var(--primary) !important;
    }
</style>
"""

THEME_CSS = (
    THEME_CSS
    .replace("__COLOR_BACKGROUND__", COLOR_BACKGROUND)
    .replace("__COLOR_SURFACE__", COLOR_SURFACE)
    .replace("__COLOR_SURFACE_LIGHT__", COLOR_SURFACE_LIGHT)
    .replace("__COLOR_SURFACE_ALT__", COLOR_SURFACE_ALT)
    .replace("__COLOR_BORDER__", COLOR_BORDER)
    .replace("__COLOR_GRID__", COLOR_GRID)
    .replace("__COLOR_TEXT__", COLOR_TEXT)
    .replace("__COLOR_TEXT_MUTED__", COLOR_TEXT_MUTED)
    .replace("__COLOR_TEXT_DARK__", COLOR_TEXT_DARK)
    .replace("__COLOR_PRIMARY__", COLOR_PRIMARY)
    .replace("__COLOR_SECONDARY__", COLOR_SECONDARY)
    .replace("__COLOR_ALERT__", COLOR_ALERT)
    .replace("__COLOR_CARD__", COLOR_CARD)
    .replace("__COLOR_CARD_TEXT__", COLOR_CARD_TEXT)
    .replace("__COLOR_CARD_MUTED__", COLOR_CARD_MUTED)
)

st.markdown(THEME_CSS, unsafe_allow_html=True)


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
# THEMED TABLE HELPER
# =========================================================

def render_themed_table(dataframe):
    """Render a dataframe as a theme-controlled HTML table."""
    if dataframe is None or dataframe.empty:
        st.info("No data available.")
        return

    st.markdown(
        '<div class="app-table-wrap">'
        + dataframe.to_html(
            index=False,
            classes="app-table",
            border=0,
            escape=True,
        )
        + '</div>',
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
            width="stretch",
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
            width="stretch",
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
        width="stretch",
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

    st.plotly_chart(fig, width="stretch")

    with st.expander("View complete feature ranking"):
        feature_table = feature_importance.copy()
        feature_table["importance"] = feature_table["importance"].map(
            lambda x: f"{x:.2%}"
        )
        feature_table.columns = ["Rank", "Feature", "Importance"]
        render_themed_table(feature_table)

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

    st.plotly_chart(fig, width="stretch")


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
        width="stretch",
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

    st.markdown(
        '<div class="app-table-wrap">'
        + display_campaign_data.to_html(
            index=False,
            classes="app-table",
            border=0,
            escape=True,
        )
        + '</div>',
        unsafe_allow_html=True,
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
                    width="stretch",
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
                    .map(
                        lambda x: "—"
                        if pd.isna(x)
                        else f"{float(x):.2%}"
                     )
             )

                display_affinity.columns = [
                    "Campaign Type",
                    "Sent",
                    "Delivered",
                    "Clicked",
                    "Not Clicked",
                    "Click Rate",
                ]

                render_themed_table(display_affinity)

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
        color_discrete_sequence=[COLOR_PRIMARY],
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
        width="stretch",
    )

    display_type_affinity = type_affinity.copy()

    if "click_rate" in display_type_affinity.columns:
        display_type_affinity["click_rate"] = (
            display_type_affinity["click_rate"]
            .map(
                lambda x: "—"
                if pd.isna(x)
                else f"{float(x):.2%}"
            )
        )

    display_type_affinity.columns = [
        column.replace("_", " ").title()
        for column in display_type_affinity.columns
    ]

    render_themed_table(display_type_affinity)



# =========================================================
# ECONOMIC INTELLIGENCE
# =========================================================

@st.cache_data(show_spinner=False)
def build_economic_intelligence_data(customer_analytics, transactions, campaigns):
    """
    Build the customer-level economic intelligence dataset.

    Customer value is historical customer value calculated from transactions.
    Retention cost is based on observed historical campaign cost.
    Economic Value at Risk combines churn probability with customer value.
    """
    # The transaction source stores dates as DD-MM-YYYY (for example,
    # 16-07-2024), while the Customer Value backend expects ISO-style
    # YYYY-MM-DD dates. Normalize the input at the integration boundary
    # without changing the backend implementation.
    transactions_for_value = transactions.copy()
    if "transaction_date" not in transactions_for_value.columns:
        raise ValueError("Transactions must contain transaction_date.")

    transaction_dates = pd.to_datetime(
        transactions_for_value["transaction_date"],
        dayfirst=True,
        errors="raise",
    )
    transactions_for_value["transaction_date"] = transaction_dates.dt.strftime(
        "%Y-%m-%d"
    )

    customer_value = calculate_customer_value(transactions_for_value).copy()

    retention_cost = calculate_retention_cost(campaigns).copy()
    customer_retention_cost = calculate_customer_retention_cost(
        retention_cost,
        customer_id_column="customer_id",
        cost_column="retention_cost",
    ).copy()
    # The retention-cost backend returns the aggregated customer cost as
    # `total_retention_cost`. Rename it at the UI integration boundary so
    # the economic dataset uses the canonical `retention_cost` field.
    customer_retention_cost = customer_retention_cost.rename(
        columns={"total_retention_cost": "retention_cost"}
    )

    customer_value["customer_id"] = customer_value["customer_id"].astype(str).str.strip()
    customer_retention_cost["customer_id"] = (
        customer_retention_cost["customer_id"].astype(str).str.strip()
    )

    analytics = customer_analytics.copy()
    analytics["customer_id"] = analytics["customer_id"].astype(str).str.strip()

    # Keep the latest customer observation so the economic view aligns with
    # the same current customer profiles used elsewhere in the application.
    analytics = prepare_customer_search_data(analytics)

    economic = analytics.merge(
        customer_value[
            ["customer_id", "customer_value", "purchase_count",
             "average_order_value", "active_days", "annualized_revenue"]
        ],
        on="customer_id",
        how="left",
    )

    economic = economic.merge(
        customer_retention_cost[["customer_id", "retention_cost"]],
        on="customer_id",
        how="left",
    )

    economic["customer_value"] = pd.to_numeric(
        economic["customer_value"], errors="coerce"
    ).fillna(0)
    economic["retention_cost"] = pd.to_numeric(
        economic["retention_cost"], errors="coerce"
    ).fillna(0)

    economic = calculate_economic_value_at_risk(
        economic,
        churn_probability_column="churn_probability",
        customer_value_column="customer_value",
        retention_cost_column="retention_cost",
    )

    economic["net_value_at_risk"] = (
        economic["expected_value_at_risk"] - economic["retention_cost"]
    )

    return economic


def render_economic_intelligence(
    customer_analytics,
    transactions,
    campaigns,
):
    """Render Page 1: Economic Intelligence."""

    render_page_header(
        "Economic Intelligence",
        "Translate customer value and retention cost into a probability-weighted view of economic churn exposure.",
    )

    with st.spinner("Preparing economic intelligence..."):
        economic = build_economic_intelligence_data(
            customer_analytics,
            transactions,
            campaigns,
        )

    if economic.empty:
        st.info("No economic intelligence data is available.")
        return

    # ---------------------------------------------------------
    # STORY INTRO
    # ---------------------------------------------------------
    st.subheader("Where Is the Economic Risk?")
    st.caption(
        "Customer value shows what each customer has historically contributed. "
        "Retention cost shows observed historical campaign cost. "
        "Economic Value at Risk combines value with predicted churn probability."
    )

    # ---------------------------------------------------------
    # TOP-LEVEL ECONOMIC KPIs
    # ---------------------------------------------------------
    total_customer_value = economic["customer_value"].sum()
    total_retention_cost = economic["retention_cost"].sum()
    total_evar = economic["expected_value_at_risk"].sum()
    total_net_evar = economic["net_value_at_risk"].sum()

    k1, k2, k3, k4 = st.columns(4, gap="medium")

    with k1:
        render_kpi_card(
            "CUSTOMER VALUE",
            f"₹{total_customer_value:,.0f}",
            "Historical value across analyzed customers",
            COLOR_PRIMARY,
        )

    with k2:
        render_kpi_card(
            "RETENTION COST",
            f"₹{total_retention_cost:,.0f}",
            "Observed historical campaign cost",
            COLOR_SECONDARY,
        )

    with k3:
        render_kpi_card(
            "EXPECTED VALUE AT RISK",
            f"₹{total_evar:,.0f}",
            "Churn probability × customer value",
            COLOR_ALERT,
        )

    with k4:
        render_kpi_card(
            "NET VALUE AT RISK",
            f"₹{total_net_evar:,.0f}",
            "Expected value at risk minus retention cost",
            COLOR_TERTIARY,
        )

    # ---------------------------------------------------------
    # CUSTOMER ECONOMICS
    # ---------------------------------------------------------
    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
    st.subheader("Customer Economics")
    st.caption(
        "Compare customer value with churn probability to identify customers "
        "where high value and high churn risk intersect."
    )

    avg_customer_value = economic["customer_value"].mean()
    avg_retention_cost = economic["retention_cost"].mean()
    avg_evar = economic["expected_value_at_risk"].mean()
    customers_with_cost = int((economic["retention_cost"] > 0).sum())

    a1, a2, a3, a4 = st.columns(4, gap="medium")

    with a1:
        render_kpi_card(
            "CUSTOMERS ANALYZED",
            f"{len(economic):,}",
            "Current customer profiles",
            COLOR_PRIMARY,
        )

    with a2:
        render_kpi_card(
            "AVG. CUSTOMER VALUE",
            f"₹{avg_customer_value:,.0f}",
            "Historical value per customer",
            COLOR_PRIMARY,
        )

    with a3:
        render_kpi_card(
            "AVG. RETENTION COST",
            f"₹{avg_retention_cost:,.0f}",
            "Historical campaign cost per customer",
            COLOR_SECONDARY,
        )

    with a4:
        render_kpi_card(
            "AVG. EVAR",
            f"₹{avg_evar:,.0f}",
            f"{customers_with_cost:,} customers have observed campaign cost",
            COLOR_ALERT,
        )

    # ---------------------------------------------------------
    # EXPOSURE RANKING
    # ---------------------------------------------------------
    st.markdown("<div style='height: 22px;'></div>", unsafe_allow_html=True)

    exposure = economic.nlargest(15, "net_value_at_risk").copy()
    exposure["customer_id"] = exposure["customer_id"].astype(str)

    fig = px.bar(
        exposure.sort_values("net_value_at_risk", ascending=True),
        x="net_value_at_risk",
        y="customer_id",
        orientation="h",
        text="net_value_at_risk",
        color_discrete_sequence=[COLOR_PRIMARY],
    )
    fig = apply_chart_style(fig)
    fig.update_layout(
        xaxis_title="Net Value at Risk (₹)",
        yaxis_title=None,
        showlegend=False,
        height=520,
    )
    fig.update_traces(
        texttemplate="₹%{text:,.0f}",
        textposition="outside",
        marker_line_width=0,
        textfont_color=COLOR_TEXT,
    )
    st.plotly_chart(fig, width="stretch")

    # ---------------------------------------------------------
    # VALUE VS CHURN RISK
    # ---------------------------------------------------------
    st.divider()
    st.subheader("Customer Value vs. Churn Probability")
    st.caption(
        "Bubble size represents Expected Value at Risk. This is an exposure view, "
        "not a forecast of guaranteed revenue loss."
    )

    scatter_data = economic.copy()
    scatter_data["economic_priority_band"] = pd.cut(
        scatter_data["expected_value_at_risk"],
        bins=[-float("inf"), scatter_data["expected_value_at_risk"].quantile(.5),
              scatter_data["expected_value_at_risk"].quantile(.75),
              float("inf")],
        labels=["Lower", "Elevated", "Highest"],
        duplicates="drop",
    )

    fig = px.scatter(
        scatter_data,
        x="customer_value",
        y="churn_probability",
        size="expected_value_at_risk",
        color="economic_priority_band",
        hover_data=[
            "customer_id",
            "retention_cost",
            "expected_value_at_risk",
            "net_value_at_risk",
        ],
        color_discrete_sequence=CHART_PALETTE,
    )
    fig = apply_chart_style(fig)
    fig.update_layout(
        xaxis_title="Historical Customer Value (₹)",
        yaxis_title="Churn Probability",
        yaxis_tickformat=".0%",
        height=500,
        legend_title="Economic Exposure",
    )
    st.plotly_chart(fig, width="stretch")

    # ---------------------------------------------------------
    # RETENTION COST BY CAMPAIGN TYPE
    # ---------------------------------------------------------
    st.divider()
    st.subheader("Historical Retention Cost by Campaign Type")
    st.caption(
        "Observed campaign costs by type. These costs are a historical proxy and "
        "should not be interpreted as the cost of a future intervention."
    )

    retention_cost_for_display = calculate_retention_cost(campaigns).copy()
    campaign_cost_by_type = (
        retention_cost_for_display.groupby("campaign_type", as_index=False)["retention_cost"]
        .sum()
        .sort_values("retention_cost", ascending=True)
    )

    if campaign_cost_by_type.empty:
        st.info("No campaign cost data is available.")
    else:
        fig = px.bar(
            campaign_cost_by_type,
            x="retention_cost",
            y="campaign_type",
            orientation="h",
            text="retention_cost",
            color_discrete_sequence=[COLOR_SECONDARY],
        )
        fig = apply_chart_style(fig)
        fig.update_layout(
            xaxis_title="Historical Retention Cost (₹)",
            yaxis_title=None,
            showlegend=False,
            height=380,
        )
        fig.update_traces(
            texttemplate="₹%{text:,.0f}",
            textposition="outside",
            marker_line_width=0,
            textfont_color=COLOR_TEXT,
        )
        st.plotly_chart(fig, width="stretch")

    # ---------------------------------------------------------
    # CUSTOMER ECONOMIC PROFILE
    # ---------------------------------------------------------
    st.divider()
    st.subheader("Top Customer Economic Exposure")
    st.caption(
        "Customers ranked by Net Value at Risk. Higher values indicate greater "
        "relative economic exposure after observed retention cost."
    )

    table = economic.nlargest(20, "net_value_at_risk")[
        [
            "customer_id",
            "customer_value",
            "churn_probability",
            "retention_cost",
            "expected_value_at_risk",
            "net_value_at_risk",
        ]
    ].copy()

    table["customer_value"] = table["customer_value"].map(lambda x: f"₹{x:,.0f}")
    table["churn_probability"] = table["churn_probability"].map(
        lambda x: f"{x:.1%}"
    )
    table["retention_cost"] = table["retention_cost"].map(lambda x: f"₹{x:,.0f}")
    table["expected_value_at_risk"] = table["expected_value_at_risk"].map(
        lambda x: f"₹{x:,.0f}"
    )
    table["net_value_at_risk"] = table["net_value_at_risk"].map(
        lambda x: f"₹{x:,.0f}"
    )

    table.columns = [
        "Customer ID",
        "Customer Value",
        "Churn Probability",
        "Retention Cost",
        "Expected Value at Risk",
        "Net Value at Risk",
    ]

    render_themed_table(table)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.info(
        "Interpretation: Expected Value at Risk is a probability-weighted economic "
        "exposure, not guaranteed revenue loss, revenue saved, ROI, or a causal "
        "estimate of campaign impact. Historical campaign cost is used here as a "
        "cost proxy."
    )

# =========================================================
# MAIN APPLICATION
# =========================================================

def _set_page(page_name):
    """Set the active application page and rerun the Streamlit app."""
    # Choosing Custom Data on the Data & Upload page is only a workflow
    # selection. It must not persist as a visible state when the user leaves
    # the page unless validated custom data is actually active.
    if page_name != "Data & Upload":
        active_dataset = st.session_state.get("active_dataset", "Demo Dataset")
        custom_data = st.session_state.get("custom_data")
        custom_is_active = (
            active_dataset == "Custom Data"
            and isinstance(custom_data, dict)
            and all(
                custom_data.get(key) is not None
                for key in ("customers", "transactions", "campaigns")
            )
        )
        if not custom_is_active:
            st.session_state["data_source_mode"] = "Demo Dataset"

    st.session_state["current_page"] = page_name
    st.rerun()


def render_sidebar():
    """Render the compact application sidebar."""

    active_dataset = st.session_state.get(
        "active_dataset",
        "Demo Dataset",
    )
    current_page = st.session_state.get(
        "current_page",
        "Home Page",
    )

    # ---------------------------------------------------------
    # BRAND
    # ---------------------------------------------------------

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            Customer<br>Intelligence
        </div>
        <div class="sidebar-tagline">
            CHURN &amp; CAMPAIGN ANALYTICS
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # DYNAMIC DATASET CONTROL
    # ---------------------------------------------------------
    # This is both the active-dataset indicator and the shortcut
    # to the Data & Upload page.

    dataset_label = (
        "●  CUSTOM DATA"
        if active_dataset == "Custom Data"
        else "●  DEMO DATASET"
    )

    st.sidebar.markdown(
        '<div class="sidebar-dataset-button">',
        unsafe_allow_html=True,
    )

    if st.sidebar.button(
        dataset_label,
        width="content",
        type="primary" if current_page == "Data & Upload" else "secondary",
        key="sidebar_dataset_control",
    ):
        _set_page("Data & Upload")

    st.sidebar.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # HOME
    # ---------------------------------------------------------

    if st.sidebar.button(
        "⌂  Home Page",
        width="stretch",
        type="primary" if current_page == "Home Page" else "secondary",
        key="sidebar_home",
    ):
        _set_page("Home Page")

    st.sidebar.markdown(
        '<div class="sidebar-divider"></div>',
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
        "▧  Data Quality",
        width="stretch",
        type="primary" if current_page == "Data Quality" else "secondary",
        key="sidebar_data_quality",
    ):
        _set_page("Data Quality")

    st.sidebar.markdown(
        '<div class="sidebar-divider"></div>',
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
        "⌕  Customer Search",
        width="stretch",
        type="primary" if current_page == "Customer Search" else "secondary",
        key="sidebar_customer_search",
    ):
        _set_page("Customer Search")

    st.sidebar.markdown(
        '<div class="sidebar-divider"></div>',
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
        (
            "▥  Customer Intelligence",
            "Customer Intelligence",
            "sidebar_customer_intelligence",
        ),
        (
            "⌁  Retention & Model Insights",
            "Retention & Model Insights",
            "sidebar_retention",
        ),
        (
            "◈  Campaign Performance",
            "Campaign Performance",
            "sidebar_campaign_performance",
        ),
        (
            "⌘  Campaign Affinity",
            "Campaign Affinity",
            "sidebar_campaign_affinity",
        ),
    ]

    for label, page_name, key in analytics_pages:
        if st.sidebar.button(
            label,
            width="stretch",
            type="primary" if current_page == page_name else "secondary",
            key=key,
        ):
            _set_page(page_name)

    st.sidebar.markdown(
        '<div class="sidebar-divider"></div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # RETENTION
    # ---------------------------------------------------------
    st.sidebar.markdown(
        '<div class="sidebar-section-label">RETENTION</div>',
        unsafe_allow_html=True,
    )

    if st.sidebar.button(
        "◉  Economic Intelligence",
        width="stretch",
        type="primary" if current_page == "Economic Intelligence" else "secondary",
        key="sidebar_economic_intelligence",
    ):
        _set_page("Economic Intelligence")


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
    # ECONOMIC INTELLIGENCE
    # =====================================================

    elif page == "Economic Intelligence":

        with st.spinner("Preparing customer intelligence..."):
            customer_analytics = load_active_customer_analytics(
                customers,
                transactions,
                campaigns,
                data_mode,
            )

        render_economic_intelligence(
            customer_analytics,
            transactions,
            campaigns,
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
# APPLICATION ENTRY POINT
if __name__ == "__main__":
    main()
