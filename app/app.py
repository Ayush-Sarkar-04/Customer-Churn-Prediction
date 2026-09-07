import pandas as pd
import streamlit as st

from data_quality_page import render_data_quality_page
from customer_search_page import render_customer_search_page

from src.analytics.customer import build_customer_analytics


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Intelligence System",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------
CUSTOMERS_PATH = "data/training/customers.csv"
TRANSACTIONS_PATH = "data/training/transactions.csv"
CAMPAIGNS_PATH = "data/training/campaigns.csv"
FEATURES_PATH = "data/training/customer_features.csv"
MODEL_PATH = "models/random_forest.joblib"


# ---------------------------------------------------------
# Load demo datasets
# ---------------------------------------------------------
@st.cache_data
def load_demo_data():
    customers = pd.read_csv(CUSTOMERS_PATH)
    transactions = pd.read_csv(TRANSACTIONS_PATH)
    campaigns = pd.read_csv(CAMPAIGNS_PATH)

    return customers, transactions, campaigns


# ---------------------------------------------------------
# Build customer analytics
# ---------------------------------------------------------
@st.cache_data
def load_customer_analytics():
    """
    Build customer-level analytics using the prepared
    feature dataset and the saved Random Forest model.
    """

    features = pd.read_csv(FEATURES_PATH)

    analytics = build_customer_analytics(
        features,
        model_path=MODEL_PATH,
    )

    return analytics


# ---------------------------------------------------------
# Prepare customer search data
# ---------------------------------------------------------
def prepare_customer_search_data(customer_analytics):
    """
    Prepare customer analytics for Customer Search.

    The feature dataset contains multiple historical
    observations for customers. The Customer Explorer
    should display only the latest observation for each
    customer.

    Returns
    -------
    pandas.DataFrame
        Latest customer-level observations.
    """

    result = customer_analytics.copy()

    if "observation_date" not in result.columns:
        raise ValueError(
            "Customer analytics must contain observation_date"
        )

    # Parse the observation dates explicitly.
    result["observation_date"] = pd.to_datetime(
        result["observation_date"],
        errors="coerce",
        dayfirst=True,
    )

    # Make sure the dataset contains valid observation dates.
    if result["observation_date"].isna().all():
        raise ValueError(
            "No valid observation dates found"
        )

    # Find the latest observation date available.
    latest_date = result["observation_date"].max()

    # Keep only observations from the latest date.
    result = result[
        result["observation_date"] == latest_date
    ].copy()

    # Safety check: one customer should appear only once
    # in the Customer Explorer.
    result = (
        result
        .drop_duplicates(
            subset=["customer_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return result


# ---------------------------------------------------------
# Main application
# ---------------------------------------------------------
def main():

    # -----------------------------------------------------
    # Sidebar navigation
    # -----------------------------------------------------
    st.sidebar.title("Customer Intelligence")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Data Quality",
            "Customer Search",
        ],
    )

    # -----------------------------------------------------
    # Load demo datasets
    # -----------------------------------------------------
    customers, transactions, campaigns = load_demo_data()

    # -----------------------------------------------------
    # Overview
    # -----------------------------------------------------
    if page == "Overview":

        st.title(
            "Customer Churn Prediction & Campaign Analytics"
        )

        st.write(
            "Analyze customer behavior, churn risk, "
            "segmentation and campaign performance."
        )

        st.divider()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Customers",
                f"{len(customers):,}",
            )

        with col2:
            st.metric(
                "Transactions",
                f"{len(transactions):,}",
            )

        with col3:
            st.metric(
                "Campaigns",
                f"{len(campaigns):,}",
            )

    # -----------------------------------------------------
    # Data Quality
    # -----------------------------------------------------
    elif page == "Data Quality":

        render_data_quality_page(
            customers,
            transactions,
            campaigns,
        )

    # -----------------------------------------------------
    # Customer Search
    # -----------------------------------------------------
    elif page == "Customer Search":

        with st.spinner(
            "Preparing customer analytics..."
        ):

            customer_analytics = load_customer_analytics()

        search_data = prepare_customer_search_data(
            customer_analytics
        )

        render_customer_search_page(
            search_data
        )


# ---------------------------------------------------------
# Application entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()