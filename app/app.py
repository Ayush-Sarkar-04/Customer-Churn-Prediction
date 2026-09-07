import pandas as pd
import streamlit as st

from data_quality_page import render_data_quality_page


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
# Main application
# ---------------------------------------------------------
def main():

    st.sidebar.title("Customer Intelligence")

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "Data Quality",
        ],
    )

    customers, transactions, campaigns = load_demo_data()

    # -----------------------------------------------------
    # Overview page
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
    # Data Quality page
    # -----------------------------------------------------
    elif page == "Data Quality":

        render_data_quality_page(
            customers,
            transactions,
            campaigns,
        )


# ---------------------------------------------------------
# Application entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()