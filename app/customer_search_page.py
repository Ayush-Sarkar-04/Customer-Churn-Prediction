import pandas as pd
import streamlit as st

from src.analytics.customer_search import (
    search_customers,
    get_customer_profile,
)


def render_customer_search_page(customer_analytics):
    """
    Render the Customer Search / Customer Explorer page.

    Exact customer ID:
        Shows the complete customer profile immediately.

    Partial customer ID:
        Shows matching customers in a result table.
    """

    st.title("Customer Search")

    st.write(
        "Search for a customer and view their RFM profile, "
        "segment, churn probability, and risk level."
    )

    st.divider()

    # ---------------------------------------------------------
    # Search input
    # ---------------------------------------------------------
    search_term = st.text_input(
        "Search Customer ID",
        placeholder="Example: C0036 or C00",
    )

    search_term = search_term.strip()

    # ---------------------------------------------------------
    # Empty search
    # ---------------------------------------------------------
    if not search_term:
        st.info(
            "Enter a customer ID or partial customer ID to search."
        )
        return

    # ---------------------------------------------------------
    # Search customers
    # ---------------------------------------------------------
    matches = search_customers(
        customer_analytics,
        search_term,
    )

    # ---------------------------------------------------------
    # No results
    # ---------------------------------------------------------
    if matches.empty:
        st.warning(
            f"No customers found matching '{search_term}'."
        )
        return

    # ---------------------------------------------------------
    # Check whether the search is an exact customer ID
    # ---------------------------------------------------------
    customer_ids = (
        matches["customer_id"]
        .astype(str)
        .str.strip()
    )

    exact_match = matches[
        customer_ids.str.casefold()
        == search_term.casefold()
    ]

    # =========================================================
    # EXACT CUSTOMER SEARCH
    # =========================================================
    if len(exact_match) == 1:

        customer = get_customer_profile(
            customer_analytics,
            exact_match.iloc[0]["customer_id"],
        )

        st.subheader(
            f"Customer Profile — {customer['customer_id']}"
        )

        # -----------------------------------------------------
        # Customer status
        # -----------------------------------------------------
        status_col1, status_col2, status_col3 = st.columns(3)

        with status_col1:
            st.metric(
                "Customer Segment",
                customer["customer_segment"],
            )

        with status_col2:
            st.metric(
                "Risk Level",
                customer["risk_level"],
            )

        with status_col3:
            churn_probability = (
                float(customer["churn_probability"])
                * 100
            )

            st.metric(
                "Churn Probability",
                f"{churn_probability:.2f}%",
            )

        st.divider()

        # -----------------------------------------------------
        # RFM profile
        # -----------------------------------------------------
        st.subheader("RFM Profile")

        rfm_col1, rfm_col2, rfm_col3, rfm_col4 = st.columns(4)

        with rfm_col1:
            st.metric(
                "Recency",
                f"{customer['recency']:.0f} days",
            )

        with rfm_col2:
            st.metric(
                "Frequency",
                f"{customer['frequency']:.0f}",
            )

        with rfm_col3:
            st.metric(
                "Monetary Value",
                f"₹{customer['monetary']:,.2f}",
            )

        with rfm_col4:
            st.metric(
                "RFM Score",
                f"{customer['rfm_score']:.0f}",
            )

        # -----------------------------------------------------
        # Purchase behaviour
        # -----------------------------------------------------
        st.subheader("Purchase Behaviour")

        purchase_col1, purchase_col2 = st.columns(2)

        with purchase_col1:
            if "average_bill" in customer.index:
                st.metric(
                    "Average Bill",
                    f"₹{customer['average_bill']:,.2f}",
                )

        with purchase_col2:
            if "average_purchase_gap" in customer.index:

                average_gap = customer["average_purchase_gap"]

                if pd.isna(average_gap):
                    gap_text = "N/A"
                else:
                    gap_text = f"{average_gap:.1f} days"

                st.metric(
                    "Average Purchase Gap",
                    gap_text,
                )

        # -----------------------------------------------------
        # Customer tenure
        # -----------------------------------------------------
        if "customer_tenure" in customer.index:
            st.metric(
                "Customer Tenure",
                f"{customer['customer_tenure']:.0f} days",
            )

        # -----------------------------------------------------
        # Campaign engagement
        # -----------------------------------------------------
        campaign_metrics = [
            "campaigns_sent",
            "campaigns_delivered",
            "campaign_clicks",
            "campaigns_redeemed",
        ]

        available_campaign_metrics = [
            column
            for column in campaign_metrics
            if column in customer.index
        ]

        if available_campaign_metrics:

            st.subheader("Campaign Engagement")

            campaign_col1, campaign_col2, campaign_col3, campaign_col4 = (
                st.columns(4)
            )

            if "campaigns_sent" in customer.index:
                with campaign_col1:
                    st.metric(
                        "Campaigns Sent",
                        f"{customer['campaigns_sent']:.0f}",
                    )

            if "campaigns_delivered" in customer.index:
                with campaign_col2:
                    st.metric(
                        "Delivered",
                        f"{customer['campaigns_delivered']:.0f}",
                    )

            if "campaign_clicks" in customer.index:
                with campaign_col3:
                    st.metric(
                        "Clicked",
                        f"{customer['campaign_clicks']:.0f}",
                    )

            if "campaigns_redeemed" in customer.index:
                with campaign_col4:
                    st.metric(
                        "Redeemed",
                        f"{customer['campaigns_redeemed']:.0f}",
                    )

            # -------------------------------------------------
            # Campaign rates
            # -------------------------------------------------
            rate_columns = [
                "delivery_rate",
                "click_rate",
                "redemption_rate",
            ]

            available_rates = [
                column
                for column in rate_columns
                if column in customer.index
            ]

            if available_rates:

                rate_col1, rate_col2, rate_col3 = st.columns(3)

                if "delivery_rate" in customer.index:
                    with rate_col1:
                        st.metric(
                            "Delivery Rate",
                            f"{customer['delivery_rate'] * 100:.2f}%",
                        )

                if "click_rate" in customer.index:
                    with rate_col2:
                        st.metric(
                            "Click Rate",
                            f"{customer['click_rate'] * 100:.2f}%",
                        )

                if "redemption_rate" in customer.index:
                    with rate_col3:
                        st.metric(
                            "Redemption Rate",
                            f"{customer['redemption_rate'] * 100:.2f}%",
                        )

            # -------------------------------------------------
            # Previous redemptions
            # -------------------------------------------------
            if "previous_redemptions" in customer.index:
                st.metric(
                    "Previous Redemptions",
                    f"{customer['previous_redemptions']:.0f}",
                )

        # -----------------------------------------------------
        # Observation information
        # -----------------------------------------------------
        if "observation_date" in customer.index:

            observation_date = customer["observation_date"]

            if pd.notna(observation_date):

                parsed_date = pd.to_datetime(
                    observation_date,
                    errors="coerce",
                )

                if pd.notna(parsed_date):
                    st.caption(
                        "Latest observation: "
                        f"{parsed_date.strftime('%d-%m-%Y')}"
                    )

        return

    # =========================================================
    # PARTIAL SEARCH
    # =========================================================
    st.subheader(
        f"Matching Customers ({len(matches)})"
    )

    display_columns = [
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "customer_segment",
        "churn_probability",
        "risk_level",
    ]

    available_columns = [
        column
        for column in display_columns
        if column in matches.columns
    ]

    result_table = matches[
        available_columns
    ].copy()

    # Convert churn probability to percentage.
    if "churn_probability" in result_table.columns:

        result_table["churn_probability"] = (
            result_table["churn_probability"]
            * 100
        ).round(2)

        result_table = result_table.rename(
            columns={
                "churn_probability": "churn_probability_%"
            }
        )

    st.dataframe(
        result_table,
        width="stretch",
        hide_index=True,
    )

    st.info(
        "Enter an exact customer ID to view the complete profile."
    )