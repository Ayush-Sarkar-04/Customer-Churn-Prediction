import html
import textwrap

import pandas as pd
import streamlit as st

from config import (
    COLOR_CARD,
    COLOR_CARD_MUTED,
    COLOR_CARD_TEXT,
    COLOR_BORDER,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
)


def _safe(value, default="—"):
    if value is None or pd.isna(value):
        return default
    return value


def _format_number(value, decimals=0):
    value = _safe(value)
    if value == "—":
        return "—"
    return f"{float(value):,.{decimals}f}"


def _format_currency(value):
    value = _safe(value)
    if value == "—":
        return "—"
    return f"₹{float(value):,.2f}"


def _format_percent(value):
    value = _safe(value)
    if value == "—":
        return "—"
    return f"{float(value):.2%}"


def _format_days(value):
    value = _safe(value)
    if value == "—":
        return "—"
    return f"{float(value):,.0f} days"


def _format_date(value):
    value = _safe(value)
    if value == "—":
        return "—"
    parsed = pd.to_datetime(value, errors="coerce", dayfirst=True)
    if pd.isna(parsed):
        return str(value)
    return parsed.strftime("%d %b %Y")


def _inject_card_styles():
    """Apply the Customer Search card treatment using config.py colors."""
    st.markdown(
        f"""
        <style>
            .customer-search-card {{
                background: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 6px;
                padding: 16px 18px 14px 18px;
                min-height: 96px;
                box-sizing: border-box;
            }}

            .customer-search-card .card-label {{
                color: {COLOR_CARD_MUTED};
                font-size: 0.66rem;
                font-weight: 700;
                letter-spacing: 0.10em;
                margin-bottom: 8px;
            }}

            .customer-search-card .card-value {{
                color: {COLOR_CARD_TEXT};
                font-size: 1.05rem;
                font-weight: 700;
                line-height: 1.25;
                margin-bottom: 6px;
            }}

            .customer-search-card .card-description {{
                color: {COLOR_CARD_MUTED};
                font-size: 0.70rem;
                line-height: 1.25;
            }}

            .customer-search-profile {{
                display: grid;
                grid-template-columns: 1.35fr 1fr 1fr 1fr;
                background: {COLOR_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 6px;
                min-height: 96px;
                overflow: hidden;
            }}

            .customer-search-profile .profile-item {{
                padding: 16px 18px 14px 18px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                min-width: 0;
            }}

            .customer-search-profile .profile-item + .profile-item {{
                border-left: 1px solid {COLOR_BORDER};
            }}

            .customer-search-profile .profile-label {{
                color: {COLOR_CARD_MUTED};
                font-size: 0.66rem;
                font-weight: 700;
                letter-spacing: 0.10em;
                margin-bottom: 8px;
            }}

            .customer-search-profile .profile-value {{
                color: {COLOR_CARD_TEXT};
                font-size: 1.05rem;
                font-weight: 700;
                line-height: 1.25;
                margin-bottom: 5px;
            }}

            .customer-search-profile .profile-description {{
                color: {COLOR_CARD_MUTED};
                font-size: 0.70rem;
                line-height: 1.25;
            }}

            .customer-search-section-gap {{
                height: 2px;
            }}

            .customer-search-row-gap {{
                height: 10px;
            }}

            @media (max-width: 900px) {{
                .customer-search-profile {{
                    grid-template-columns: 1fr 1fr;
                }}

                .customer-search-profile .profile-item:nth-child(3) {{
                    border-left: 0;
                    border-top: 1px solid {COLOR_BORDER};
                }}

                .customer-search-profile .profile-item:nth-child(4) {{
                    border-top: 1px solid {COLOR_BORDER};
                }}
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _card(label, value, description):
    st.markdown(
        f"""
        <div class="customer-search-card">
            <div class="card-label">{html.escape(str(label))}</div>
            <div class="card-value">{html.escape(str(value))}</div>
            <div class="card-description">{html.escape(str(description))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _profile_card(customer):
    items = [
        ("CUSTOMER ID", _safe(customer.get("customer_id")), "Customer profile"),
        (
            "CUSTOMER SINCE",
            _format_date(customer.get("registration_date")),
            "Registration date",
        ),
        (
            "CUSTOMER SEGMENT",
            _safe(customer.get("customer_segment")),
            "RFM-based segment",
        ),
        ("LOCATION", _safe(customer.get("city")), "Customer location"),
    ]

    profile_items_html = []

    for label, value, description in items:
        profile_items_html.append(
            (
                '<div class="profile-item">'
                f'<div class="profile-label">{html.escape(str(label))}</div>'
                f'<div class="profile-value">{html.escape(str(value))}</div>'
                f'<div class="profile-description">{html.escape(str(description))}</div>'
                '</div>'
            )
        )

    # Build the HTML without leading whitespace/newlines.
    # Leading spaces in the generated HTML make Streamlit interpret it
    # as a Markdown code block instead of rendering the profile card.
    profile_html = (
        '<div class="customer-search-profile">'
        + "".join(profile_items_html)
        + "</div>"
    )

    st.markdown(profile_html, unsafe_allow_html=True)


def _details_table(customer):
    details = [
        ("Customer ID", _safe(customer.get("customer_id"))),
        ("Gender", _safe(customer.get("gender"))),
        ("Age", _format_number(customer.get("age"))),
        ("City", _safe(customer.get("city"))),
        ("Customer Since", _format_date(customer.get("registration_date"))),
        ("Observation Date", _format_date(customer.get("observation_date"))),
        ("Churn Prediction", _safe(customer.get("churn_prediction"))),
        ("Campaigns Received", _format_number(customer.get("campaigns_received"))),
        (
            "Campaigns Since Last Purchase",
            _format_number(customer.get("campaigns_since_last_purchase")),
        ),
    ]

    details_df = pd.DataFrame(details, columns=["Metric", "Value"])

    st.markdown(
        '<div class="app-table-wrap">'
        + details_df.to_html(
            index=False,
            classes="app-table",
            border=0,
            escape=True,
        )
        + '</div>',
        unsafe_allow_html=True,
    )


def _section(title, description):
    st.subheader(title)
    st.caption(description)


def render_customer_search_page(search_data):
    _inject_card_styles()

    if "customer_id" not in search_data.columns:
        st.error("Customer search data does not contain customer_id.")
        return

    data = search_data.copy()

    # Preserve the existing customer-search behaviour.
    data["customer_id"] = data["customer_id"].astype(str).str.strip()
    data = data[
        data["customer_id"].notna() & (data["customer_id"] != "")
    ].copy()

    data = (
        data.sort_values("customer_id")
        .drop_duplicates("customer_id", keep="last")
        .reset_index(drop=True)
    )

    customer_ids = data["customer_id"].tolist()

    if not customer_ids:
        st.info("No customer profiles are available.")
        return

    # ---------------------------------------------------------
    # PAGE HEADER
    # ---------------------------------------------------------
    st.title("Customer Search")
    st.caption(
        "Search for a customer to view their profile, RFM, purchase behaviour, "
        "campaign engagement and churn intelligence."
    )

    # ---------------------------------------------------------
    # SEARCH
    # ---------------------------------------------------------
    search_query = st.text_input(
        "Search Customer",
        placeholder="Type a customer ID, e.g. C00 or C0036",
        key="customer_search_query",
    ).strip()

    if not search_query:
        st.caption(
            "Enter a customer ID or partial ID to begin. "
            "For example, C00 will show matching customers."
        )
        return

    query_lower = search_query.lower()

    matching_ids = [
        customer_id
        for customer_id in customer_ids
        if query_lower in customer_id.lower()
    ]

    if not matching_ids:
        st.warning(f"No customer IDs found matching '{search_query}'.")
        return

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
            key="customer_search_result",
        )

    customer_rows = data[data["customer_id"] == selected_customer_id]

    if customer_rows.empty:
        st.error("Customer profile could not be loaded.")
        return

    customer = customer_rows.iloc[0]

    # ---------------------------------------------------------
    # CUSTOMER PROFILE
    # ---------------------------------------------------------
    _section(
        f"Customer Profile — {selected_customer_id}",
        "Key information about the selected customer.",
    )
    _profile_card(customer)

    # ---------------------------------------------------------
    # CUSTOMER RISK
    # ---------------------------------------------------------
    _section(
        "Customer Risk",
        "Current churn and customer segmentation indicators.",
    )

    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        _card(
            "CUSTOMER SEGMENT",
            _safe(customer.get("customer_segment")),
            "RFM-based customer segment",
        )

    with c2:
        _card(
            "RISK LEVEL",
            _safe(customer.get("risk_level")),
            "Predicted churn-risk category",
        )

    with c3:
        _card(
            "CHURN PROBABILITY",
            _format_percent(customer.get("churn_probability")),
            "Predicted probability of future churn",
        )

    # ---------------------------------------------------------
    # RFM PROFILE
    # ---------------------------------------------------------
    _section(
        "RFM Profile",
        "Recency, Frequency and Monetary value for the selected customer.",
    )

    c1, c2, c3, c4 = st.columns(4, gap="small")

    with c1:
        _card(
            "RECENCY",
            _format_days(customer.get("recency")),
            "Days since last purchase",
        )

    with c2:
        _card(
            "FREQUENCY",
            _format_number(customer.get("frequency")),
            "Total number of purchases",
        )

    with c3:
        _card(
            "MONETARY VALUE",
            _format_currency(customer.get("monetary")),
            "Total spend (INR)",
        )

    with c4:
        _card(
            "RFM SCORE",
            _format_number(customer.get("rfm_score")),
            "Combined RFM score",
        )

    # ---------------------------------------------------------
    # PURCHASE BEHAVIOUR
    # ---------------------------------------------------------
    _section(
        "Purchase Behaviour",
        "Key purchase indicators for the selected customer.",
    )

    c1, c2 = st.columns(2, gap="small")

    with c1:
        _card(
            "AVERAGE BILL",
            _format_currency(customer.get("average_bill")),
            "Average value per transaction",
        )

    with c2:
        _card(
            "CUSTOMER TENURE",
            _format_days(customer.get("customer_tenure")),
            "Days since customer registration",
        )

    # ---------------------------------------------------------
    # CAMPAIGN ENGAGEMENT
    # ---------------------------------------------------------
    _section(
        "Campaign Engagement",
        "Historical campaign reach, interaction and redemption behaviour.",
    )

    c1, c2, c3, c4 = st.columns(4, gap="small")

    with c1:
        _card(
            "CAMPAIGNS SENT",
            _format_number(customer.get("campaigns_sent")),
            "Campaign messages sent",
        )

    with c2:
        _card(
            "DELIVERED",
            _format_number(customer.get("campaigns_delivered")),
            "Campaign messages delivered",
        )

    with c3:
        _card(
            "CLICKED",
            _format_number(customer.get("campaign_clicks")),
            "Campaign interactions",
        )

    with c4:
        _card(
            "DELIVERY RATE",
            _format_percent(customer.get("delivery_rate")),
            "Delivered messages as a share of sent",
        )

    st.markdown(
        '<div class="customer-search-row-gap"></div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3, gap="small")

    with c1:
        _card(
            "CLICK RATE",
            _format_percent(customer.get("click_rate")),
            "Clicks as a share of delivered",
        )

    with c2:
        _card(
            "REDEMPTION RATE",
            _format_percent(customer.get("redemption_rate")),
            "Redemptions as a share of delivered",
        )

    with c3:
        _card(
            "PREVIOUS REDEMPTIONS",
            _format_number(customer.get("previous_redemptions")),
            "Historical campaign redemptions",
        )

    # Keep a small, consistent gap between the second campaign row
    # and the Profile Details expander.
    st.markdown(
        '<div class="customer-search-row-gap"></div>',
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------
    # PROFILE DETAILS
    # ---------------------------------------------------------
    with st.expander("Profile Details"):
        _details_table(customer)