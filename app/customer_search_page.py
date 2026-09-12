import pandas as pd
import streamlit as st


# =========================================================
# CUSTOMER SEARCH — CLEAN DASHBOARD UI
# =========================================================

BG = "#0E1117"
SURFACE = "#161A1F"
SURFACE_ALT = "#1E2329"
BORDER = "#30343A"
LINE = "#3A4046"
TEXT = "#E8E8E8"
MUTED = "#92979D"
WHITE = "#E8E8E8"


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


def _metric_card(label, value, description):
    """One consistent bordered metric card. No nested Streamlit borders."""
    st.html(
        f"""
        <div style="
            width:100%;
            min-height:92px;
            box-sizing:border-box;
            background:{SURFACE};
            border:1px solid {BORDER};
            border-radius:7px;
            padding:13px 15px 12px 15px;
            overflow:hidden;
        ">
            <div style="
                height:2px;
                width:100%;
                background:{LINE};
                margin:0 0 10px 0;
            "></div>
            <div style="
                color:{MUTED};
                font-size:10px;
                font-weight:700;
                letter-spacing:.08em;
                text-transform:uppercase;
                line-height:1.2;
                margin:0 0 6px 0;
            ">{label}</div>
            <div style="
                color:{TEXT};
                font-size:20px;
                font-weight:700;
                line-height:1.2;
                margin:0 0 5px 0;
                white-space:nowrap;
                overflow:hidden;
                text-overflow:ellipsis;
            ">{value}</div>
            <div style="
                color:{MUTED};
                font-size:10px;
                line-height:1.3;
                margin:0;
            ">{description}</div>
        </div>
        """,
    )


def _profile_identity(customer):
    """Render the profile header shown in the approved Customer Search render."""
    customer_id = _safe(customer.get("customer_id"))
    registration = _format_date(customer.get("registration_date"))
    segment = _safe(customer.get("customer_segment"))
    city = _safe(customer.get("city"))

    st.html(
        f"""
        <div style="
            width:100%;
            box-sizing:border-box;
            background:{SURFACE};
            border:1px solid {BORDER};
            border-radius:7px;
            padding:16px 18px;
            margin:0;
        ">
            <div style="
                display:grid;
                grid-template-columns:1.7fr 1fr 1fr 1fr;
                align-items:center;
                column-gap:24px;
            ">
                <div style="
                    min-width:0;
                    padding-right:22px;
                    border-right:1px solid {BORDER};
                ">
                    <div style="
                        color:{MUTED};
                        font-size:9px;
                        font-weight:700;
                        letter-spacing:.09em;
                        text-transform:uppercase;
                        margin-bottom:6px;
                    ">CUSTOMER ID</div>
                    <div style="
                        color:{TEXT};
                        font-size:22px;
                        font-weight:700;
                        line-height:1.15;
                        overflow:hidden;
                        text-overflow:ellipsis;
                        white-space:nowrap;
                    ">{customer_id}</div>
                    <div style="
                        color:{MUTED};
                        font-size:10px;
                        margin-top:4px;
                    ">Customer profile</div>
                </div>

                <div>
                    <div style="
                        color:{MUTED};
                        font-size:9px;
                        font-weight:700;
                        letter-spacing:.07em;
                        text-transform:uppercase;
                        margin-bottom:6px;
                    ">CUSTOMER SINCE</div>
                    <div style="
                        color:{TEXT};
                        font-size:13px;
                        font-weight:600;
                    ">{registration}</div>
                </div>

                <div>
                    <div style="
                        color:{MUTED};
                        font-size:9px;
                        font-weight:700;
                        letter-spacing:.07em;
                        text-transform:uppercase;
                        margin-bottom:6px;
                    ">CUSTOMER SEGMENT</div>
                    <div style="
                        color:{TEXT};
                        font-size:13px;
                        font-weight:600;
                    ">{segment}</div>
                </div>

                <div>
                    <div style="
                        color:{MUTED};
                        font-size:9px;
                        font-weight:700;
                        letter-spacing:.07em;
                        text-transform:uppercase;
                        margin-bottom:6px;
                    ">LOCATION</div>
                    <div style="
                        color:{TEXT};
                        font-size:13px;
                        font-weight:600;
                    ">{city}</div>
                </div>
            </div>
        </div>
        """,
    )


def _section_heading(title, description):
    st.html(
        f"""
        <div style="
            margin:22px 0 10px 0;
            padding:0;
        ">
            <div style="
                color:{TEXT};
                font-size:19px;
                font-weight:700;
                line-height:1.2;
                margin:0 0 4px 0;
            ">{title}</div>
            <div style="
                color:{MUTED};
                font-size:11px;
                line-height:1.35;
                margin:0;
            ">{description}</div>
        </div>
        """,
    )


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
        ("Campaigns Since Last Purchase", _format_number(customer.get("campaigns_since_last_purchase"))),
    ]

    details_df = pd.DataFrame(details, columns=["Metric", "Value"])

    st.dataframe(
        details_df,
        use_container_width=True,
        hide_index=True,
        height=360,
    )


def render_customer_search_page(search_data):
    """
    Render the complete Customer Search page.

    Preserves:
    - customer search and partial-ID matching
    - customer identity/profile
    - segment, risk and churn probability
    - RFM profile
    - purchase behaviour
    - campaign engagement
    - profile details
    """

    if "customer_id" not in search_data.columns:
        st.error("Customer search data does not contain customer_id.")
        return

    data = search_data.copy()

    data["customer_id"] = (
        data["customer_id"].astype(str).str.strip()
    )

    data = data[
        data["customer_id"].notna()
        & (data["customer_id"] != "")
    ].copy()

    data = (
        data.sort_values("customer_id")
        .drop_duplicates(subset=["customer_id"], keep="last")
        .reset_index(drop=True)
    )

    customer_ids = data["customer_id"].tolist()

    if not customer_ids:
        st.info("No customer profiles are available.")
        return

    # ---------------------------------------------------------
    # PAGE HEADER
    # ---------------------------------------------------------

    st.html(
        f"""
        <div style="margin:0 0 18px 0;">
            <div style="
                color:{TEXT};
                font-size:31px;
                font-weight:750;
                letter-spacing:-.03em;
                line-height:1.1;
            ">Customer Search</div>
            <div style="
                color:{MUTED};
                font-size:12px;
                margin-top:5px;
                line-height:1.4;
            ">Search for a customer to view their profile, RFM, purchase behaviour,
            campaign engagement and churn intelligence.</div>
        </div>
        """,
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
        st.warning(
            f"No customer IDs found matching '{search_query}'."
        )
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

    customer_rows = data[
        data["customer_id"] == selected_customer_id
    ]

    if customer_rows.empty:
        st.error("Customer profile could not be loaded.")
        return

    customer = customer_rows.iloc[0]

    # ---------------------------------------------------------
    # CUSTOMER PROFILE
    # ---------------------------------------------------------

    _section_heading(
        f"Customer Profile — {selected_customer_id}",
        "Key information about the selected customer.",
    )

    _profile_identity(customer)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

    # ---------------------------------------------------------
    # EXECUTIVE SUMMARY
    # ---------------------------------------------------------

    summary_1, summary_2, summary_3 = st.columns(
        3,
        gap="small",
    )

    with summary_1:
        _metric_card(
            "CUSTOMER SEGMENT",
            str(_safe(customer.get("customer_segment"))),
            "RFM-based customer segment",
        )

    with summary_2:
        _metric_card(
            "RISK LEVEL",
            str(_safe(customer.get("risk_level"))),
            "Predicted churn-risk category",
        )

    with summary_3:
        _metric_card(
            "CHURN PROBABILITY",
            _format_percent(customer.get("churn_probability")),
            "Predicted probability of future churn",
        )

    # ---------------------------------------------------------
    # RFM PROFILE
    # ---------------------------------------------------------

    _section_heading(
        "RFM Profile",
        "Recency, Frequency and Monetary value for the selected customer.",
    )

    rfm_1, rfm_2, rfm_3, rfm_4 = st.columns(
        4,
        gap="small",
    )

    with rfm_1:
        _metric_card(
            "RECENCY",
            _format_days(customer.get("recency")),
            "Days since last purchase",
        )

    with rfm_2:
        _metric_card(
            "FREQUENCY",
            _format_number(customer.get("frequency")),
            "Total number of purchases",
        )

    with rfm_3:
        _metric_card(
            "MONETARY VALUE",
            _format_currency(customer.get("monetary")),
            "Total spend (INR)",
        )

    with rfm_4:
        _metric_card(
            "RFM SCORE",
            _format_number(customer.get("rfm_score")),
            "Combined RFM score",
        )

    # ---------------------------------------------------------
    # PURCHASE BEHAVIOUR
    # ---------------------------------------------------------

    _section_heading(
        "Purchase Behaviour",
        "Key purchase indicators for the selected customer.",
    )

    purchase_1, purchase_2 = st.columns(
        2,
        gap="small",
    )

    with purchase_1:
        _metric_card(
            "AVERAGE BILL",
            _format_currency(customer.get("average_bill")),
            "Average value per transaction",
        )

    with purchase_2:
        _metric_card(
            "CUSTOMER TENURE",
            _format_days(customer.get("customer_tenure")),
            "Days since customer registration",
        )

    # ---------------------------------------------------------
    # CAMPAIGN ENGAGEMENT
    # ---------------------------------------------------------

    _section_heading(
        "Campaign Engagement",
        "Historical campaign reach, interaction and redemption behaviour.",
    )

    campaign_1, campaign_2, campaign_3, campaign_4 = st.columns(
        4,
        gap="small",
    )

    with campaign_1:
        _metric_card(
            "CAMPAIGNS SENT",
            _format_number(customer.get("campaigns_sent")),
            "Campaign messages sent",
        )

    with campaign_2:
        _metric_card(
            "DELIVERED",
            _format_number(customer.get("campaigns_delivered")),
            "Campaign messages delivered",
        )

    with campaign_3:
        _metric_card(
            "CLICKED",
            _format_number(customer.get("campaign_clicks")),
            "Campaign interactions",
        )

    with campaign_4:
        _metric_card(
            "DELIVERY RATE",
            _format_percent(customer.get("delivery_rate")),
            "Delivered messages as a share of sent",
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    engagement_1, engagement_2, engagement_3 = st.columns(
        3,
        gap="small",
    )

    with engagement_1:
        _metric_card(
            "CLICK RATE",
            _format_percent(customer.get("click_rate")),
            "Clicks as a share of delivered",
        )

    with engagement_2:
        _metric_card(
            "REDEMPTION RATE",
            _format_percent(customer.get("redemption_rate")),
            "Redemptions as a share of delivered",
        )

    with engagement_3:
        _metric_card(
            "PREVIOUS REDEMPTIONS",
            _format_number(customer.get("previous_redemptions")),
            "Historical campaign redemptions",
        )

    # ---------------------------------------------------------
    # PROFILE DETAILS
    # ---------------------------------------------------------

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    with st.expander("Profile Details"):
        _details_table(customer)