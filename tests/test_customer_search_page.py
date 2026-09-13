import pandas as pd

from streamlit.testing.v1 import AppTest


def create_customer_analytics():
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "recency": [10, 25, 60],
            "frequency": [20, 15, 5],
            "monetary": [5000.0, 4000.0, 1500.0],
            "average_bill": [250.0, 266.67, 300.0],
            "average_purchase_gap": [5.0, 8.0, 15.0],
            "rfm_score": [12, 10, 6],
            "customer_segment": [
                "Champions",
                "Loyal Customers",
                "At Risk",
            ],
            "churn_prediction": [0, 0, 1],
            "churn_probability": [0.05, 0.20, 0.75],
            "risk_level": ["Low", "Low", "Very High"],
            "campaigns_sent": [10, 8, 12],
            "campaigns_delivered": [9, 7, 10],
            "campaign_clicks": [4, 3, 2],
            "campaigns_redeemed": [2, 1, 0],
            "delivery_rate": [0.90, 0.875, 0.8333],
            "click_rate": [0.4444, 0.4286, 0.20],
            "redemption_rate": [0.2222, 0.1429, 0.0],
            "previous_redemptions": [3, 2, 0],
        }
    )


def build_app_code(df):
    return f"""
import pandas as pd
from app.customer_search_page import render_customer_search_page

customer_analytics = pd.DataFrame(
    {df.to_dict(orient="list")!r}
)

render_customer_search_page(
    customer_analytics
)
"""


def test_customer_search_page_renders():
    df = create_customer_analytics()

    at = AppTest.from_string(build_app_code(df))
    at.run()

    assert not at.exception

    assert len(at.text_input) == 1
    assert at.text_input[0].label == "Search Customer"


def test_customer_search_page_exact_customer():
    df = create_customer_analytics()

    at = AppTest.from_string(build_app_code(df))
    at.run()

    assert not at.exception

    at.text_input[0].set_value("C003")
    at.run()

    assert not at.exception

    # Verify that the selected customer's profile is rendered.
    assert len(at.metric) > 0 or len(at.dataframe) > 0


def test_customer_search_page_no_results():
    df = create_customer_analytics()

    at = AppTest.from_string(build_app_code(df))
    at.run()

    at.text_input[0].set_value("C999")
    at.run()

    assert not at.exception

    assert len(at.warning) == 1
    assert (
        "No customer IDs found matching 'C999'."
        in at.warning[0].value
    )