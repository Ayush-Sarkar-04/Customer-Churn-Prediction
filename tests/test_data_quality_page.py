import pandas as pd

from streamlit.testing.v1 import AppTest


def create_valid_customer_data():
    return pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003"],
            "gender": ["Male", "Female", "Male"],
            "age": [25, 30, 28],
            "city": ["Delhi", "Mumbai", "Kolkata"],
            "registration_date": [
                "2026-01-10",
                "2026-02-15",
                "2026-03-20",
            ],
        }
    )


def create_valid_transaction_data():
    return pd.DataFrame(
        {
            "transaction_id": ["T001", "T002", "T003"],
            "customer_id": ["C001", "C002", "C003"],
            "transaction_date": [
                "2026-04-10",
                "2026-04-15",
                "2026-04-20",
            ],
            "bill_amount": [500.0, 750.0, 1000.0],
            "outlet": ["Delhi", "Mumbai", "Kolkata"],
        }
    )


def create_valid_campaign_data():
    return pd.DataFrame(
        {
            "campaign_id": ["CMP001", "CMP002", "CMP003"],
            "customer_id": ["C001", "C002", "C003"],
            "campaign_date": [
                "2026-04-01",
                "2026-04-05",
                "2026-04-10",
            ],
            "campaign_type": [
                "Birthday",
                "Discount",
                "Loyalty",
            ],
            "reward_type": [
                "Coupon",
                "Discount",
                "Points",
            ],
            "reward_value": [500.0, 10.0, 100.0],
            "campaign_cost": [50.0, 75.0, 40.0],
            "sent": [1, 1, 1],
            "delivered": [1, 1, 1],
            "clicked": [1, 0, 1],
            "redeemed": [1, 0, 1],
            "redemption_date": [
                "2026-04-02",
                None,
                "2026-04-12",
            ],
        }
    )


def test_data_quality_page_renders():
    customers = create_valid_customer_data()
    transactions = create_valid_transaction_data()
    campaigns = create_valid_campaign_data()

    app_code = f"""
import pandas as pd
from app.data_quality_page import render_data_quality_page

customers_df = pd.DataFrame({customers.to_dict(orient="list")!r})
transactions_df = pd.DataFrame({transactions.to_dict(orient="list")!r})
campaigns_df = pd.DataFrame({campaigns.to_dict(orient="list")!r})

render_data_quality_page(
    customers_df,
    transactions_df,
    campaigns_df,
)
"""

    at = AppTest.from_string(app_code)
    at.run()

    assert not at.exception

    assert len(at.title) == 1
    assert at.title[0].value == "Data Quality"

    assert len(at.success) == 1
    assert at.success[0].value == "✓ DATASET VALID"

    assert len(at.metric) == 4

    assert at.metric[0].label == "Files Checked"
    assert at.metric[0].value == "3"

    assert at.metric[1].label == "Total Rows"
    assert at.metric[1].value == "9"

    assert at.metric[2].label == "Checks Passed"
    assert at.metric[2].value == "19"

    assert at.metric[3].label == "Checks Failed"
    assert at.metric[3].value == "0"