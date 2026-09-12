import joblib
import numpy as np
import pandas as pd

from src.data.custom_pipeline import build_custom_customer_analytics
from src.ml.prepare_dataset import FEATURE_COLUMNS


MODEL_PATH = "models/random_forest.joblib"


def make_custom_data():
    customers = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004", "C005"],
            "gender": ["M", "F", "M", "F", "M"],
            "age": [25, 31, 42, 28, 36],
            "city": ["Kolkata"] * 5,
            "registration_date": [
                "01-01-2024",
                "15-02-2024",
                "10-03-2024",
                "20-04-2024",
                "05-05-2024",
            ],
        }
    )

    transactions = pd.DataFrame(
        {
            "transaction_id": [
                "T001",
                "T002",
                "T003",
                "T004",
                "T005",
                "T006",
                "T007",
                "T008",
                "T009",
                "T010",
            ],
            "customer_id": [
                "C001",
                "C001",
                "C002",
                "C002",
                "C003",
                "C003",
                "C003",
                "C004",
                "C005",
                "C005",
            ],
            "transaction_date": [
                "01-06-2024",
                "01-07-2024",
                "15-06-2024",
                "15-07-2024",
                "20-06-2024",
                "20-07-2024",
                "20-08-2024",
                "10-08-2024",
                "05-06-2024",
                "05-08-2024",
            ],
            "bill_amount": [
                1000,
                1500,
                2000,
                2500,
                500,
                750,
                1250,
                3000,
                1200,
                1800,
            ],
            "outlet": ["Outlet 1"] * 10,
        }
    )

    campaigns = pd.DataFrame(
        {
            "campaign_id": [
                "CP001",
                "CP002",
                "CP003",
                "CP004",
                "CP005",
                "CP006",
                "CP007",
                "CP008",
            ],
            "customer_id": [
                "C001",
                "C001",
                "C002",
                "C003",
                "C003",
                "C004",
                "C005",
                "C005",
            ],
            "campaign_date": [
                "10-07-2024",
                "20-07-2024",
                "20-07-2024",
                "25-07-2024",
                "25-08-2024",
                "15-08-2024",
                "10-07-2024",
                "10-08-2024",
            ],
            "campaign_type": [
                "Birthday",
                "Discount",
                "Festival",
                "Loyalty",
                "Win-back",
                "New Product",
                "Discount",
                "Festival",
            ],
            "reward_type": ["Discount"] * 8,
            "reward_value": [100] * 8,
            "campaign_cost": [10] * 8,
            "sent": [1] * 8,
            "delivered": [1, 1, 1, 1, 1, 1, 1, 1],
            "clicked": [1, 0, 1, 1, 0, 0, 1, 1],
            "redeemed": [0, 0, 1, 0, 0, 0, 1, 0],
            "redemption_date": [
                None,
                None,
                "21-07-2024",
                None,
                None,
                None,
                "11-07-2024",
                None,
            ],
        }
    )

    return {
        "customers": customers,
        "transactions": transactions,
        "campaigns": campaigns,
    }


def test_custom_pipeline_returns_expected_rows():
    custom_data = make_custom_data()

    result = build_custom_customer_analytics(
        custom_data,
        model_path=MODEL_PATH,
    )

    assert len(result) == 5
    assert result["customer_id"].nunique() == 5


def test_custom_pipeline_produces_all_model_features():
    custom_data = make_custom_data()

    result = build_custom_customer_analytics(
        custom_data,
        model_path=MODEL_PATH,
    )

    missing = [
        column
        for column in FEATURE_COLUMNS
        if column not in result.columns
    ]

    assert missing == []


def test_custom_pipeline_produces_predictions():
    custom_data = make_custom_data()

    result = build_custom_customer_analytics(
        custom_data,
        model_path=MODEL_PATH,
    )

    assert "churn_prediction" in result.columns
    assert "churn_probability" in result.columns

    assert result["churn_prediction"].isin([0, 1]).all()

    assert result["churn_probability"].between(0, 1).all()


def test_custom_pipeline_produces_risk_levels():
    custom_data = make_custom_data()

    result = build_custom_customer_analytics(
        custom_data,
        model_path=MODEL_PATH,
    )

    allowed_risk_levels = {
        "Low",
        "Medium",
        "High",
        "Very High",
    }

    assert result["risk_level"].notna().all()
    assert set(result["risk_level"].unique()).issubset(
        allowed_risk_levels
    )


def test_custom_pipeline_produces_customer_segments():
    custom_data = make_custom_data()

    result = build_custom_customer_analytics(
        custom_data,
        model_path=MODEL_PATH,
    )

    allowed_segments = {
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "Regular Customers",
        "At Risk",
        "Inactive",
        "Lost",
    }

    assert "customer_segment" in result.columns
    assert result["customer_segment"].notna().all()
    assert set(result["customer_segment"].unique()).issubset(
        allowed_segments
    )


def test_custom_pipeline_produces_retention_priority():
    custom_data = make_custom_data()

    result = build_custom_customer_analytics(
        custom_data,
        model_path=MODEL_PATH,
    )

    required_columns = {
        "customer_value_percentile",
        "retention_priority_score",
        "retention_priority",
        "expected_revenue_at_risk",
    }

    assert required_columns.issubset(result.columns)

    assert result["retention_priority_score"].between(
        0, 100
    ).all()

    assert result["expected_revenue_at_risk"].notna().all()


def test_custom_pipeline_adds_custom_data_metadata():
    custom_data = make_custom_data()

    result = build_custom_customer_analytics(
        custom_data,
        model_path=MODEL_PATH,
    )

    assert (result["data_mode"] == "Custom Data").all()
    assert result["prediction_observation_date"].notna().all()


def test_custom_pipeline_uses_latest_observation_date():
    custom_data = make_custom_data()

    result = build_custom_customer_analytics(
        custom_data,
        model_path=MODEL_PATH,
    )

    expected_date = pd.Timestamp("2024-08-25")

    assert (
        result["prediction_observation_date"]
        == expected_date
    ).all()


def test_custom_pipeline_does_not_modify_input_data():
    custom_data = make_custom_data()

    customers_before = custom_data["customers"].copy(deep=True)
    transactions_before = custom_data["transactions"].copy(deep=True)
    campaigns_before = custom_data["campaigns"].copy(deep=True)

    build_custom_customer_analytics(
        custom_data,
        model_path=MODEL_PATH,
    )

    pd.testing.assert_frame_equal(
        custom_data["customers"],
        customers_before,
    )

    pd.testing.assert_frame_equal(
        custom_data["transactions"],
        transactions_before,
    )

    pd.testing.assert_frame_equal(
        custom_data["campaigns"],
        campaigns_before,
    )


def test_custom_pipeline_handles_small_dataset():
    custom_data = make_custom_data()

    # Keep only three customers to explicitly exercise
    # the small-dataset segmentation fallback.
    customer_ids = ["C001", "C002", "C003"]

    custom_data["customers"] = custom_data["customers"][
        custom_data["customers"]["customer_id"].isin(customer_ids)
    ].copy()

    custom_data["transactions"] = custom_data["transactions"][
        custom_data["transactions"]["customer_id"].isin(customer_ids)
    ].copy()

    custom_data["campaigns"] = custom_data["campaigns"][
        custom_data["campaigns"]["customer_id"].isin(customer_ids)
    ].copy()

    result = build_custom_customer_analytics(
        custom_data,
        model_path=MODEL_PATH,
    )

    assert len(result) == 3
    assert result["customer_segment"].notna().all()
    assert result["churn_probability"].between(0, 1).all()


def test_custom_pipeline_model_is_loadable():
    model = joblib.load(MODEL_PATH)

    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")