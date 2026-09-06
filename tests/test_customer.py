import pandas as pd

from src.analytics.customer import (
    build_customer_analytics,
    calculate_segment_risk_counts,
)


DATA_PATH = "data/training/customer_features.csv"
MODEL_PATH = "models/random_forest.joblib"


def test_build_customer_analytics_requires_columns():
    df = pd.DataFrame(
        {
            "customer_id": ["C001"],
            "recency": [10],
            "frequency": [5],
        }
    )

    try:
        build_customer_analytics(df)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "monetary" in str(exc)


def test_build_customer_analytics_real_dataset():
    df = pd.read_csv(DATA_PATH)

    result = build_customer_analytics(
        df,
        model_path=MODEL_PATH
    )

    required_columns = {
        "customer_id",
        "recency",
        "frequency",
        "monetary",
        "recency_score",
        "frequency_score",
        "monetary_score",
        "rfm_score",
        "customer_segment",
        "churn_prediction",
        "churn_probability",
        "risk_level",
    }

    assert len(result) == len(df)
    assert required_columns.issubset(result.columns)

    assert result["churn_prediction"].isin([0, 1]).all()
    assert result["churn_probability"].between(0, 1).all()

    valid_segments = {
        "Champions",
        "Loyal Customers",
        "Potential Loyalists",
        "Regular Customers",
        "At Risk",
        "Inactive",
        "Lost",
    }

    assert result["customer_segment"].isin(valid_segments).all()

    valid_risk_levels = {
        "Low",
        "Medium",
        "High",
        "Very High",
        "Churned",
    }

    assert result["risk_level"].isin(valid_risk_levels).all()


def test_customer_ids_are_preserved():
    df = pd.read_csv(DATA_PATH)

    result = build_customer_analytics(
        df,
        model_path=MODEL_PATH
    )

    assert result["customer_id"].tolist() == df["customer_id"].tolist()


def test_calculate_segment_risk_counts():
    data = pd.DataFrame(
        {
            "customer_id": [
                "C001",
                "C002",
                "C003",
                "C004",
            ],
            "customer_segment": [
                "Champions",
                "Champions",
                "At Risk",
                "At Risk",
            ],
            "risk_level": [
                "Low",
                "Churned",
                "Medium",
                "Churned",
            ],
        }
    )

    result = calculate_segment_risk_counts(data)

    assert set(result.columns) == {
        "customer_segment",
        "risk_level",
        "customer_count",
    }

    assert result["customer_count"].sum() == 4


def test_calculate_segment_risk_counts_missing_columns():
    data = pd.DataFrame(
        {
            "customer_segment": ["Champions"],
        }
    )

    try:
        calculate_segment_risk_counts(data)
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "risk_level" in str(exc)