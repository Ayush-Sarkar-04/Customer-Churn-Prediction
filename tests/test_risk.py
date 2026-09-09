import pandas as pd

from src.ml.risk import classify_risk, add_risk_level


def test_classify_risk():

    assert classify_risk(0.10) == "Low"
    assert classify_risk(0.30) == "Medium"
    assert classify_risk(0.60) == "High"
    assert classify_risk(0.80) == "Very High"


def test_add_risk_level():

    df = pd.DataFrame(
        {
            "customer_id": ["C001", "C002", "C003", "C004"],
            "churn_probability": [
                0.10,
                0.30,
                0.60,
                0.80,
            ],
        }
    )

    result = add_risk_level(df)

    assert list(result["risk_level"]) == [
        "Low",
        "Medium",
        "High",
        "Very High",
    ]