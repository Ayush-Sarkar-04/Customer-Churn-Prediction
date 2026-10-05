import pandas as pd
import pytest

from src.ml.risk import classify_risk, add_risk_level


def test_classify_risk():

    assert classify_risk(0.10) == "Low"
    assert classify_risk(0.30) == "Medium"
    assert classify_risk(0.60) == "High"
    assert classify_risk(0.80) == "Very High"


def test_classify_risk_boundaries():

    assert classify_risk(0.00) == "Low"
    assert classify_risk(0.249999) == "Low"
    assert classify_risk(0.25) == "Medium"
    assert classify_risk(0.499999) == "Medium"
    assert classify_risk(0.50) == "High"
    assert classify_risk(0.749999) == "High"
    assert classify_risk(0.75) == "Very High"
    assert classify_risk(1.00) == "Very High"


def test_classify_risk_missing_probability():

    assert classify_risk(float("nan")) == "Unknown"


def test_classify_risk_invalid_probability():

    with pytest.raises(ValueError):
        classify_risk(-0.01)

    with pytest.raises(ValueError):
        classify_risk(1.01)


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


def test_add_risk_level_with_missing_probability():

    df = pd.DataFrame(
        {
            "customer_id": ["C001", "C002"],
            "churn_probability": [0.20, float("nan")],
        }
    )

    result = add_risk_level(df)

    assert list(result["risk_level"]) == [
        "Low",
        "Unknown",
    ]
