import pandas as pd

from src.ml.risk import (
    classify_risk,
    assign_risk_levels
)


def test_classify_risk():

    assert classify_risk(0.10, 0) == "Low"
    assert classify_risk(0.30, 0) == "Medium"
    assert classify_risk(0.60, 0) == "High"
    assert classify_risk(0.80, 0) == "Very High"

    # Churned prediction takes priority
    assert classify_risk(0.20, 1) == "Churned"


def test_assign_risk_levels():

    predictions = pd.DataFrame({
        "churn_prediction": [0, 0, 0, 0, 1],
        "churn_probability": [0.10, 0.30, 0.60, 0.80, 0.20]
    })

    result = assign_risk_levels(predictions)

    print("\nRISK CLASSIFICATION")
    print("===================")
    print(result)

    assert "risk_level" in result.columns

    assert result["risk_level"].tolist() == [
        "Low",
        "Medium",
        "High",
        "Very High",
        "Churned"
    ]

    assert len(result) == len(predictions)