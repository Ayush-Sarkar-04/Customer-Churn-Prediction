def classify_risk(churn_probability, churn_prediction):
    """
    Classify customers into churn risk categories.

    Risk levels:
        Low:       probability < 0.25
        Medium:    probability < 0.50
        High:      probability < 0.75
        Very High: probability >= 0.75
        Churned:   model prediction = 1
    """

    if churn_prediction == 1:
        return "Churned"

    if churn_probability < 0.25:
        return "Low"

    if churn_probability < 0.50:
        return "Medium"

    if churn_probability < 0.75:
        return "High"

    return "Very High"


def assign_risk_levels(prediction_df):
    """
    Add a risk level column to the prediction DataFrame.
    """

    result = prediction_df.copy()

    result["risk_level"] = result.apply(
        lambda row: classify_risk(
            row["churn_probability"],
            row["churn_prediction"]
        ),
        axis=1
    )

    return result