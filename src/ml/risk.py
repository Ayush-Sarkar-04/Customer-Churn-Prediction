import pandas as pd


def classify_risk(churn_probability):
    """
    Classify customers into risk levels based on
    predicted churn probability.

    Risk thresholds:
        < 0.25  -> Low
        < 0.50  -> Medium
        < 0.75  -> High
        >= 0.75 -> Very High
    """

    if pd.isna(churn_probability):
        return "Low"

    probability = float(churn_probability)

    if not 0 <= probability <= 1:
        raise ValueError(
            "churn_probability must be between 0 and 1"
        )

    if probability < 0.25:
        return "Low"

    if probability < 0.50:
        return "Medium"

    if probability < 0.75:
        return "High"

    return "Very High"


def add_risk_level(df):
    """
    Add a risk_level column based on churn_probability.

    Risk level is determined only by predicted probability.
    Churn prediction remains a separate field.
    """

    result = df.copy()

    if "churn_probability" not in result.columns:
        raise ValueError(
            "DataFrame must contain churn_probability"
        )

    result["risk_level"] = result["churn_probability"].apply(
        classify_risk
    )

    return result


def assign_risk_levels(churn_probabilities, churn_predictions=None):
    """
    Assign risk levels to predicted churn probabilities.

    Parameters
    ----------
    churn_probabilities : array-like
        Predicted churn probabilities.

    churn_predictions : array-like, optional
        Binary churn predictions.

        Retained for compatibility with the existing
        prediction pipeline. It is intentionally not used
        to determine risk level.

    Returns
    -------
    pandas.Series
        Risk level for each customer.
    """

    probabilities = pd.Series(churn_probabilities)

    return probabilities.apply(classify_risk)