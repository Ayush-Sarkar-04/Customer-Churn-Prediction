import joblib
import pandas as pd

from src.ml.prepare_dataset import prepare_ml_dataset
from src.ml.risk import assign_risk_levels


MODEL_PATH = "models/random_forest.joblib"


def load_model(model_path=MODEL_PATH):
    """
    Load a saved churn prediction model.
    """

    return joblib.load(model_path)


def predict_churn(model, X):
    """
    Generate churn predictions and churn probabilities.
    """

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return predictions, probabilities


def generate_predictions(X, model_path=MODEL_PATH):
    """
    Generate churn predictions from prepared features.
    """

    model = load_model(model_path)

    predictions, probabilities = predict_churn(
        model,
        X
    )

    results = pd.DataFrame({
        "churn_prediction": predictions,
        "churn_probability": probabilities
    })

    return results


def predict_customers(df, model_path=MODEL_PATH):
    """
    Generate complete customer-level churn predictions
    with risk classification.

    Output:
        customer_id
        churn_prediction
        churn_probability
        risk_level
    """

    # Prepare ML features
    X, _ = prepare_ml_dataset(df)

    # Generate predictions
    predictions = generate_predictions(
        X,
        model_path
    )

    # Keep customer IDs
    results = pd.DataFrame({
        "customer_id": df["customer_id"].values,
        "churn_prediction": predictions["churn_prediction"].values,
        "churn_probability": predictions["churn_probability"].values
    })

    # Assign risk levels
    results = assign_risk_levels(results)

    return results