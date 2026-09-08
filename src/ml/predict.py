import os

import joblib
import pandas as pd

from src.ml.prepare_dataset import prepare_ml_dataset
from src.ml.risk import assign_risk_levels


MODEL_PATH = "models/random_forest.joblib"


def save_model(model, model_path):
    """
    Save a trained churn prediction model to disk.
    """

    directory = os.path.dirname(model_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    joblib.dump(model, model_path)

    return model_path


def load_model(model_path=MODEL_PATH):
    """
    Load a saved churn prediction model.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


def predict_churn(model, X):
    """
    Generate churn predictions and churn probabilities.

    Returns
    -------
    tuple
        predictions, probabilities
    """

    predictions = model.predict(X)

    probabilities = model.predict_proba(X)[:, 1]

    return predictions, probabilities


def generate_predictions(X, model_path=MODEL_PATH):
    """
    Generate churn predictions from prepared features.

    Returns
    -------
    pandas.DataFrame
        churn_prediction
        churn_probability
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
    """

    # ---------------------------------------------------------
    # 1. Prepare ML features
    # ---------------------------------------------------------

    X, _ = prepare_ml_dataset(df)

    # ---------------------------------------------------------
    # 2. Generate predictions
    # ---------------------------------------------------------

    predictions = generate_predictions(
        X,
        model_path
    )

    # ---------------------------------------------------------
    # 3. Align customer IDs with prepared rows
    # ---------------------------------------------------------

    prepared_df = df.loc[X.index].copy()

    results = pd.DataFrame({
        "customer_id": prepared_df["customer_id"].values,
        "churn_prediction": predictions["churn_prediction"].values,
        "churn_probability": predictions["churn_probability"].values
    })

    # ---------------------------------------------------------
    # 4. Assign risk levels
    # ---------------------------------------------------------

    results["risk_level"] = assign_risk_levels(
        results["churn_probability"],
        results["churn_prediction"]
    )

    return results