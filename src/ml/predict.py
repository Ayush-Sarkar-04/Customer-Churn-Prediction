import os

import joblib
import pandas as pd

from src.ml.prepare_dataset import prepare_ml_dataset
from src.ml.risk import assign_risk_levels


MODEL_PATH = "models/random_forest.joblib"


def save_model(model, model_path):
    """
    Save a trained churn prediction model to disk.

    Parameters
    ----------
    model : object
        Trained scikit-learn model.

    model_path : str
        Destination path for the saved model.
    """

    directory = os.path.dirname(model_path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    joblib.dump(model, model_path)

    return model_path


def load_model(model_path=MODEL_PATH):
    """
    Load a saved churn prediction model.

    Parameters
    ----------
    model_path : str
        Path to the saved model.

    Returns
    -------
    object
        Loaded trained model.
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    return joblib.load(model_path)


def predict_churn(model, X):
    """
    Generate churn predictions and churn probabilities.

    Parameters
    ----------
    model : object
        Trained churn prediction model.

    X : pandas.DataFrame
        Prepared ML feature dataset.

    Returns
    -------
    tuple
        Churn predictions and churn probabilities.
    """

    predictions = model.predict(X)
    probabilities = model.predict_proba(X)[:, 1]

    return predictions, probabilities


def generate_predictions(X, model_path=MODEL_PATH):
    """
    Generate churn predictions from prepared features.

    Parameters
    ----------
    X : pandas.DataFrame
        Prepared ML feature dataset.

    model_path : str
        Path to the saved churn prediction model.

    Returns
    -------
    pandas.DataFrame
        Prediction results containing:
        - churn_prediction
        - churn_probability
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

    Parameters
    ----------
    df : pandas.DataFrame
        Customer feature / observation dataset.

    model_path : str
        Path to the saved churn prediction model.

    Returns
    -------
    pandas.DataFrame
        Customer prediction results containing:
        - customer_id
        - churn_prediction
        - churn_probability
        - risk_level
    """

    # ---------------------------------------------------------
    # 1. Prepare ML features
    # ---------------------------------------------------------
    X, _ = prepare_ml_dataset(df)

    # ---------------------------------------------------------
    # 2. Generate churn predictions
    # ---------------------------------------------------------
    predictions = generate_predictions(
        X,
        model_path
    )

    # ---------------------------------------------------------
    # 3. Keep customer IDs
    # ---------------------------------------------------------
    results = pd.DataFrame({
        "customer_id": df["customer_id"].values,
        "churn_prediction": predictions["churn_prediction"].values,
        "churn_probability": predictions["churn_probability"].values
    })

    # ---------------------------------------------------------
    # 4. Assign risk levels
    # ---------------------------------------------------------
    results = assign_risk_levels(results)

    return results