import pandas as pd

from src.ml.predict import (
    load_model,
    predict_churn,
    generate_predictions,
    predict_customers
)

from src.ml.prepare_dataset import (
    load_ml_dataset,
    prepare_ml_dataset
)


DATA_PATH = "data/training/customer_features.csv"
MODEL_PATH = "models/random_forest.joblib"


def test_predict_churn():

    # Load dataset
    df = load_ml_dataset(DATA_PATH)

    # Prepare features
    X, y = prepare_ml_dataset(df)

    # Load saved model
    model = load_model(MODEL_PATH)

    # Generate predictions
    predictions, probabilities = predict_churn(
        model,
        X
    )

    print("\nPREDICTION TEST")
    print("================")
    print(f"Total records: {len(X)}")
    print(f"Predictions generated: {len(predictions)}")
    print(f"Probabilities generated: {len(probabilities)}")

    print("\nSAMPLE PREDICTIONS:")
    print(predictions[:10])

    print("\nSAMPLE PROBABILITIES:")
    print(probabilities[:10])

    assert len(predictions) == len(X)
    assert len(probabilities) == len(X)

    assert set(predictions).issubset({0, 1})

    assert (
        (probabilities >= 0)
        & (probabilities <= 1)
    ).all()


def test_generate_predictions():

    # Load dataset
    df = load_ml_dataset(DATA_PATH)

    # Prepare features
    X, y = prepare_ml_dataset(df)

    # Generate prediction dataframe
    results = generate_predictions(
        X,
        MODEL_PATH
    )

    print("\nPREDICTION DATAFRAME")
    print("====================")
    print(results.head(10))

    assert isinstance(results, pd.DataFrame)

    assert len(results) == len(X)

    assert "churn_prediction" in results.columns
    assert "churn_probability" in results.columns


def test_predict_customers():

    # Load dataset
    df = load_ml_dataset(DATA_PATH)

    # Generate complete customer predictions
    results = predict_customers(
        df,
        MODEL_PATH
    )

    print("\nCUSTOMER CHURN PREDICTIONS")
    print("==========================")
    print(results.head(10))

    assert len(results) == len(df)

    assert "customer_id" in results.columns
    assert "churn_prediction" in results.columns
    assert "churn_probability" in results.columns
    assert "risk_level" in results.columns

    assert set(
        results["churn_prediction"].unique()
    ).issubset({0, 1})

    assert (
        (results["churn_probability"] >= 0)
        & (results["churn_probability"] <= 1)
    ).all()

    valid_risk_levels = {
        "Low",
        "Medium",
        "High",
        "Very High",
        "Churned"
    }

    assert set(
        results["risk_level"].unique()
    ).issubset(valid_risk_levels)