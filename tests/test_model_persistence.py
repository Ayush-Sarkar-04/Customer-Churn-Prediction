import os

from src.ml.predict import (
    save_model,
    load_model
)

from src.ml.train import (
    split_dataset,
    train_models
)

from src.ml.prepare_dataset import (
    load_ml_dataset,
    prepare_ml_dataset
)


DATA_PATH = "data/training/customer_features.csv"


def test_model_persistence():

    # Load dataset
    df = load_ml_dataset(DATA_PATH)

    # Prepare features and target
    X, y = prepare_ml_dataset(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train models
    trained_models = train_models(
        X_train,
        y_train
    )

    # Select Random Forest
    model = trained_models["Random Forest"]

    # Save model
    model_path = save_model(
        model,
        "test_random_forest"
    )

    print("\nMODEL PERSISTENCE TEST")
    print("======================")
    print(f"Model saved to: {model_path}")

    # Verify file exists
    assert os.path.exists(model_path)

    # Load model
    loaded_model = load_model(model_path)

    print("Model loaded successfully.")

    # Verify loaded model can predict
    predictions = loaded_model.predict(X_test)

    print(f"Predictions generated: {len(predictions)}")

    assert len(predictions) == len(X_test)

    # Clean up test model
    os.remove(model_path)

    print("Temporary test model removed.")

    assert not os.path.exists(model_path)