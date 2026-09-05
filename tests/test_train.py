from src.ml.prepare_dataset import (
    load_ml_dataset,
    prepare_ml_dataset
)

from src.ml.train import (
    split_dataset,
    train_models
)


DATA_PATH = "data/training/customer_features.csv"


def test_train_models():

    df = load_ml_dataset(DATA_PATH)

    X, y = prepare_ml_dataset(df)

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    models = train_models(X_train, y_train)

    print("\nTRAINING TEST")
    print("=============")

    print("Training records:", len(X_train))
    print("Testing records:", len(X_test))

    print("\nMODELS TRAINED:")

    for name, model in models.items():
        print(f"- {name}")

    assert len(X_train) == 4012
    assert len(X_test) == 1003

    assert len(models) == 3

    assert "Logistic Regression" in models
    assert "Decision Tree" in models
    assert "Random Forest" in models