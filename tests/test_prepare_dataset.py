from src.ml.prepare_dataset import (
    load_ml_dataset,
    prepare_ml_dataset
)


DATA_PATH = "data/training/customer_features.csv"


def test_prepare_ml_dataset():

    df = load_ml_dataset(DATA_PATH)

    X, y = prepare_ml_dataset(df)

    print("\nML DATASET:")
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    print("\nFEATURE COLUMNS:")
    print(X.columns.tolist())

    print("\nTARGET DISTRIBUTION:")
    print(y.value_counts())

    assert len(X) == len(y)
    assert X.shape[1] == 15
    assert set(y.unique()).issubset({0, 1})