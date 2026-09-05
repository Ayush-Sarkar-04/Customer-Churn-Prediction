from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.ml.prepare_dataset import (
    load_ml_dataset,
    prepare_ml_dataset
)


DATA_PATH = "data/training/customer_features.csv"


def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.

    Stratification preserves the churn class distribution.
    """

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def create_models():
    """
    Create the three classification models.
    """

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=42
            ))
        ]),

        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }

    return models


def train_models(X_train, y_train):
    """
    Train all churn prediction models.
    """

    models = create_models()

    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def main():

    # Load dataset
    df = load_ml_dataset(DATA_PATH)

    # Prepare features and target
    X, y = prepare_ml_dataset(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train models
    trained_models = train_models(X_train, y_train)

    print("ML TRAINING COMPLETE")
    print("====================")

    print(f"Total records: {len(X)}")
    print(f"Training records: {len(X_train)}")
    print(f"Testing records: {len(X_test)}")

    print("\nTRAINING CHURN DISTRIBUTION:")
    print(y_train.value_counts())

    print("\nTESTING CHURN DISTRIBUTION:")
    print(y_test.value_counts())

    print("\nTRAINED MODELS:")

    for name in trained_models:
        print(f"- {name}")


if __name__ == "__main__":
    main()