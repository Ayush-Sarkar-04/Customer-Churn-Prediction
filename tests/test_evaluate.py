from src.ml.prepare_dataset import (
    load_ml_dataset,
    prepare_ml_dataset
)

from src.ml.train import (
    split_dataset,
    train_models
)

from src.ml.evaluate import (
    evaluate_all_models,
    select_best_model
)


DATA_PATH = "data/training/customer_features.csv"


def test_evaluate_models():

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

    # Evaluate models
    results = evaluate_all_models(
        trained_models,
        X_test,
        y_test
    )

    print("\nMODEL EVALUATION")
    print("================")

    for name, metrics in results.items():

        print(f"\n{name}")

        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")

        print("Confusion Matrix:")
        print(metrics["confusion_matrix"])

    # Verify all models were evaluated
    assert len(results) == 3

    assert "Logistic Regression" in results
    assert "Decision Tree" in results
    assert "Random Forest" in results

    # Verify required metrics exist
    for metrics in results.values():

        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1_score"] <= 1
        assert 0 <= metrics["roc_auc"] <= 1

        assert metrics["confusion_matrix"].shape == (2, 2)

    # Select best model
    best_model = select_best_model(
        results,
        metric="roc_auc"
    )

    print(f"\nBEST MODEL (ROC-AUC): {best_model}")

    assert best_model in trained_models