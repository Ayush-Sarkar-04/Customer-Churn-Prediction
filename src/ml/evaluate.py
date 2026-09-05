from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained classification model.

    Returns:
        Dictionary containing classification metrics
        and the confusion matrix.
    """

    # Predictions
    y_pred = model.predict(X_test)

    # Prediction probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_test,
        y_prob
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm
    }


def evaluate_all_models(
    trained_models,
    X_test,
    y_test
):
    """
    Evaluate all trained models.

    Returns:
        Dictionary containing evaluation results
        for each model.
    """

    results = {}

    for name, model in trained_models.items():
        results[name] = evaluate_model(
            model,
            X_test,
            y_test
        )

    return results

def select_best_model(results, metric="roc_auc"):
    """
    Select the best-performing model based on a chosen metric.
    """

    if not results:
        raise ValueError("No model evaluation results available.")

    if metric not in results[next(iter(results))]:
        raise ValueError(
            f"Metric '{metric}' is not available in evaluation results."
        )

    best_model_name = max(
        results,
        key=lambda name: results[name][metric]
    )

    return best_model_name