import numpy as np
import pandas as pd


FAILURE_MODE_LABELS = {
    "TP": "True Positive",
    "TN": "True Negative",
    "FP": "False Positive",
    "FN": "False Negative",
}


def _validate_inputs(y_true, y_pred, y_prob=None):
    """Validate model evaluation inputs."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        raise ValueError("y_true cannot be empty.")

    if len(y_pred) == 0:
        raise ValueError("y_pred cannot be empty.")

    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(
            "y_true must contain only binary values: 0 or 1."
        )

    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError(
            "y_pred must contain only binary values: 0 or 1."
        )

    if y_prob is not None:
        y_prob = np.asarray(y_prob)

        if len(y_prob) != len(y_true):
            raise ValueError(
                "y_prob must have the same length as "
                "y_true and y_pred."
            )

        if not np.isfinite(y_prob).all():
            raise ValueError(
                "y_prob must contain only finite values."
            )

        if ((y_prob < 0) | (y_prob > 1)).any():
            raise ValueError(
                "y_prob values must be between 0 and 1."
            )

        y_prob = y_prob.astype(float)

    return (
        y_true.astype(int),
        y_pred.astype(int),
        y_prob,
    )


def classify_failure_modes(
    y_true,
    y_pred,
    y_prob=None,
):
    """
    Classify every prediction into a model outcome.

    Returns a DataFrame containing:

    - actual
    - predicted
    - predicted_probability
    - failure_mode
    - failure_type
    """

    y_true, y_pred, y_prob = _validate_inputs(
        y_true,
        y_pred,
        y_prob,
    )

    data = {
        "actual": y_true,
        "predicted": y_pred,
    }

    if y_prob is not None:
        data["predicted_probability"] = y_prob

    result = pd.DataFrame(data)

    result["failure_mode"] = np.select(
        [
            (result["actual"] == 1)
            & (result["predicted"] == 1),

            (result["actual"] == 0)
            & (result["predicted"] == 0),

            (result["actual"] == 0)
            & (result["predicted"] == 1),

            (result["actual"] == 1)
            & (result["predicted"] == 0),
        ],
        [
            "TP",
            "TN",
            "FP",
            "FN",
        ],
        default="UNKNOWN",
    )

    result["failure_type"] = result["failure_mode"].map(
        FAILURE_MODE_LABELS
    )

    return result


def calculate_failure_mode_analysis(
    y_true,
    y_pred,
    y_prob=None,
):
    """
    Calculate aggregate statistics for each prediction outcome.

    Returns one row for each:

    - True Positive
    - True Negative
    - False Positive
    - False Negative

    Metrics include:

    - count
    - rate
    - average predicted probability
    """

    classified = classify_failure_modes(
        y_true,
        y_pred,
        y_prob,
    )

    total_observations = len(classified)

    rows = []

    for mode_code, mode_label in FAILURE_MODE_LABELS.items():

        mode_data = classified[
            classified["failure_mode"] == mode_code
        ]

        count = len(mode_data)

        row = {
            "failure_mode": mode_code,
            "failure_type": mode_label,
            "count": count,
            "rate": (
                count / total_observations
                if total_observations > 0
                else 0.0
            ),
        }

        if y_prob is not None and count > 0:
            row["average_predicted_probability"] = float(
                mode_data["predicted_probability"].mean()
            )
        else:
            row["average_predicted_probability"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_failure_mode_summary(
    y_true,
    y_pred,
    y_prob=None,
):
    """
    Calculate a concise summary of model failure modes.

    Returns:

    - total observations
    - true positives
    - true negatives
    - false positives
    - false negatives
    - false positive rate
    - false negative rate
    - error rate
    - precision
    - recall
    - accuracy
    - missed churn count
    - unnecessary targeting count
    """

    y_true, y_pred, y_prob = _validate_inputs(
        y_true,
        y_pred,
        y_prob,
    )

    tp = int(
        ((y_true == 1) & (y_pred == 1)).sum()
    )

    tn = int(
        ((y_true == 0) & (y_pred == 0)).sum()
    )

    fp = int(
        ((y_true == 0) & (y_pred == 1)).sum()
    )

    fn = int(
        ((y_true == 1) & (y_pred == 0)).sum()
    )

    total = len(y_true)

    actual_churn = int(
        (y_true == 1).sum()
    )

    actual_non_churn = int(
        (y_true == 0).sum()
    )

    predicted_churn = int(
        (y_pred == 1).sum()
    )

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    false_positive_rate = (
        fp / actual_non_churn
        if actual_non_churn > 0
        else 0.0
    )

    false_negative_rate = (
        fn / actual_churn
        if actual_churn > 0
        else 0.0
    )

    error_rate = (
        (fp + fn) / total
        if total > 0
        else 0.0
    )

    return {
        "total_observations": total,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "actual_churn": actual_churn,
        "actual_non_churn": actual_non_churn,
        "predicted_churn": predicted_churn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "error_rate": error_rate,
        "missed_churn_count": fn,
        "unnecessary_targeting_count": fp,
    }


def get_failure_observations(
    y_true,
    y_pred,
    y_prob=None,
    failure_modes=None,
):
    """
    Return individual observations belonging to selected
    failure modes.

    By default, returns only False Positives and
    False Negatives.
    """

    if failure_modes is None:
        failure_modes = ["FP", "FN"]

    valid_modes = set(FAILURE_MODE_LABELS.keys())

    invalid_modes = set(failure_modes) - valid_modes

    if invalid_modes:
        raise ValueError(
            "Invalid failure modes: "
            + ", ".join(sorted(invalid_modes))
        )

    classified = classify_failure_modes(
        y_true,
        y_pred,
        y_prob,
    )

    return classified[
        classified["failure_mode"].isin(failure_modes)
    ].reset_index(drop=True)