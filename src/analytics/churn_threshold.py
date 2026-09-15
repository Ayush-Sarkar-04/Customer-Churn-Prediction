import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


DEFAULT_THRESHOLDS = np.round(
    np.arange(0.20, 0.81, 0.05),
    2,
)

BASELINE_THRESHOLD = 0.50


def calculate_threshold_sensitivity(
    y_true,
    y_prob,
    thresholds=None,
):
    """
    Evaluate churn classification performance across
    multiple probability thresholds.
    """

    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if len(y_true) == 0:
        raise ValueError("y_true cannot be empty.")

    if len(y_prob) == 0:
        raise ValueError("y_prob cannot be empty.")

    if len(y_true) != len(y_prob):
        raise ValueError(
            "y_true and y_prob must have the same length."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(
            "y_true must contain only binary values: 0 or 1."
        )

    if not np.isfinite(y_prob).all():
        raise ValueError(
            "y_prob must contain only finite values."
        )

    if ((y_prob < 0) | (y_prob > 1)).any():
        raise ValueError(
            "y_prob values must be between 0 and 1."
        )

    thresholds = np.asarray(thresholds, dtype=float)

    if len(thresholds) == 0:
        raise ValueError(
            "At least one threshold is required."
        )

    if not np.isfinite(thresholds).all():
        raise ValueError(
            "Thresholds must contain only finite values."
        )

    if ((thresholds < 0) | (thresholds > 1)).any():
        raise ValueError(
            "Thresholds must be between 0 and 1."
        )

    if len(np.unique(thresholds)) != len(thresholds):
        raise ValueError(
            "Thresholds must not contain duplicates."
        )

    results = []

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        results.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy_score(
                    y_true, y_pred
                ),
                "precision": precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                ),
                "predicted_churn_count": int(
                    y_pred.sum()
                ),
                "predicted_churn_rate": float(
                    y_pred.mean()
                ),
                "non_churn_count": int(
                    (y_pred == 0).sum()
                ),
                "non_churn_rate": float(
                    (y_pred == 0).mean()
                ),
                "is_baseline": bool(
                    np.isclose(
                        threshold,
                        BASELINE_THRESHOLD
                    )
                ),
            }
        )

    return pd.DataFrame(results)