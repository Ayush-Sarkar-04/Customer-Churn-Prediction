"""
Segment-Level Model Performance Audit

Evaluates churn-model performance separately across customer segments.

The purpose is to identify whether model performance varies materially
between segments instead of relying only on overall model metrics.
"""

import numpy as np
import pandas as pd


REQUIRED_SEGMENT_COLUMN = "segment"


def _validate_inputs(y_true, y_pred, segments, y_prob=None):
    """Validate inputs used for segment-level model evaluation."""

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Preserve missing values and mixed segment types.
    segments = np.asarray(segments, dtype=object)

    if len(y_true) == 0:
        raise ValueError("y_true cannot be empty.")

    if len(y_pred) != len(y_true):
        raise ValueError(
            "y_pred must have the same length as y_true."
        )

    if len(segments) != len(y_true):
        raise ValueError(
            "segments must have the same length as y_true."
        )

    if not np.isin(y_true, [0, 1]).all():
        raise ValueError(
            "y_true must contain only binary values: 0 or 1."
        )

    if not np.isin(y_pred, [0, 1]).all():
        raise ValueError(
            "y_pred must contain only binary values: 0 or 1."
        )

    if pd.isna(segments).any():
        raise ValueError(
            "segments cannot contain missing values."
        )

    if y_prob is not None:
        y_prob = np.asarray(y_prob)

        if len(y_prob) != len(y_true):
            raise ValueError(
                "y_prob must have the same length as y_true."
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
        segments,
        y_prob,
    )


def calculate_segment_model_audit(
    y_true,
    y_pred,
    segments,
    y_prob=None,
):
    """
    Calculate model performance separately for each segment.

    Metrics returned:

    - segment
    - sample_count
    - actual_churn_rate
    - predicted_churn_rate
    - accuracy
    - precision
    - recall
    - f1_score
    - false_positive_rate
    - false_negative_rate
    - average_predicted_probability
    - brier_score
    """

    (
        y_true,
        y_pred,
        segments,
        y_prob,
    ) = _validate_inputs(
        y_true,
        y_pred,
        segments,
        y_prob,
    )

    rows = []

    unique_segments = pd.unique(segments)

    for segment in unique_segments:
        mask = segments == segment

        actual = y_true[mask]
        predicted = y_pred[mask]

        sample_count = len(actual)

        tp = int(
            ((actual == 1) & (predicted == 1)).sum()
        )

        tn = int(
            ((actual == 0) & (predicted == 0)).sum()
        )

        fp = int(
            ((actual == 0) & (predicted == 1)).sum()
        )

        fn = int(
            ((actual == 1) & (predicted == 0)).sum()
        )

        actual_churn_count = int(
            (actual == 1).sum()
        )

        predicted_churn_count = int(
            (predicted == 1).sum()
        )

        actual_non_churn_count = int(
            (actual == 0).sum()
        )

        accuracy = (
            (tp + tn) / sample_count
            if sample_count > 0
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

        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        false_positive_rate = (
            fp / actual_non_churn_count
            if actual_non_churn_count > 0
            else 0.0
        )

        false_negative_rate = (
            fn / actual_churn_count
            if actual_churn_count > 0
            else 0.0
        )

        row = {
            "segment": segment,
            "sample_count": sample_count,
            "actual_churn_count": actual_churn_count,
            "predicted_churn_count": predicted_churn_count,
            "actual_churn_rate": actual_churn_count / sample_count,
            "predicted_churn_rate": predicted_churn_count / sample_count,
            "true_positives": tp,
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
        }

        if y_prob is not None:
            probabilities = y_prob[mask]

            row["average_predicted_probability"] = float(
                probabilities.mean()
            )

            row["brier_score"] = float(
                np.mean(
                    (probabilities - actual) ** 2
                )
            )
        else:
            row["average_predicted_probability"] = np.nan
            row["brier_score"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def calculate_segment_model_summary(
    audit_result,
):
    """
    Summarize segment-level model performance.

    Returns overall segment-audit statistics and identifies
    the strongest and weakest segments by F1 score.
    """

    if not isinstance(audit_result, pd.DataFrame):
        raise TypeError(
            "audit_result must be a pandas DataFrame."
        )

    if audit_result.empty:
        return {
            "segment_count": 0,
            "total_observations": 0,
            "best_segment": None,
            "worst_segment": None,
            "best_f1_score": None,
            "worst_f1_score": None,
            "f1_score_range": 0.0,
        }

    required_columns = {
        "segment",
        "sample_count",
        "f1_score",
    }

    missing = required_columns - set(audit_result.columns)

    if missing:
        raise ValueError(
            "audit_result is missing required columns: "
            + ", ".join(sorted(missing))
        )

    best_index = audit_result["f1_score"].idxmax()
    worst_index = audit_result["f1_score"].idxmin()

    best_row = audit_result.loc[best_index]
    worst_row = audit_result.loc[worst_index]

    best_f1 = float(best_row["f1_score"])
    worst_f1 = float(worst_row["f1_score"])

    return {
        "segment_count": int(len(audit_result)),
        "total_observations": int(
            audit_result["sample_count"].sum()
        ),
        "best_segment": best_row["segment"],
        "worst_segment": worst_row["segment"],
        "best_f1_score": best_f1,
        "worst_f1_score": worst_f1,
        "f1_score_range": best_f1 - worst_f1,
    }


def rank_segments_by_metric(
    audit_result,
    metric="f1_score",
    ascending=False,
):
    """
    Rank segments according to a selected performance metric.
    """

    if not isinstance(audit_result, pd.DataFrame):
        raise TypeError(
            "audit_result must be a pandas DataFrame."
        )

    if metric not in audit_result.columns:
        raise ValueError(
            f"Metric column '{metric}' does not exist."
        )

    return audit_result.sort_values(
        by=[metric, "segment"],
        ascending=[ascending, True],
        kind="mergesort",
    ).reset_index(drop=True)