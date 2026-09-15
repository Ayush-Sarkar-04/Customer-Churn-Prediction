"""
Model Calibration / Reliability Analysis

Evaluates whether predicted churn probabilities broadly
correspond to observed churn frequencies.
"""

import numpy as np
import pandas as pd

from sklearn.metrics import brier_score_loss


DEFAULT_CALIBRATION_BINS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

DEFAULT_BIN_LABELS = [
    "0–20%",
    "20–40%",
    "40–60%",
    "60–80%",
    "80–100%",
]


def _validate_inputs(y_true, y_prob):
    """Validate observed churn labels and predicted probabilities."""

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

    return y_true.astype(int), y_prob.astype(float)


def calculate_calibration_reliability(
    y_true,
    y_prob,
    bins=None,
    bin_labels=None,
):
    """
    Calculate probability-bin reliability statistics.

    For each probability bin, calculates:

    - Mean predicted probability
    - Observed churn rate
    - Calibration gap
    - Absolute calibration gap
    - Customer count

    Calibration gap is:

        observed churn rate - mean predicted probability

    A value close to zero indicates closer agreement between
    predicted probability and observed churn frequency.
    """

    y_true, y_prob = _validate_inputs(
        y_true,
        y_prob,
    )

    if bins is None:
        bins = DEFAULT_CALIBRATION_BINS

    if bin_labels is None:
        bin_labels = DEFAULT_BIN_LABELS

    bins = np.asarray(bins, dtype=float)

    if len(bins) < 2:
        raise ValueError(
            "At least two bin boundaries are required."
        )

    if not np.isfinite(bins).all():
        raise ValueError(
            "Calibration bins must contain only finite values."
        )

    if not np.all(np.diff(bins) > 0):
        raise ValueError(
            "Calibration bins must be strictly increasing."
        )

    if bins[0] < 0 or bins[-1] > 1:
        raise ValueError(
            "Calibration bins must fall between 0 and 1."
        )

    if len(bin_labels) != len(bins) - 1:
        raise ValueError(
            "Number of bin labels must equal "
            "number of bins minus one."
        )

    calibration_data = pd.DataFrame(
        {
            "predicted_probability": y_prob,
            "observed_churn": y_true,
        }
    )

    calibration_data["probability_bin"] = pd.cut(
        calibration_data["predicted_probability"],
        bins=bins,
        labels=bin_labels,
        include_lowest=True,
    )

    reliability = (
        calibration_data
        .groupby(
            "probability_bin",
            observed=False,
        )
        .agg(
            mean_predicted_probability=(
                "predicted_probability",
                "mean",
            ),
            observed_churn_rate=(
                "observed_churn",
                "mean",
            ),
            customers=(
                "observed_churn",
                "size",
            ),
        )
        .reset_index()
    )

    reliability["calibration_gap"] = (
        reliability["observed_churn_rate"]
        - reliability["mean_predicted_probability"]
    )

    reliability["absolute_calibration_gap"] = (
        reliability["calibration_gap"].abs()
    )

    return reliability


def calculate_calibration_summary(
    y_true,
    y_prob,
    bins=None,
    bin_labels=None,
):
    """
    Calculate overall calibration and reliability metrics.

    Returns:

    - Brier score
    - Mean predicted churn probability
    - Observed churn rate
    - Overall probability gap
    - Weighted mean absolute calibration gap
    - Number of observations
    """

    y_true, y_prob = _validate_inputs(
        y_true,
        y_prob,
    )

    reliability = calculate_calibration_reliability(
        y_true,
        y_prob,
        bins=bins,
        bin_labels=bin_labels,
    )

    brier_score = float(
        brier_score_loss(
            y_true,
            y_prob,
        )
    )

    mean_predicted_probability = float(
        y_prob.mean()
    )

    observed_churn_rate = float(
        y_true.mean()
    )

    overall_probability_gap = (
        observed_churn_rate
        - mean_predicted_probability
    )

    total_customers = len(y_true)

    # Empty probability bins have NaN calibration gaps.
    # Only populated bins should contribute to the
    # weighted calibration error.
    populated_bins = reliability[
        reliability["customers"] > 0
    ]

    if populated_bins.empty:
        weighted_mean_absolute_calibration_gap = 0.0
    else:
        weighted_mean_absolute_calibration_gap = float(
            np.average(
                populated_bins["absolute_calibration_gap"],
                weights=populated_bins["customers"],
            )
        )

    return {
        "brier_score": brier_score,
        "mean_predicted_probability": (
            mean_predicted_probability
        ),
        "observed_churn_rate": (
            observed_churn_rate
        ),
        "overall_probability_gap": (
            overall_probability_gap
        ),
        "weighted_mean_absolute_calibration_gap": (
            weighted_mean_absolute_calibration_gap
        ),
        "observations": total_customers,
    }