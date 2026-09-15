import numpy as np
import pandas as pd
import pytest

from src.analytics.model_calibration import (
    calculate_calibration_reliability,
    calculate_calibration_summary,
    DEFAULT_CALIBRATION_BINS,
    DEFAULT_BIN_LABELS,
)


class TestCalibrationReliability:

    def test_default_bins_are_used(self):
        y_true = [0, 1, 0, 1, 1]
        y_prob = [0.10, 0.70, 0.30, 0.90, 0.50]

        result = calculate_calibration_reliability(
            y_true,
            y_prob,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(DEFAULT_CALIBRATION_BINS) - 1
        assert result["probability_bin"].tolist() == DEFAULT_BIN_LABELS

    def test_custom_bins_are_used(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.10, 0.40, 0.70, 0.90]

        bins = [0.0, 0.5, 1.0]
        labels = ["0–50%", "50–100%"]

        result = calculate_calibration_reliability(
            y_true,
            y_prob,
            bins=bins,
            bin_labels=labels,
        )

        assert len(result) == 2
        assert result["probability_bin"].tolist() == labels

    def test_output_columns(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.10, 0.40, 0.70, 0.90]

        result = calculate_calibration_reliability(
            y_true,
            y_prob,
        )

        expected_columns = {
            "probability_bin",
            "mean_predicted_probability",
            "observed_churn_rate",
            "customers",
            "calibration_gap",
            "absolute_calibration_gap",
        }

        assert set(result.columns) == expected_columns

    def test_customer_counts_are_correct(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.10, 0.20, 0.70, 0.80]

        result = calculate_calibration_reliability(
            y_true,
            y_prob,
        )

        assert result["customers"].sum() == 4

    def test_mean_predicted_probability_is_correct(self):
        y_true = [0, 1]
        y_prob = [0.10, 0.30]

        result = calculate_calibration_reliability(
            y_true,
            y_prob,
        )

        first_bin = result.iloc[0]

        assert first_bin[
            "mean_predicted_probability"
        ] == pytest.approx(0.10)

    def test_observed_churn_rate_is_correct(self):
        y_true = [0, 1, 1]
        y_prob = [0.10, 0.20, 0.30]

        result = calculate_calibration_reliability(
            y_true,
            y_prob,
        )

        first_bin = result.iloc[0]

        # 0.10 and 0.20 are included in the 0–20% bin.
        # Their observed outcomes are 0 and 1.
        assert first_bin[
            "observed_churn_rate"
        ] == pytest.approx(0.50)

    def test_calibration_gap_is_observed_minus_predicted(self):
        y_true = [0, 1]
        y_prob = [0.10, 0.30]

        result = calculate_calibration_reliability(
            y_true,
            y_prob,
        )

        row = result.iloc[0]

        expected_gap = 0.0 - 0.10

        assert row["calibration_gap"] == pytest.approx(
            expected_gap
        )

    def test_absolute_calibration_gap_is_non_negative(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.10, 0.30, 0.70, 0.90]

        result = calculate_calibration_reliability(
            y_true,
            y_prob,
        )

        populated_bins = result[
            result["customers"] > 0
        ]

        assert (
            populated_bins["absolute_calibration_gap"] >= 0
        ).all()

    def test_empty_bins_are_retained(self):
        y_true = [0, 1]
        y_prob = [0.10, 0.90]

        result = calculate_calibration_reliability(
            y_true,
            y_prob,
        )

        assert len(result) == 5

        assert (
            result["customers"] == 0
        ).sum() == 3

    def test_boundary_probability_zero_is_included(self):
        y_true = [0]
        y_prob = [0.0]

        result = calculate_calibration_reliability(
            y_true,
            y_prob,
        )

        assert result["customers"].sum() == 1

    def test_boundary_probability_one_is_included(self):
        y_true = [1]
        y_prob = [1.0]

        result = calculate_calibration_reliability(
            y_true,
            y_prob,
        )

        assert result["customers"].sum() == 1

    def test_empty_y_true_is_rejected(self):
        with pytest.raises(ValueError):
            calculate_calibration_reliability(
                [],
                [],
            )

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            calculate_calibration_reliability(
                [0, 1, 0],
                [0.20, 0.80],
            )

    def test_non_binary_y_true_is_rejected(self):
        with pytest.raises(ValueError):
            calculate_calibration_reliability(
                [0, 1, 2],
                [0.20, 0.80, 0.40],
            )

    def test_probability_outside_range_is_rejected(self):
        with pytest.raises(ValueError):
            calculate_calibration_reliability(
                [0, 1],
                [-0.10, 1.10],
            )

    def test_invalid_bins_are_rejected(self):
        with pytest.raises(ValueError):
            calculate_calibration_reliability(
                [0, 1],
                [0.20, 0.80],
                bins=[0.0, 0.7, 0.6, 1.0],
                bin_labels=[
                    "0–70%",
                    "70–60%",
                    "60–100%",
                ],
            )

    def test_bin_labels_must_match_bins(self):
        with pytest.raises(ValueError):
            calculate_calibration_reliability(
                [0, 1],
                [0.20, 0.80],
                bins=[0.0, 0.5, 1.0],
                bin_labels=["Only One"],
            )


class TestCalibrationSummary:

    def test_summary_returns_expected_keys(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.10, 0.80, 0.30, 0.70]

        result = calculate_calibration_summary(
            y_true,
            y_prob,
        )

        expected_keys = {
            "brier_score",
            "mean_predicted_probability",
            "observed_churn_rate",
            "overall_probability_gap",
            "weighted_mean_absolute_calibration_gap",
            "observations",
        }

        assert set(result.keys()) == expected_keys

    def test_summary_observation_count_is_correct(self):
        y_true = [0, 1, 0, 1, 1]
        y_prob = [0.10, 0.80, 0.20, 0.70, 0.60]

        result = calculate_calibration_summary(
            y_true,
            y_prob,
        )

        assert result["observations"] == 5

    def test_mean_predicted_probability_is_correct(self):
        y_true = [0, 1]
        y_prob = [0.20, 0.80]

        result = calculate_calibration_summary(
            y_true,
            y_prob,
        )

        assert result[
            "mean_predicted_probability"
        ] == pytest.approx(0.50)

    def test_observed_churn_rate_is_correct(self):
        y_true = [0, 1, 1, 0]
        y_prob = [0.20, 0.80, 0.70, 0.30]

        result = calculate_calibration_summary(
            y_true,
            y_prob,
        )

        assert result[
            "observed_churn_rate"
        ] == pytest.approx(0.50)

    def test_overall_probability_gap_is_correct(self):
        y_true = [0, 1]
        y_prob = [0.20, 0.80]

        result = calculate_calibration_summary(
            y_true,
            y_prob,
        )

        assert result[
            "overall_probability_gap"
        ] == pytest.approx(0.0)

    def test_perfect_predictions_have_zero_brier_score(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.0, 1.0, 0.0, 1.0]

        result = calculate_calibration_summary(
            y_true,
            y_prob,
        )

        assert result["brier_score"] == pytest.approx(0.0)

    def test_brier_score_is_between_zero_and_one(self):
        y_true = [0, 1, 0, 1, 1]
        y_prob = [0.10, 0.80, 0.40, 0.60, 0.90]

        result = calculate_calibration_summary(
            y_true,
            y_prob,
        )

        assert 0 <= result["brier_score"] <= 1

    def test_calibration_gap_is_zero_when_means_match(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.20, 0.80, 0.20, 0.80]

        result = calculate_calibration_summary(
            y_true,
            y_prob,
        )

        assert result[
            "overall_probability_gap"
        ] == pytest.approx(0.0)

    def test_weighted_absolute_calibration_gap_is_non_negative(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.10, 0.80, 0.40, 0.60]

        result = calculate_calibration_summary(
            y_true,
            y_prob,
        )

        assert (
            result[
                "weighted_mean_absolute_calibration_gap"
            ]
            >= 0
        )

    def test_weighted_absolute_calibration_gap_is_finite(self):
        y_true = [0, 1]
        y_prob = [0.10, 0.90]

        result = calculate_calibration_summary(
            y_true,
            y_prob,
        )

        assert np.isfinite(
            result[
                "weighted_mean_absolute_calibration_gap"
            ]
        )

    def test_input_arrays_are_not_modified(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.10, 0.80, 0.30, 0.70])

        original_true = y_true.copy()
        original_prob = y_prob.copy()

        calculate_calibration_summary(
            y_true,
            y_prob,
        )

        np.testing.assert_array_equal(
            y_true,
            original_true,
        )

        np.testing.assert_array_equal(
            y_prob,
            original_prob,
        )