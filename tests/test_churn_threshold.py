import numpy as np
import pandas as pd
import pytest

from src.analytics.churn_threshold import (
    calculate_threshold_sensitivity,
    DEFAULT_THRESHOLDS,
    BASELINE_THRESHOLD,
)


class TestCalculateThresholdSensitivity:

    def test_default_thresholds_are_used(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.10, 0.80, 0.30, 0.60]

        result = calculate_threshold_sensitivity(
            y_true,
            y_prob,
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(DEFAULT_THRESHOLDS)
        assert result["threshold"].tolist() == pytest.approx(
            DEFAULT_THRESHOLDS.tolist()
        )

    def test_custom_thresholds_are_used(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.10, 0.80, 0.30, 0.60]

        thresholds = [0.30, 0.50, 0.70]

        result = calculate_threshold_sensitivity(
            y_true,
            y_prob,
            thresholds=thresholds,
        )

        assert result["threshold"].tolist() == thresholds

    def test_output_columns(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.10, 0.80, 0.30, 0.60]

        result = calculate_threshold_sensitivity(
            y_true,
            y_prob,
            thresholds=[0.50],
        )

        expected_columns = {
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "predicted_churn_count",
            "predicted_churn_rate",
            "non_churn_count",
            "non_churn_rate",
            "is_baseline",
        }

        assert set(result.columns) == expected_columns

    def test_threshold_classification_logic(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.10, 0.80, 0.30, 0.60]

        result = calculate_threshold_sensitivity(
            y_true,
            y_prob,
            thresholds=[0.50],
        )

        row = result.iloc[0]

        # 0.80 and 0.60 are >= 0.50
        assert row["predicted_churn_count"] == 2
        assert row["non_churn_count"] == 2
        assert row["predicted_churn_rate"] == pytest.approx(0.50)
        assert row["non_churn_rate"] == pytest.approx(0.50)

    def test_threshold_uses_greater_than_or_equal(self):
        y_true = [0, 1, 1]
        y_prob = [0.50, 0.50, 0.80]

        result = calculate_threshold_sensitivity(
            y_true,
            y_prob,
            thresholds=[0.50],
        )

        assert result.iloc[0]["predicted_churn_count"] == 3

    def test_metrics_are_calculated_correctly(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.10, 0.80, 0.30, 0.60]

        result = calculate_threshold_sensitivity(
            y_true,
            y_prob,
            thresholds=[0.50],
        )

        row = result.iloc[0]

        # Predictions: [0, 1, 0, 1]
        # Perfect classification
        assert row["accuracy"] == pytest.approx(1.0)
        assert row["precision"] == pytest.approx(1.0)
        assert row["recall"] == pytest.approx(1.0)
        assert row["f1"] == pytest.approx(1.0)

    def test_baseline_threshold_is_marked(self):
        y_true = [0, 1]
        y_prob = [0.20, 0.80]

        result = calculate_threshold_sensitivity(
            y_true,
            y_prob,
            thresholds=[0.40, 0.50, 0.60],
        )

        baseline = result[
            result["threshold"] == BASELINE_THRESHOLD
        ]

        assert len(baseline) == 1
        assert bool(baseline.iloc[0]["is_baseline"]) is True

    def test_non_baseline_threshold_is_not_marked(self):
        y_true = [0, 1]
        y_prob = [0.20, 0.80]

        result = calculate_threshold_sensitivity(
            y_true,
            y_prob,
            thresholds=[0.40],
        )

        assert bool(result.iloc[0]["is_baseline"]) is False

    def test_lower_threshold_flags_more_customers(self):
        y_true = [0, 1, 0, 1]
        y_prob = [0.10, 0.40, 0.60, 0.90]

        result = calculate_threshold_sensitivity(
            y_true,
            y_prob,
            thresholds=[0.30, 0.70],
        )

        low = result.iloc[0]
        high = result.iloc[1]

        assert (
            low["predicted_churn_count"]
            >= high["predicted_churn_count"]
        )

    def test_empty_y_true_is_rejected(self):
        with pytest.raises(ValueError):
            calculate_threshold_sensitivity(
                [],
                [],
                thresholds=[0.50],
            )

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(ValueError):
            calculate_threshold_sensitivity(
                [0, 1, 0],
                [0.10, 0.80],
                thresholds=[0.50],
            )

    def test_non_binary_y_true_is_rejected(self):
        with pytest.raises(ValueError):
            calculate_threshold_sensitivity(
                [0, 1, 2],
                [0.10, 0.80, 0.60],
                thresholds=[0.50],
            )

    def test_probability_outside_range_is_rejected(self):
        with pytest.raises(ValueError):
            calculate_threshold_sensitivity(
                [0, 1],
                [-0.10, 1.20],
                thresholds=[0.50],
            )

    def test_invalid_threshold_is_rejected(self):
        with pytest.raises(ValueError):
            calculate_threshold_sensitivity(
                [0, 1],
                [0.20, 0.80],
                thresholds=[1.20],
            )

    def test_empty_threshold_list_is_rejected(self):
        with pytest.raises(ValueError):
            calculate_threshold_sensitivity(
                [0, 1],
                [0.20, 0.80],
                thresholds=[],
            )

    def test_duplicate_thresholds_are_rejected(self):
        with pytest.raises(ValueError):
            calculate_threshold_sensitivity(
                [0, 1],
                [0.20, 0.80],
                thresholds=[0.50, 0.50],
            )

    def test_input_arrays_are_not_modified(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([0.10, 0.80, 0.30, 0.60])

        original_true = y_true.copy()
        original_prob = y_prob.copy()

        calculate_threshold_sensitivity(
            y_true,
            y_prob,
            thresholds=[0.30, 0.50, 0.70],
        )

        np.testing.assert_array_equal(
            y_true,
            original_true,
        )

        np.testing.assert_array_equal(
            y_prob,
            original_prob,
        )

    def test_probability_boundary_values_are_valid(self):
        y_true = [0, 1]
        y_prob = [0.0, 1.0]

        result = calculate_threshold_sensitivity(
            y_true,
            y_prob,
            thresholds=[0.0, 1.0],
        )

        assert len(result) == 2

    def test_metrics_remain_between_zero_and_one(self):
        y_true = [0, 1, 0, 1, 1, 0]
        y_prob = [0.10, 0.80, 0.40, 0.60, 0.90, 0.20]

        result = calculate_threshold_sensitivity(
            y_true,
            y_prob,
        )

        for column in [
            "accuracy",
            "precision",
            "recall",
            "f1",
            "predicted_churn_rate",
            "non_churn_rate",
        ]:
            assert result[column].between(0, 1).all()