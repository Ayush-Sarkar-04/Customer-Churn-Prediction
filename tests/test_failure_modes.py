import numpy as np
import pandas as pd
import pytest

from src.analytics.failure_modes import (
    FAILURE_MODE_LABELS,
    classify_failure_modes,
    calculate_failure_mode_analysis,
    calculate_failure_mode_summary,
    get_failure_observations,
)


# =========================================================
# TEST DATA
# =========================================================

Y_TRUE = np.array([1, 0, 0, 1])
Y_PRED = np.array([1, 0, 1, 0])
Y_PROB = np.array([0.90, 0.10, 0.70, 0.30])


# =========================================================
# FAILURE MODE CLASSIFICATION
# =========================================================


class TestClassifyFailureModes:

    def test_all_four_failure_modes_are_identified(self):
        result = classify_failure_modes(
            Y_TRUE,
            Y_PRED,
            Y_PROB,
        )

        assert result["failure_mode"].tolist() == [
            "TP",
            "TN",
            "FP",
            "FN",
        ]

    def test_failure_type_labels_are_correct(self):
        result = classify_failure_modes(
            Y_TRUE,
            Y_PRED,
        )

        assert result["failure_type"].tolist() == [
            "True Positive",
            "True Negative",
            "False Positive",
            "False Negative",
        ]

    def test_failure_mode_mapping_is_complete(self):
        assert FAILURE_MODE_LABELS == {
            "TP": "True Positive",
            "TN": "True Negative",
            "FP": "False Positive",
            "FN": "False Negative",
        }

    def test_predicted_probability_is_preserved(self):
        result = classify_failure_modes(
            Y_TRUE,
            Y_PRED,
            Y_PROB,
        )

        np.testing.assert_allclose(
            result["predicted_probability"].to_numpy(),
            Y_PROB,
        )

    def test_output_length_matches_input(self):
        result = classify_failure_modes(
            Y_TRUE,
            Y_PRED,
            Y_PROB,
        )

        assert len(result) == len(Y_TRUE)

    def test_without_probability_probability_column_is_absent(self):
        result = classify_failure_modes(
            Y_TRUE,
            Y_PRED,
        )

        assert "predicted_probability" not in result.columns

    def test_inputs_are_not_modified(self):
        y_true = Y_TRUE.copy()
        y_pred = Y_PRED.copy()
        y_prob = Y_PROB.copy()

        classify_failure_modes(
            y_true,
            y_pred,
            y_prob,
        )

        np.testing.assert_array_equal(y_true, Y_TRUE)
        np.testing.assert_array_equal(y_pred, Y_PRED)
        np.testing.assert_array_equal(y_prob, Y_PROB)


# =========================================================
# FAILURE MODE AGGREGATION
# =========================================================


class TestFailureModeAnalysis:

    def test_counts_are_correct(self):
        result = calculate_failure_mode_analysis(
            Y_TRUE,
            Y_PRED,
            Y_PROB,
        )

        counts = dict(
            zip(
                result["failure_mode"],
                result["count"],
            )
        )

        assert counts["TP"] == 1
        assert counts["TN"] == 1
        assert counts["FP"] == 1
        assert counts["FN"] == 1

    def test_rates_are_correct(self):
        result = calculate_failure_mode_analysis(
            Y_TRUE,
            Y_PRED,
        )

        rates = dict(
            zip(
                result["failure_mode"],
                result["rate"],
            )
        )

        assert rates["TP"] == pytest.approx(0.25)
        assert rates["TN"] == pytest.approx(0.25)
        assert rates["FP"] == pytest.approx(0.25)
        assert rates["FN"] == pytest.approx(0.25)

    def test_rates_sum_to_one(self):
        result = calculate_failure_mode_analysis(
            Y_TRUE,
            Y_PRED,
        )

        assert result["rate"].sum() == pytest.approx(1.0)

    def test_average_probability_is_correct(self):
        result = calculate_failure_mode_analysis(
            Y_TRUE,
            Y_PRED,
            Y_PROB,
        )

        averages = dict(
            zip(
                result["failure_mode"],
                result["average_predicted_probability"],
            )
        )

        assert averages["TP"] == pytest.approx(0.90)
        assert averages["TN"] == pytest.approx(0.10)
        assert averages["FP"] == pytest.approx(0.70)
        assert averages["FN"] == pytest.approx(0.30)

    def test_average_probability_is_nan_without_probability_input(self):
        result = calculate_failure_mode_analysis(
            Y_TRUE,
            Y_PRED,
        )

        assert result["average_predicted_probability"].isna().all()

    def test_expected_failure_mode_rows_are_present(self):
        result = calculate_failure_mode_analysis(
            Y_TRUE,
            Y_PRED,
        )

        assert result["failure_mode"].tolist() == [
            "TP",
            "TN",
            "FP",
            "FN",
        ]


# =========================================================
# FAILURE MODE SUMMARY
# =========================================================


class TestFailureModeSummary:

    def test_summary_counts_are_correct(self):
        result = calculate_failure_mode_summary(
            Y_TRUE,
            Y_PRED,
        )

        assert result["total_observations"] == 4
        assert result["true_positives"] == 1
        assert result["true_negatives"] == 1
        assert result["false_positives"] == 1
        assert result["false_negatives"] == 1

    def test_actual_and_predicted_counts_are_correct(self):
        result = calculate_failure_mode_summary(
            Y_TRUE,
            Y_PRED,
        )

        assert result["actual_churn"] == 2
        assert result["actual_non_churn"] == 2
        assert result["predicted_churn"] == 2

    def test_accuracy_is_correct(self):
        result = calculate_failure_mode_summary(
            Y_TRUE,
            Y_PRED,
        )

        assert result["accuracy"] == pytest.approx(0.50)

    def test_precision_is_correct(self):
        result = calculate_failure_mode_summary(
            Y_TRUE,
            Y_PRED,
        )

        assert result["precision"] == pytest.approx(0.50)

    def test_recall_is_correct(self):
        result = calculate_failure_mode_summary(
            Y_TRUE,
            Y_PRED,
        )

        assert result["recall"] == pytest.approx(0.50)

    def test_false_positive_rate_is_correct(self):
        result = calculate_failure_mode_summary(
            Y_TRUE,
            Y_PRED,
        )

        assert result["false_positive_rate"] == pytest.approx(0.50)

    def test_false_negative_rate_is_correct(self):
        result = calculate_failure_mode_summary(
            Y_TRUE,
            Y_PRED,
        )

        assert result["false_negative_rate"] == pytest.approx(0.50)

    def test_error_rate_is_correct(self):
        result = calculate_failure_mode_summary(
            Y_TRUE,
            Y_PRED,
        )

        assert result["error_rate"] == pytest.approx(0.50)

    def test_missed_churn_count_equals_false_negatives(self):
        result = calculate_failure_mode_summary(
            Y_TRUE,
            Y_PRED,
        )

        assert result["missed_churn_count"] == 1

    def test_unnecessary_targeting_count_equals_false_positives(self):
        result = calculate_failure_mode_summary(
            Y_TRUE,
            Y_PRED,
        )

        assert result["unnecessary_targeting_count"] == 1


# =========================================================
# FAILURE OBSERVATIONS
# =========================================================


class TestFailureObservations:

    def test_default_returns_only_fp_and_fn(self):
        result = get_failure_observations(
            Y_TRUE,
            Y_PRED,
            Y_PROB,
        )

        assert result["failure_mode"].tolist() == [
            "FP",
            "FN",
        ]

    def test_can_filter_to_false_positives(self):
        result = get_failure_observations(
            Y_TRUE,
            Y_PRED,
            Y_PROB,
            failure_modes=["FP"],
        )

        assert len(result) == 1
        assert result.iloc[0]["failure_mode"] == "FP"
        assert result.iloc[0]["actual"] == 0
        assert result.iloc[0]["predicted"] == 1

    def test_can_filter_to_false_negatives(self):
        result = get_failure_observations(
            Y_TRUE,
            Y_PRED,
            Y_PROB,
            failure_modes=["FN"],
        )

        assert len(result) == 1
        assert result.iloc[0]["failure_mode"] == "FN"
        assert result.iloc[0]["actual"] == 1
        assert result.iloc[0]["predicted"] == 0

    def test_can_request_multiple_failure_modes(self):
        result = get_failure_observations(
            Y_TRUE,
            Y_PRED,
            failure_modes=["TP", "FN"],
        )

        assert result["failure_mode"].tolist() == [
            "TP",
            "FN",
        ]

    def test_invalid_failure_mode_is_rejected(self):
        with pytest.raises(ValueError, match="Invalid failure modes"):
            get_failure_observations(
                Y_TRUE,
                Y_PRED,
                failure_modes=["INVALID"],
            )


# =========================================================
# INPUT VALIDATION
# =========================================================


class TestFailureModeValidation:

    def test_empty_y_true_is_rejected(self):
        with pytest.raises(ValueError, match="y_true cannot be empty"):
            classify_failure_modes(
                [],
                [],
            )

    def test_empty_y_pred_is_rejected(self):
        with pytest.raises(ValueError, match="y_pred cannot be empty"):
            classify_failure_modes(
                [0, 1],
                [],
            )

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(
            ValueError,
            match="same length",
        ):
            classify_failure_modes(
                [0, 1],
                [0],
            )

    def test_non_binary_actual_labels_are_rejected(self):
        with pytest.raises(
            ValueError,
            match="binary values",
        ):
            classify_failure_modes(
                [0, 1, 2],
                [0, 1, 0],
            )

    def test_non_binary_predictions_are_rejected(self):
        with pytest.raises(
            ValueError,
            match="binary values",
        ):
            classify_failure_modes(
                [0, 1, 0],
                [0, 1, 2],
            )

    def test_probability_length_mismatch_is_rejected(self):
        with pytest.raises(
            ValueError,
            match="same length",
        ):
            classify_failure_modes(
                [0, 1],
                [0, 1],
                [0.5],
            )

    def test_probability_below_zero_is_rejected(self):
        with pytest.raises(
            ValueError,
            match="between 0 and 1",
        ):
            classify_failure_modes(
                [0, 1],
                [0, 1],
                [-0.1, 0.5],
            )

    def test_probability_above_one_is_rejected(self):
        with pytest.raises(
            ValueError,
            match="between 0 and 1",
        ):
            classify_failure_modes(
                [0, 1],
                [0, 1],
                [0.5, 1.1],
            )

    def test_non_finite_probability_is_rejected(self):
        with pytest.raises(
            ValueError,
            match="finite values",
        ):
            classify_failure_modes(
                [0, 1],
                [0, 1],
                [np.nan, 0.5],
            )

    def test_infinite_probability_is_rejected(self):
        with pytest.raises(
            ValueError,
            match="finite values",
        ):
            classify_failure_modes(
                [0, 1],
                [0, 1],
                [np.inf, 0.5],
            )


# =========================================================
# EDGE CASES
# =========================================================


class TestFailureModeEdgeCases:

    def test_all_predictions_correct(self):
        y_true = [1, 1, 0, 0]
        y_pred = [1, 1, 0, 0]

        result = calculate_failure_mode_summary(
            y_true,
            y_pred,
        )

        assert result["true_positives"] == 2
        assert result["true_negatives"] == 2
        assert result["false_positives"] == 0
        assert result["false_negatives"] == 0
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["error_rate"] == pytest.approx(0.0)

    def test_all_predictions_are_churn(self):
        y_true = [1, 0, 1, 0]
        y_pred = [1, 1, 1, 1]

        result = calculate_failure_mode_summary(
            y_true,
            y_pred,
        )

        assert result["true_positives"] == 2
        assert result["false_positives"] == 2
        assert result["true_negatives"] == 0
        assert result["false_negatives"] == 0

    def test_all_predictions_are_non_churn(self):
        y_true = [1, 0, 1, 0]
        y_pred = [0, 0, 0, 0]

        result = calculate_failure_mode_summary(
            y_true,
            y_pred,
        )

        assert result["true_negatives"] == 2
        assert result["false_negatives"] == 2
        assert result["true_positives"] == 0
        assert result["false_positives"] == 0

    def test_summary_handles_no_actual_non_churn(self):
        y_true = [1, 1, 1]
        y_pred = [1, 0, 1]

        result = calculate_failure_mode_summary(
            y_true,
            y_pred,
        )

        assert result["false_positive_rate"] == pytest.approx(0.0)

    def test_summary_handles_no_actual_churn(self):
        y_true = [0, 0, 0]
        y_pred = [0, 1, 0]

        result = calculate_failure_mode_summary(
            y_true,
            y_pred,
        )

        assert result["false_negative_rate"] == pytest.approx(0.0)

    def test_summary_handles_no_predicted_churn(self):
        y_true = [0, 1, 0]
        y_pred = [0, 0, 0]

        result = calculate_failure_mode_summary(
            y_true,
            y_pred,
        )

        assert result["precision"] == pytest.approx(0.0)