import numpy as np
import pandas as pd
import pytest

from src.analytics.segment_model_audit import (
    calculate_segment_model_audit,
    calculate_segment_model_summary,
    rank_segments_by_metric,
)


# =========================================================
# TEST DATA
# =========================================================

Y_TRUE = np.array([
    1, 0, 0, 1,
    1, 0, 1, 0,
])

Y_PRED = np.array([
    1, 0, 1, 0,
    1, 1, 0, 0,
])

Y_PROB = np.array([
    0.90, 0.10, 0.70, 0.30,
    0.80, 0.60, 0.40, 0.20,
])

SEGMENTS = np.array([
    "Champions",
    "Champions",
    "Champions",
    "Champions",
    "At Risk",
    "At Risk",
    "At Risk",
    "At Risk",
])


# =========================================================
# SEGMENT AUDIT
# =========================================================


class TestCalculateSegmentModelAudit:

    def test_returns_one_row_per_segment(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        assert len(result) == 2
        assert result["segment"].tolist() == [
            "Champions",
            "At Risk",
        ]

    def test_sample_counts_are_correct(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        counts = dict(
            zip(
                result["segment"],
                result["sample_count"],
            )
        )

        assert counts["Champions"] == 4
        assert counts["At Risk"] == 4

    def test_actual_churn_counts_are_correct(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        counts = dict(
            zip(
                result["segment"],
                result["actual_churn_count"],
            )
        )

        assert counts["Champions"] == 2
        assert counts["At Risk"] == 2

    def test_predicted_churn_counts_are_correct(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        counts = dict(
            zip(
                result["segment"],
                result["predicted_churn_count"],
            )
        )

        assert counts["Champions"] == 2
        assert counts["At Risk"] == 2

    def test_confusion_counts_are_correct(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        champions = result[
            result["segment"] == "Champions"
        ].iloc[0]

        at_risk = result[
            result["segment"] == "At Risk"
        ].iloc[0]

        assert champions["true_positives"] == 1
        assert champions["true_negatives"] == 1
        assert champions["false_positives"] == 1
        assert champions["false_negatives"] == 1

        assert at_risk["true_positives"] == 1
        assert at_risk["true_negatives"] == 1
        assert at_risk["false_positives"] == 1
        assert at_risk["false_negatives"] == 1

    def test_actual_churn_rate_is_correct(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        assert (
            result.loc[
                result["segment"] == "Champions",
                "actual_churn_rate",
            ].iloc[0]
            == pytest.approx(0.50)
        )

        assert (
            result.loc[
                result["segment"] == "At Risk",
                "actual_churn_rate",
            ].iloc[0]
            == pytest.approx(0.50)
        )

    def test_predicted_churn_rate_is_correct(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        assert (
            result.loc[
                result["segment"] == "Champions",
                "predicted_churn_rate",
            ].iloc[0]
            == pytest.approx(0.50)
        )

        assert (
            result.loc[
                result["segment"] == "At Risk",
                "predicted_churn_rate",
            ].iloc[0]
            == pytest.approx(0.50)
        )

    def test_accuracy_is_correct(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        assert result["accuracy"].tolist() == pytest.approx([
            0.50,
            0.50,
        ])

    def test_precision_is_correct(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        assert result["precision"].tolist() == pytest.approx([
            0.50,
            0.50,
        ])

    def test_recall_is_correct(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        assert result["recall"].tolist() == pytest.approx([
            0.50,
            0.50,
        ])

    def test_f1_score_is_correct(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        assert result["f1_score"].tolist() == pytest.approx([
            0.50,
            0.50,
        ])

    def test_false_positive_rate_is_correct(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        assert result["false_positive_rate"].tolist() == pytest.approx([
            0.50,
            0.50,
        ])

    def test_false_negative_rate_is_correct(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        assert result["false_negative_rate"].tolist() == pytest.approx([
            0.50,
            0.50,
        ])


# =========================================================
# PROBABILITY-BASED METRICS
# =========================================================


class TestSegmentProbabilityMetrics:

    def test_average_predicted_probability_is_correct(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
            Y_PROB,
        )

        champions = result[
            result["segment"] == "Champions"
        ].iloc[0]

        at_risk = result[
            result["segment"] == "At Risk"
        ].iloc[0]

        assert champions[
            "average_predicted_probability"
        ] == pytest.approx(0.50)

        assert at_risk[
            "average_predicted_probability"
        ] == pytest.approx(0.50)

    def test_brier_score_is_correct(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
            Y_PROB,
        )

        champions = result[
            result["segment"] == "Champions"
        ].iloc[0]

        at_risk = result[
            result["segment"] == "At Risk"
        ].iloc[0]

        # Champions:
        # (0.90 - 1)^2 + (0.10 - 0)^2
        # + (0.70 - 0)^2 + (0.30 - 1)^2
        # = 0.01 + 0.01 + 0.49 + 0.49
        # = 1.00 / 4 = 0.25

        assert champions[
            "brier_score"
        ] == pytest.approx(0.25)

        # At Risk:
        # (0.80 - 1)^2 + (0.60 - 0)^2
        # + (0.40 - 1)^2 + (0.20 - 0)^2
        # = 0.04 + 0.36 + 0.36 + 0.04
        # = 0.80 / 4 = 0.20

        assert at_risk[
            "brier_score"
        ] == pytest.approx(0.20)

    def test_probability_columns_are_nan_without_probabilities(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        assert result[
            "average_predicted_probability"
        ].isna().all()

        assert result[
            "brier_score"
        ].isna().all()

    def test_probability_values_remain_finite(self):
        result = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
            Y_PROB,
        )

        assert np.isfinite(
            result["average_predicted_probability"]
        ).all()

        assert np.isfinite(
            result["brier_score"]
        ).all()


# =========================================================
# SUMMARY
# =========================================================


class TestSegmentModelSummary:

    def test_summary_counts_segments(self):
        audit = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        result = calculate_segment_model_summary(audit)

        assert result["segment_count"] == 2

    def test_summary_counts_total_observations(self):
        audit = calculate_segment_model_audit(
            Y_TRUE,
            Y_PRED,
            SEGMENTS,
        )

        result = calculate_segment_model_summary(audit)

        assert result["total_observations"] == 8

    def test_best_segment_is_identified(self):
        audit = pd.DataFrame({
            "segment": [
                "A",
                "B",
                "C",
            ],
            "sample_count": [
                100,
                100,
                100,
            ],
            "f1_score": [
                0.50,
                0.80,
                0.60,
            ],
        })

        result = calculate_segment_model_summary(audit)

        assert result["best_segment"] == "B"
        assert result["best_f1_score"] == pytest.approx(0.80)

    def test_worst_segment_is_identified(self):
        audit = pd.DataFrame({
            "segment": [
                "A",
                "B",
                "C",
            ],
            "sample_count": [
                100,
                100,
                100,
            ],
            "f1_score": [
                0.50,
                0.80,
                0.60,
            ],
        })

        result = calculate_segment_model_summary(audit)

        assert result["worst_segment"] == "A"
        assert result["worst_f1_score"] == pytest.approx(0.50)

    def test_f1_score_range_is_correct(self):
        audit = pd.DataFrame({
            "segment": [
                "A",
                "B",
                "C",
            ],
            "sample_count": [
                100,
                100,
                100,
            ],
            "f1_score": [
                0.50,
                0.80,
                0.60,
            ],
        })

        result = calculate_segment_model_summary(audit)

        assert result["f1_score_range"] == pytest.approx(0.30)

    def test_empty_audit_returns_empty_summary(self):
        audit = pd.DataFrame()

        result = calculate_segment_model_summary(audit)

        assert result["segment_count"] == 0
        assert result["total_observations"] == 0
        assert result["best_segment"] is None
        assert result["worst_segment"] is None
        assert result["f1_score_range"] == pytest.approx(0.0)

    def test_summary_requires_dataframe(self):
        with pytest.raises(
            TypeError,
            match="pandas DataFrame",
        ):
            calculate_segment_model_summary([])

    def test_summary_rejects_missing_columns(self):
        audit = pd.DataFrame({
            "segment": ["A"],
            "sample_count": [10],
        })

        with pytest.raises(
            ValueError,
            match="missing required columns",
        ):
            calculate_segment_model_summary(audit)


# =========================================================
# SEGMENT RANKING
# =========================================================


class TestSegmentRanking:

    def test_ranks_segments_by_f1_descending(self):
        audit = pd.DataFrame({
            "segment": [
                "A",
                "B",
                "C",
            ],
            "f1_score": [
                0.50,
                0.80,
                0.60,
            ],
        })

        result = rank_segments_by_metric(
            audit,
            metric="f1_score",
        )

        assert result["segment"].tolist() == [
            "B",
            "C",
            "A",
        ]

    def test_ranks_segments_ascending_when_requested(self):
        audit = pd.DataFrame({
            "segment": [
                "A",
                "B",
                "C",
            ],
            "f1_score": [
                0.50,
                0.80,
                0.60,
            ],
        })

        result = rank_segments_by_metric(
            audit,
            metric="f1_score",
            ascending=True,
        )

        assert result["segment"].tolist() == [
            "A",
            "C",
            "B",
        ]

    def test_metric_must_exist(self):
        audit = pd.DataFrame({
            "segment": ["A"],
            "f1_score": [0.50],
        })

        with pytest.raises(
            ValueError,
            match="does not exist",
        ):
            rank_segments_by_metric(
                audit,
                metric="roc_auc",
            )

    def test_ranking_requires_dataframe(self):
        with pytest.raises(
            TypeError,
            match="pandas DataFrame",
        ):
            rank_segments_by_metric(
                [],
                metric="f1_score",
            )

    def test_ties_are_deterministic(self):
        audit = pd.DataFrame({
            "segment": [
                "B",
                "A",
                "C",
            ],
            "f1_score": [
                0.80,
                0.80,
                0.60,
            ],
        })

        result = rank_segments_by_metric(
            audit,
            metric="f1_score",
        )

        assert result["segment"].tolist() == [
            "A",
            "B",
            "C",
        ]


# =========================================================
# INPUT VALIDATION
# =========================================================


class TestSegmentAuditValidation:

    def test_empty_y_true_is_rejected(self):
        with pytest.raises(
            ValueError,
            match="y_true cannot be empty",
        ):
            calculate_segment_model_audit(
                [],
                [],
                [],
            )

    def test_prediction_length_mismatch_is_rejected(self):
        with pytest.raises(
            ValueError,
            match="same length",
        ):
            calculate_segment_model_audit(
                [0, 1],
                [0],
                ["A", "B"],
            )

    def test_segment_length_mismatch_is_rejected(self):
        with pytest.raises(
            ValueError,
            match="same length",
        ):
            calculate_segment_model_audit(
                [0, 1],
                [0, 1],
                ["A"],
            )

    def test_non_binary_actual_labels_are_rejected(self):
        with pytest.raises(
            ValueError,
            match="binary values",
        ):
            calculate_segment_model_audit(
                [0, 1, 2],
                [0, 1, 0],
                ["A", "A", "B"],
            )

    def test_non_binary_predictions_are_rejected(self):
        with pytest.raises(
            ValueError,
            match="binary values",
        ):
            calculate_segment_model_audit(
                [0, 1, 0],
                [0, 1, 2],
                ["A", "A", "B"],
            )

    def test_missing_segment_is_rejected(self):
        with pytest.raises(
            ValueError,
            match="missing values",
        ):
            calculate_segment_model_audit(
                [0, 1],
                [0, 1],
                ["A", np.nan],
            )

    def test_probability_length_mismatch_is_rejected(self):
        with pytest.raises(
            ValueError,
            match="same length",
        ):
            calculate_segment_model_audit(
                [0, 1],
                [0, 1],
                ["A", "B"],
                [0.5],
            )

    def test_probability_below_zero_is_rejected(self):
        with pytest.raises(
            ValueError,
            match="between 0 and 1",
        ):
            calculate_segment_model_audit(
                [0, 1],
                [0, 1],
                ["A", "B"],
                [-0.1, 0.5],
            )

    def test_probability_above_one_is_rejected(self):
        with pytest.raises(
            ValueError,
            match="between 0 and 1",
        ):
            calculate_segment_model_audit(
                [0, 1],
                [0, 1],
                ["A", "B"],
                [0.5, 1.1],
            )

    def test_non_finite_probability_is_rejected(self):
        with pytest.raises(
            ValueError,
            match="finite values",
        ):
            calculate_segment_model_audit(
                [0, 1],
                [0, 1],
                ["A", "B"],
                [np.nan, 0.5],
            )

    def test_infinite_probability_is_rejected(self):
        with pytest.raises(
            ValueError,
            match="finite values",
        ):
            calculate_segment_model_audit(
                [0, 1],
                [0, 1],
                ["A", "B"],
                [np.inf, 0.5],
            )


# =========================================================
# EDGE CASES
# =========================================================


class TestSegmentAuditEdgeCases:

    def test_single_segment_is_supported(self):
        result = calculate_segment_model_audit(
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            ["A", "A", "A", "A"],
        )

        assert len(result) == 1
        assert result.iloc[0]["segment"] == "A"
        assert result.iloc[0]["sample_count"] == 4

    def test_segment_with_no_predicted_churn_has_zero_precision(self):
        result = calculate_segment_model_audit(
            [0, 1, 0],
            [0, 0, 0],
            ["A", "A", "A"],
        )

        assert result.iloc[0]["precision"] == pytest.approx(0.0)

    def test_segment_with_no_actual_churn_has_zero_fnr(self):
        result = calculate_segment_model_audit(
            [0, 0, 0],
            [0, 1, 0],
            ["A", "A", "A"],
        )

        assert result.iloc[0][
            "false_negative_rate"
        ] == pytest.approx(0.0)

    def test_segment_with_no_actual_non_churn_has_zero_fpr(self):
        result = calculate_segment_model_audit(
            [1, 1, 1],
            [1, 0, 1],
            ["A", "A", "A"],
        )

        assert result.iloc[0][
            "false_positive_rate"
        ] == pytest.approx(0.0)

    def test_inputs_are_not_modified(self):
        y_true = Y_TRUE.copy()
        y_pred = Y_PRED.copy()
        segments = SEGMENTS.copy()
        y_prob = Y_PROB.copy()

        calculate_segment_model_audit(
            y_true,
            y_pred,
            segments,
            y_prob,
        )

        np.testing.assert_array_equal(
            y_true,
            Y_TRUE,
        )

        np.testing.assert_array_equal(
            y_pred,
            Y_PRED,
        )

        np.testing.assert_array_equal(
            segments,
            SEGMENTS,
        )

        np.testing.assert_array_equal(
            y_prob,
            Y_PROB,
        )