import pandas as pd
import pytest

from sklearn.ensemble import RandomForestClassifier

from src.ml.feature_importance import get_feature_importance


FEATURE_NAMES = [
    "customer_tenure",
    "recency",
    "frequency",
    "monetary",
]


def create_trained_model():
    X = pd.DataFrame({
        "customer_tenure": [10, 20, 30, 40, 50, 60],
        "recency": [5, 10, 15, 20, 25, 30],
        "frequency": [1, 2, 3, 4, 5, 6],
        "monetary": [100, 200, 300, 400, 500, 600],
    })

    y = [0, 0, 0, 1, 1, 1]

    model = RandomForestClassifier(
        n_estimators=10,
        random_state=42,
    )

    model.fit(X, y)

    return model


def test_feature_importance_returns_dataframe():
    model = create_trained_model()

    result = get_feature_importance(
        model,
        FEATURE_NAMES,
    )

    assert isinstance(result, pd.DataFrame)


def test_feature_importance_contains_expected_columns():
    model = create_trained_model()

    result = get_feature_importance(
        model,
        FEATURE_NAMES,
    )

    assert list(result.columns) == [
        "rank",
        "feature",
        "importance",
    ]


def test_feature_importance_contains_all_features():
    model = create_trained_model()

    result = get_feature_importance(
        model,
        FEATURE_NAMES,
    )

    assert set(result["feature"]) == set(FEATURE_NAMES)
    assert len(result) == len(FEATURE_NAMES)


def test_feature_importance_is_sorted_descending():
    model = create_trained_model()

    result = get_feature_importance(
        model,
        FEATURE_NAMES,
    )

    importances = result["importance"].tolist()

    assert importances == sorted(
        importances,
        reverse=True,
    )


def test_feature_importance_ranks_are_sequential():
    model = create_trained_model()

    result = get_feature_importance(
        model,
        FEATURE_NAMES,
    )

    assert result["rank"].tolist() == [
        1,
        2,
        3,
        4,
    ]


def test_feature_importance_rejects_unsupported_model():
    model = "not a trained model"

    with pytest.raises(ValueError):
        get_feature_importance(
            model,
            FEATURE_NAMES,
        )


def test_feature_importance_rejects_empty_feature_names():
    model = create_trained_model()

    with pytest.raises(ValueError):
        get_feature_importance(
            model,
            [],
        )


def test_feature_importance_rejects_mismatched_features():
    model = create_trained_model()

    with pytest.raises(ValueError):
        get_feature_importance(
            model,
            ["feature_a", "feature_b"],
        )


def test_load_random_forest_feature_importance():
    from src.ml.feature_importance import (
        load_random_forest_feature_importance
    )

    result = load_random_forest_feature_importance()

    assert isinstance(result, pd.DataFrame)

    assert list(result.columns) == [
        "rank",
        "feature",
        "importance",
    ]

    assert len(result) == 15

    assert result["rank"].tolist() == list(range(1, 16))

    assert result["importance"].between(0, 1).all()

    assert result["importance"].tolist() == sorted(
        result["importance"].tolist(),
        reverse=True,
    )