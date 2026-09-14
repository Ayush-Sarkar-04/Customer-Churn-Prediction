from pathlib import Path

import pandas as pd
import pytest

from src.data.analytics_loader import (
    load_analytics_data,
    validate_campaign_outcomes,
    validate_synthetic_experiment,
)


ANALYTICS_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "analytics"
)


def test_load_analytics_data():
    campaign_outcomes, experiment = load_analytics_data(
        ANALYTICS_PATH
    )

    assert len(campaign_outcomes) == 8000
    assert len(experiment) == 6000


def test_campaign_outcomes_schema():
    campaign_outcomes = pd.read_csv(
        ANALYTICS_PATH / "campaign_outcomes.csv"
    )

    validate_campaign_outcomes(
        campaign_outcomes
    )


def test_synthetic_experiment_schema():
    experiment = pd.read_csv(
        ANALYTICS_PATH / "synthetic_campaign_experiment.csv"
    )

    validate_synthetic_experiment(
        experiment
    )


def test_campaign_outcomes_have_unique_campaign_ids():
    campaign_outcomes = pd.read_csv(
        ANALYTICS_PATH / "campaign_outcomes.csv"
    )

    assert campaign_outcomes[
        "campaign_id"
    ].is_unique


def test_experiment_has_unique_customer_campaign_pairs():
    experiment = pd.read_csv(
        ANALYTICS_PATH
        / "synthetic_campaign_experiment.csv"
    )

    assert not experiment[
        ["customer_id", "campaign_type"]
    ].duplicated().any()


def test_experiment_has_treatment_and_control():
    experiment = pd.read_csv(
        ANALYTICS_PATH
        / "synthetic_campaign_experiment.csv"
    )

    groups = set(
        experiment["treatment_group"].unique()
    )

    assert groups == {
        "treatment",
        "control",
    }


def test_control_rows_have_zero_cost():
    experiment = pd.read_csv(
        ANALYTICS_PATH
        / "synthetic_campaign_experiment.csv"
    )

    control = experiment[
        experiment["treatment_group"] == "control"
    ]

    assert control[
        "synthetic_campaign_cost"
    ].eq(0).all()


def test_purchase_windows_are_nested():
    campaign_outcomes = pd.read_csv(
        ANALYTICS_PATH / "campaign_outcomes.csv"
    )

    assert (
        campaign_outcomes["purchase_7d"]
        <= campaign_outcomes["purchase_14d"]
    ).all()

    assert (
        campaign_outcomes["purchase_14d"]
        <= campaign_outcomes["purchase_30d"]
    ).all()


def test_revenue_windows_are_nested():
    campaign_outcomes = pd.read_csv(
        ANALYTICS_PATH / "campaign_outcomes.csv"
    )

    assert (
        campaign_outcomes["revenue_7d"]
        <= campaign_outcomes["revenue_14d"]
    ).all()

    assert (
        campaign_outcomes["revenue_14d"]
        <= campaign_outcomes["revenue_30d"]
    ).all()


def test_missing_campaign_outcomes_file():
    with pytest.raises(FileNotFoundError):
        load_analytics_data(
            ANALYTICS_PATH / "missing"
        )


def test_missing_experiment_file(tmp_path):
    campaign_outcomes = pd.read_csv(
        ANALYTICS_PATH / "campaign_outcomes.csv"
    )

    campaign_outcomes.to_csv(
        tmp_path / "campaign_outcomes.csv",
        index=False,
    )

    with pytest.raises(FileNotFoundError):
        load_analytics_data(tmp_path)