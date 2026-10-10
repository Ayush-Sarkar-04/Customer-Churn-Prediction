import pandas as pd
import pytest

from src.ml.prepare_dataset import (
    DATASET_COLUMNS,
    FEATURE_COLUMNS,
    LEGACY_HEADER_WITH_MISSING_CAMPAIGNS_REDEEMED,
    _validate_csv_width,
    load_ml_dataset,
    prepare_ml_dataset,
)


def _sample_row():
    return [
        "C1", "31-03-2024", "154", "49", "2", "1219.7", "609.85", "32",
        "0", "0", "0", "0", "0", "0", "0", "0", "0", "1", "0",
    ]


def test_validate_csv_width_accepts_known_legacy_header(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text(
        ",".join(LEGACY_HEADER_WITH_MISSING_CAMPAIGNS_REDEEMED)
        + "\n" + ",".join(_sample_row()) + "\n",
        encoding="utf-8",
    )
    assert _validate_csv_width(path) == LEGACY_HEADER_WITH_MISSING_CAMPAIGNS_REDEEMED


def test_load_adds_missing_header_label_in_memory_and_preserves_bytes(tmp_path):
    path = tmp_path / "dataset.csv"
    original = (
        ",".join(LEGACY_HEADER_WITH_MISSING_CAMPAIGNS_REDEEMED)
        + "\n" + ",".join(_sample_row()) + "\n"
    ).encode()
    path.write_bytes(original)

    df = load_ml_dataset(path)

    assert list(df.columns) == DATASET_COLUMNS
    assert df.loc[0, "campaigns_redeemed"] == 0
    assert df.loc[0, "future_90d_purchases"] == 1
    assert df.loc[0, "churn"] == 0
    assert path.read_bytes() == original


def test_validate_rejects_wrong_row_width(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text(
        ",".join(LEGACY_HEADER_WITH_MISSING_CAMPAIGNS_REDEEMED)
        + "\nC1,2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fields; expected 19"):
        _validate_csv_width(path)


def test_prepare_uses_exact_14_feature_contract():
    values = {column: [1] for column in FEATURE_COLUMNS}
    values["churn"] = [1]
    X, y = prepare_ml_dataset(pd.DataFrame(values))
    assert list(X.columns) == FEATURE_COLUMNS
    assert X.shape == (1, 14)
    assert y.tolist() == [1]


def test_prepare_rejects_non_binary_churn():
    values = {column: [1] for column in FEATURE_COLUMNS}
    values["churn"] = [2]
    with pytest.raises(ValueError, match="binary"):
        prepare_ml_dataset(pd.DataFrame(values))
