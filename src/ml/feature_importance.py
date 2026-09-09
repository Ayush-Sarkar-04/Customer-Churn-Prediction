import joblib
import pandas as pd

from src.ml.prepare_dataset import FEATURE_COLUMNS


def get_feature_importance(model, feature_names):
    """
    Extract feature importance from a trained tree-based model.

    Parameters
    ----------
    model : trained model
        A trained model that provides feature_importances_.

    feature_names : list
        Names of the features used to train the model.

    Returns
    -------
    pandas.DataFrame
        Feature importance table sorted from highest
        to lowest importance.
    """

    if not hasattr(model, "feature_importances_"):
        raise ValueError(
            "Model does not provide feature_importances_."
        )

    if not feature_names:
        raise ValueError(
            "feature_names cannot be empty."
        )

    importances = model.feature_importances_

    if len(importances) != len(feature_names):
        raise ValueError(
            "Number of feature importances does not "
            "match number of feature names."
        )

    result = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })

    result = result.sort_values(
        "importance",
        ascending=False
    ).reset_index(drop=True)

    result["rank"] = result.index + 1

    return result[
        ["rank", "feature", "importance"]
    ]


def load_random_forest_feature_importance(
    model_path="models/random_forest.joblib"
):
    """
    Load the selected Random Forest model and return
    its feature importance table.

    The feature names are taken directly from the
    approved ML feature schema.
    """

    model = joblib.load(model_path)

    return get_feature_importance(
        model,
        FEATURE_COLUMNS
    )