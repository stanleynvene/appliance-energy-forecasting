"""Feature-based machine-learning models.

XGBoost is used when it is installed, with scikit-learn's
``HistGradientBoostingRegressor`` as a drop-in alternative so that the pipeline
still runs in a minimal environment.

The operational model is a *direct* multi-horizon regressor: one model is
trained on (forecast origin, horizon) pairs, and every predictor is a quantity
that has already been measured at the origin or is a calendar feature of the
target time. Nothing that would be unknown 24 hours in advance enters the
operational feature set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..features import direct_feature_columns

try:  # pragma: no cover - depends on the environment
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:  # pragma: no cover
    HAS_XGBOOST = False

from sklearn.ensemble import HistGradientBoostingRegressor


def make_regressor(random_state: int = config.RANDOM_STATE):
    """Return the gradient boosting regressor used throughout the project."""
    if HAS_XGBOOST:
        return XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.8,
            min_child_weight=5,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=random_state,
            n_jobs=4,
        )

    return HistGradientBoostingRegressor(
        max_iter=600,
        learning_rate=0.03,
        max_leaf_nodes=31,
        random_state=random_state,
    )


def backend_name() -> str:
    return "xgboost" if HAS_XGBOOST else "sklearn.HistGradientBoostingRegressor"


# ------------------------------------------------------------------
# Direct multi-horizon model
# ------------------------------------------------------------------

def split_direct_table(table: pd.DataFrame, first_test_time, last_train_time):
    """Split a direct design matrix into training and evaluation rows.

    A row belongs to the training set only if *both* its origin and its target
    time fall inside the training period. Evaluation rows are those whose
    target time falls in the test period and whose origin is the corresponding
    daily forecast origin.
    """
    train_mask = table["target_time"] <= last_train_time
    test_mask = table["target_time"] >= first_test_time

    return table.loc[train_mask].copy(), table.loc[test_mask].copy()


def select_forecast_origins(table: pd.DataFrame, origins) -> pd.DataFrame:
    """Keep only the rows issued from the given rolling forecast origins."""
    origins = pd.DatetimeIndex(origins)
    return table.loc[table["origin"].isin(origins)].copy()


def fit_feature_model(X_train, y_train, random_state: int = config.RANDOM_STATE):
    """Fit the gradient boosting regressor."""
    model = make_regressor(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def forecast_feature_model(model, X_test, index, name: str = "feature_model") -> pd.Series:
    """Predict and return a named series on the requested index."""
    prediction = np.asarray(model.predict(X_test), dtype=float)
    return pd.Series(prediction, index=index, name=name)


def run_direct_model(
    table: pd.DataFrame,
    origins,
    last_train_time,
    first_test_time,
    name: str = "feature_model",
    random_state: int = config.RANDOM_STATE,
):
    """Train on the training rows and forecast the rolling-origin test blocks."""
    train_rows, test_rows = split_direct_table(table, first_test_time, last_train_time)
    test_rows = select_forecast_origins(test_rows, origins).sort_values("target_time")

    feature_cols = direct_feature_columns(table)

    model = fit_feature_model(
        train_rows[feature_cols], train_rows["y"], random_state=random_state
    )

    prediction = forecast_feature_model(
        model,
        test_rows[feature_cols],
        index=pd.DatetimeIndex(test_rows["target_time"]),
        name=name,
    )
    prediction.index.name = "date"

    return model, prediction, feature_cols, train_rows


def feature_importance(model, feature_names) -> pd.Series:
    """Feature importances as a sorted series, whichever backend is in use."""
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
        return pd.Series(values, index=feature_names).sort_values(ascending=False)

    return pd.Series(dtype=float)


def permutation_feature_importance(model, X, y, n_repeats: int = 5,
                                   random_state: int = config.RANDOM_STATE) -> pd.Series:
    """Permutation importance on held-out rows, used when native gains are absent."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state,
        scoring="neg_mean_absolute_error",
    )

    return pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)


# ------------------------------------------------------------------
# One-step model (leakage demonstration)
# ------------------------------------------------------------------

def run_one_step_model(
    one_step_table: pd.DataFrame,
    test_index,
    target: str = config.TARGET,
    name: str = "feature_model_one_step",
    random_state: int = config.RANDOM_STATE,
):
    """Fit and score the one-step-ahead design from the demo pipeline.

    Scoring this model on the 14-day test set answers the question "what would
    the accuracy look like if the previous hour were always known?", which is
    not the forecasting task specified in the brief.
    """
    train_rows = one_step_table.loc[one_step_table.index < test_index[0]]
    test_rows = one_step_table.loc[one_step_table.index.isin(test_index)]

    feature_cols = [c for c in one_step_table.columns if c != target]

    model = fit_feature_model(
        train_rows[feature_cols], train_rows[target], random_state=random_state
    )

    prediction = forecast_feature_model(
        model, test_rows[feature_cols], index=test_rows.index, name=name
    )

    return model, prediction, feature_cols
