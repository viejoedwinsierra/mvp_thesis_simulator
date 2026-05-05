from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover
    sm = None

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from premodeling_modeling.modeling_config import ModelingConfig
from premodeling_modeling.premodeling_tables import build_premodeling_datasets


def _require_statsmodels():
    if sm is None:
        raise ImportError("statsmodels no está instalado. Instala statsmodels para ejecutar modelos estadísticos.")


def _prepare_matrix_for_statsmodels(X: pd.DataFrame, max_features: int) -> pd.DataFrame:
    X_num = X.copy()

    for col in X_num.columns:
        if not pd.api.types.is_numeric_dtype(X_num[col]):
            X_num[col] = pd.to_numeric(X_num[col], errors="coerce")

    X_num = X_num.replace([np.inf, -np.inf], np.nan)
    X_num = X_num.dropna(axis=1, how="all")
    X_num = X_num.loc[:, X_num.nunique(dropna=True) > 1]
    X_num = X_num.fillna(X_num.median(numeric_only=True))

    # Evita modelos demasiado grandes por dummies.
    if X_num.shape[1] > max_features:
        variances = X_num.var(numeric_only=True).sort_values(ascending=False)
        selected = list(variances.head(max_features).index)
        X_num = X_num[selected]

    return sm.add_constant(X_num, has_constant="add")


def _train_test(X: pd.DataFrame, y: pd.Series, config: ModelingConfig):
    stratify = y if y.nunique(dropna=True) == 2 else None

    return train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=stratify,
    )


def fit_ols_log_linear(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelingConfig,
):
    _require_statsmodels()
    X_sm = _prepare_matrix_for_statsmodels(X_train, config.max_features_for_statsmodels)
    model = sm.OLS(y_train, X_sm)
    return model.fit()


def fit_glm_gamma_log(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelingConfig,
):
    _require_statsmodels()
    y_positive = pd.Series(y_train).copy()

    # Si y ya viene log1p transformada, sigue siendo no negativa.
    # Para Gamma se requiere estrictamente positiva.
    y_positive = y_positive.clip(lower=1e-6)

    X_sm = _prepare_matrix_for_statsmodels(X_train, config.max_features_for_statsmodels)
    model = sm.GLM(
        y_positive,
        X_sm,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    )

    return model.fit()


def fit_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ModelingConfig,
):
    _require_statsmodels()
    X_sm = _prepare_matrix_for_statsmodels(X_train, config.max_features_for_statsmodels)
    model = sm.Logit(y_train, X_sm)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return model.fit(disp=False, maxiter=200)
        except Exception:
            return model.fit_regularized(disp=False, maxiter=200)


def _predict_with_result(result, X: pd.DataFrame, config: ModelingConfig) -> np.ndarray:
    X_sm = _prepare_matrix_for_statsmodels(X, config.max_features_for_statsmodels)
    train_cols = list(result.model.exog_names)

    for col in train_cols:
        if col not in X_sm.columns:
            X_sm[col] = 0.0

    X_sm = X_sm[train_cols]

    return np.asarray(result.predict(X_sm))


def evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_classification(y_true: pd.Series, y_prob: np.ndarray) -> dict:
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = np.nan

    return metrics


def fit_statistical_models(
    df: pd.DataFrame,
    config: ModelingConfig | None = None,
) -> dict[str, dict]:
    """Entrena modelos estadísticos interpretables por target.

    Esta es la fase final de modelamiento, posterior a premodeling.
    """
    config = config or ModelingConfig()
    datasets = build_premodeling_datasets(df)

    results: dict[str, dict] = {}

    for target, payload in datasets.items():
        X = payload["X"]
        y = payload["y"]
        metadata = payload["metadata"]

        if target not in config.model_specs:
            continue

        X_train, X_test, y_train, y_test = _train_test(X, y, config)

        target_results = {
            "metadata": metadata,
            "models": {},
            "split": {
                "train_rows": int(len(X_train)),
                "test_rows": int(len(X_test)),
                "test_size": config.test_size,
                "random_state": config.random_state,
            },
        }

        for spec in config.model_specs[target]:
            try:
                if spec.model_type == "ols_log_linear":
                    result = fit_ols_log_linear(X_train, y_train, config)
                    pred = _predict_with_result(result, X_test, config)
                    metrics = evaluate_regression(y_test, pred)
                elif spec.model_type == "glm_gamma_log":
                    result = fit_glm_gamma_log(X_train, y_train, config)
                    pred = _predict_with_result(result, X_test, config)
                    metrics = evaluate_regression(y_test, pred)
                elif spec.model_type == "logistic_regression":
                    result = fit_logistic_regression(X_train, y_train, config)
                    pred = _predict_with_result(result, X_test, config)
                    pred = np.clip(pred, 0, 1)
                    metrics = evaluate_classification(y_test, pred)
                else:
                    continue

                target_results["models"][spec.model_type] = {
                    "spec": spec,
                    "result": result,
                    "metrics": metrics,
                    "predictions": pred,
                    "y_test": y_test,
                }
            except Exception as exc:
                target_results["models"][spec.model_type] = {
                    "spec": spec,
                    "error": str(exc),
                }

        results[target] = target_results

    return results
