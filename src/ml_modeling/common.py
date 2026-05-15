from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import os

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


IDENTIFIER_COLUMNS = {
    "file_id",
    "run_id",
    "sequence",
    "content_hash",
    "hash_head",
    "hash_tail",
    "source_file",
    "source_path",
}

LEAKAGE_COLUMNS = {
    "severity",
    "is_duplicate",
    "error_duplicado",
    "error_orphan",
    "error_null",
    "error_blob_timeout",
}

DEFAULT_NUMERIC_FEATURES = [
    "size_mb",
    "days_stored",
    "days_since_last_access",
    "movement_storage",
    "transfer_duration_sec",
    "transfer_speed_mbps",
    "created_hour",
    "hourly_arrival_count",
    "hourly_capacity",
    "queue_pressure",
    "congestion_factor",
]

DEFAULT_CATEGORICAL_FEATURES = [
    "file_type",
    "storage_tier",
    "day_of_week",
    "time_slot",
    "case_group",
    "scenario_name",
]


@dataclass
class MLModelResult:
    family: str
    technique: str
    target: str
    task_type: str
    output_dir: Path
    metrics: pd.DataFrame
    feature_importance: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
    model_path: Path | None = None
    report_path: Path | None = None
    plots: list[Path] | None = None


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_table(df: pd.DataFrame | None, path: str | Path) -> Path | None:
    if df is None:
        return None
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_json(path: str | Path, data: dict) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def infer_features(
    df: pd.DataFrame,
    target: str,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    extra_exclusions: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    extra_exclusions = extra_exclusions or set()
    excluded = IDENTIFIER_COLUMNS | LEAKAGE_COLUMNS | extra_exclusions | {target}

    if numeric_features is None:
        numeric_features = [
            c for c in DEFAULT_NUMERIC_FEATURES
            if c in df.columns and c not in excluded and pd.api.types.is_numeric_dtype(df[c])
        ]

    if categorical_features is None:
        categorical_features = [
            c for c in DEFAULT_CATEGORICAL_FEATURES
            if c in df.columns and c not in excluded
        ]

    numeric_features = [c for c in numeric_features if c in df.columns and c not in excluded]
    categorical_features = [c for c in categorical_features if c in df.columns and c not in excluded]

    return numeric_features, categorical_features


def build_model_frame(
    df: pd.DataFrame,
    target: str,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    extra_exclusions: set[str] | None = None,
    log_target: bool = False,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    if target not in df.columns:
        raise ValueError(f"No existe target en dataframe: {target}")

    numeric_features, categorical_features = infer_features(
        df=df,
        target=target,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        extra_exclusions=extra_exclusions,
    )

    cols = numeric_features + categorical_features + [target]
    work = df[cols].copy()

    for col in numeric_features + [target]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce")

    if log_target:
        work = work[work[target] > 0].copy()
        work[target] = np.log1p(work[target])

    for col in categorical_features:
        work[col] = work[col].fillna("UNKNOWN").astype(str)

    work = work.replace([np.inf, -np.inf], np.nan).dropna()

    if work.empty:
        raise ValueError(f"No quedan datos validos para target={target}")

    X = work[numeric_features + categorical_features].copy()
    y = work[target].copy()

    return X, y, numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", _one_hot_encoder(), categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_data(X, y, task_type: str, test_size: float = 0.25, random_state: int = 42):
    stratify = y if task_type == "classification" and y.nunique(dropna=True) > 1 else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify)


def rmse_value(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def regression_metrics(y_true, y_pred, technique: str, target: str) -> pd.DataFrame:
    return pd.DataFrame([{
        "technique": technique,
        "target": target,
        "task_type": "regression",
        "n": len(y_true),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": rmse_value(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }])


def classification_metrics(y_true, y_pred, y_score, technique: str, target: str) -> pd.DataFrame:
    row = {
        "technique": technique,
        "target": target,
        "task_type": "classification",
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    try:
        row["roc_auc"] = roc_auc_score(y_true, y_score)
    except Exception:
        row["roc_auc"] = np.nan
    return pd.DataFrame([row])


def prediction_table(y_true, y_pred, target: str, y_score=None) -> pd.DataFrame:
    df = pd.DataFrame({
        "target": target,
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
    })

    if y_score is not None:
        df["score"] = np.asarray(y_score)

    if np.issubdtype(np.asarray(y_true).dtype, np.number) and np.issubdtype(np.asarray(y_pred).dtype, np.number):
        df["residual"] = np.asarray(y_true) - np.asarray(y_pred)

    return df


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def extract_feature_importance(model, preprocessor, technique: str) -> pd.DataFrame:
    feature_names = get_feature_names(preprocessor)
    estimator = model.named_steps.get("model")

    if estimator is None:
        return pd.DataFrame(columns=["feature", "importance", "technique"])

    values = None
    column = "importance"

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        column = "importance"

    elif hasattr(estimator, "coef_"):
        coef = estimator.coef_
        if getattr(coef, "ndim", 1) > 1:
            coef = coef[0]
        values = coef
        column = "coefficient"

    if values is None or not feature_names:
        return pd.DataFrame(columns=["feature", column, "abs_value", "technique"])

    out = pd.DataFrame({
        "feature": feature_names[:len(values)],
        column: values[:len(feature_names)],
        "technique": technique,
    })

    out["abs_value"] = out[column].abs()
    return out.sort_values("abs_value", ascending=False).reset_index(drop=True)


def persist_model(model, output_dir: Path, technique: str, target: str) -> Path:
    model_dir = ensure_dir(output_dir / "models")
    path = model_dir / f"{technique}_{target}.joblib"
    joblib.dump(model, path)
    return path


def safe_rel(path: Path, start: Path) -> str:
    return os.path.relpath(path, start=start).replace("\\", "/")


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    if isinstance(value, (int, float)):
        if math.isfinite(value):
            return f"{value:,.{digits}f}"
        return "N/A"
    return str(value)
