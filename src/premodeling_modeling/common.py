from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import json
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
from sklearn.pipeline import Pipeline
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

DERIVED_COLUMNS_DEFAULT_EXCLUDE = {
    "size_range",
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

REGRESSION_TARGETS = [
    "storage_cost",
    "transfer_duration_sec",
    "queue_pressure",
]

CLASSIFICATION_TARGETS = [
    "has_error",
]


@dataclass
class TechniqueResult:
    technique: str
    target: str
    output_dir: Path
    metrics: pd.DataFrame
    coefficients: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
    diagnostics: pd.DataFrame | None = None
    model_path: Path | None = None
    report_path: Path | None = None
    plots: list[Path] | None = None


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: str | Path, data: dict[str, Any]) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_table(df: pd.DataFrame | None, path: str | Path) -> Path | None:
    if df is None:
        return None
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def safe_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def remove_invalid_columns(
    df: pd.DataFrame,
    target: str,
    extra_exclusions: set[str] | None = None,
) -> pd.DataFrame:
    extra_exclusions = extra_exclusions or set()

    excluded = (
        IDENTIFIER_COLUMNS
        | LEAKAGE_COLUMNS
        | DERIVED_COLUMNS_DEFAULT_EXCLUDE
        | extra_exclusions
        | {target}
    )

    return df.drop(columns=[c for c in excluded if c in df.columns], errors="ignore")


def infer_feature_columns(
    df: pd.DataFrame,
    target: str,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    extra_exclusions: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    work = remove_invalid_columns(df, target=target, extra_exclusions=extra_exclusions)

    if numeric_features is None:
        numeric_features = [
            col for col in DEFAULT_NUMERIC_FEATURES
            if col in work.columns and col != target
        ]

    if categorical_features is None:
        categorical_features = [
            col for col in DEFAULT_CATEGORICAL_FEATURES
            if col in work.columns and col != target
        ]

    numeric_features = [
        col for col in numeric_features
        if col in work.columns and pd.api.types.is_numeric_dtype(work[col])
    ]

    categorical_features = [
        col for col in categorical_features
        if col in work.columns
    ]

    return numeric_features, categorical_features


def build_model_frame(
    df: pd.DataFrame,
    target: str,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    log_target: bool = False,
    log_numeric_features: list[str] | None = None,
    extra_exclusions: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    if target not in df.columns:
        raise ValueError(f"No existe target en dataframe: {target}")

    numeric_features, categorical_features = infer_feature_columns(
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

    if log_numeric_features:
        for col in log_numeric_features:
            if col in work.columns:
                work = work[work[col] > 0].copy()
                work[col] = np.log1p(work[col])

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


def _make_one_hot_encoder() -> OneHotEncoder:
    """
    Compatibilidad con distintas versiones de scikit-learn.

    Algunas versiones usan `sparse_output`.
    Otras versiones antiguas usan `sparse`.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = False,
) -> ColumnTransformer:
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    categorical_transformer = _make_one_hot_encoder()

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.25,
    random_state: int = 42,
    stratify: bool = False,
):
    stratify_values = y if stratify and y.nunique(dropna=True) > 1 else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_values,
    )


def regression_metrics(y_true, y_pred, technique: str, target: str) -> pd.DataFrame:
    """
    RMSE compatible con versiones de scikit-learn donde
    mean_squared_error no soporta el argumento squared=False.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))

    return pd.DataFrame([{
        "technique": technique,
        "target": target,
        "n": len(y_true),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
    }])


def classification_metrics(y_true, y_pred, y_score, technique: str, target: str) -> pd.DataFrame:
    row = {
        "technique": technique,
        "target": target,
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


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return []


def build_prediction_table(
    y_true,
    y_pred,
    target: str,
    scenario_series: pd.Series | None = None,
) -> pd.DataFrame:
    out = pd.DataFrame({
        "target": target,
        "y_true": np.asarray(y_true),
        "y_pred": np.asarray(y_pred),
        "residual": np.asarray(y_true) - np.asarray(y_pred),
    })

    if scenario_series is not None and len(scenario_series) == len(out):
        out["scenario_name"] = scenario_series.values

    return out


def make_html_table(df: pd.DataFrame | None, title: str, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return f"<section class='card'><h2>{title}</h2><p>No disponible.</p></section>"

    table = df.head(max_rows).copy()

    for col in table.select_dtypes(include="number").columns:
        table[col] = table[col].round(5)

    return f"""
    <section class='card'>
        <h2>{title}</h2>
        <div class='table-container'>
            {table.to_html(index=False, border=0)}
        </div>
    </section>
    """


def build_basic_html(title: str, body: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 32px;
                color: #263238;
                background: #f7f9fb;
            }}
            h1 {{ color: #17324d; }}
            h2 {{
                color: #17324d;
                border-bottom: 2px solid #e5edf3;
                padding-bottom: 8px;
            }}
            .card {{
                background: white;
                padding: 20px;
                margin-bottom: 24px;
                border-radius: 12px;
                box-shadow: 0 1px 6px rgba(0,0,0,0.08);
            }}
            .note {{
                background: #eef6ff;
                border-left: 5px solid #2c7be5;
                padding: 14px;
                border-radius: 8px;
                margin-bottom: 24px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 13px;
            }}
            th {{
                background-color: #17324d;
                color: white;
                padding: 8px;
                text-align: left;
            }}
            td {{
                border: 1px solid #ddd;
                padding: 7px;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 18px;
            }}
            img {{
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 6px;
                background: white;
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        {body}
    </body>
    </html>
    """


def build_plot_grid(plot_paths: list[Path], relative_to: Path) -> str:
    if not plot_paths:
        return "<section class='card'><h2>Gráficos</h2><p>No se generaron gráficos.</p></section>"

    html = "<section class='card'><h2>Gráficos</h2><div class='grid'>"

    for path in plot_paths:
        try:
            rel = path.relative_to(relative_to).as_posix()
        except ValueError:
            rel = path.as_posix()

        title = path.stem.replace("_", " ").title()
        html += f"<div><h3>{title}</h3><img src='{rel}' alt='{title}'></div>"

    html += "</div></section>"
    return html
