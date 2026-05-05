from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from premodeling_modeling.premodeling_config import PremodelingConfig, TARGET_COLUMNS


def _existing(columns: list[str], df: pd.DataFrame) -> list[str]:
    return [col for col in columns if col in df.columns]


def _safe_log1p(series: pd.Series) -> pd.Series:
    data = pd.to_numeric(series, errors="coerce")
    data = data.clip(lower=0)
    return np.log1p(data)


def build_variable_plan_table(
    df: pd.DataFrame,
    config: PremodelingConfig | None = None,
) -> pd.DataFrame:
    """Tabla de plan de variables por target.

    Esta tabla NO decide modelos finales; documenta el dataset que se prepara
    para la fase posterior.
    """
    config = config or PremodelingConfig()
    rows = []

    for target, spec in config.target_specs.items():
        if target not in df.columns:
            rows.append({
                "target": target,
                "variable": target,
                "role": "target_missing",
                "included": "NO",
                "reason": "target no existe en el dataframe",
            })
            continue

        rows.append({
            "target": target,
            "variable": target,
            "role": "target",
            "included": "SI",
            "reason": spec.notes,
        })

        for variable in spec.candidate_features:
            included = variable in df.columns

            if variable in config.leakage_columns:
                role = "excluded_leakage"
                reason = "variable excluida por fuga de información"
                included = False
            elif variable in config.identifier_columns:
                role = "excluded_identifier"
                reason = "identificador técnico sin valor generalizable"
                included = False
            elif variable in config.derived_columns:
                role = "candidate_derived"
                reason = "variable derivada; revisar redundancia en documentación"
            elif variable in config.categorical_columns:
                role = "candidate_categorical"
                reason = "variable categórica codificable"
            else:
                role = "candidate_numeric"
                reason = "variable numérica candidata"

            rows.append({
                "target": target,
                "variable": variable,
                "role": role,
                "included": "SI" if included else "NO",
                "reason": reason if variable in df.columns else "variable no existe en el dataframe",
            })

    return pd.DataFrame(rows)


def prepare_target_dataset(
    df: pd.DataFrame,
    target: str,
    config: PremodelingConfig | None = None,
    encode_categoricals: bool = True,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Construye X, y y metadatos para un target.

    No hace split ni entrena modelos.
    """
    config = config or PremodelingConfig()

    if target not in config.target_specs:
        raise ValueError(f"Target no configurado: {target}")

    spec = config.target_specs[target]

    if target not in df.columns:
        raise ValueError(f"Target no existe en dataframe: {target}")

    candidate_features = _existing(spec.candidate_features, df)

    excluded = set(config.identifier_columns) | set(config.leakage_columns) | {target}
    candidate_features = [col for col in candidate_features if col not in excluded]

    data = df[candidate_features + [target]].copy()
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna(axis=0)

    y = data[target].copy()
    X = data[candidate_features].copy()

    transformed_columns = []

    for col in list(X.columns):
        if col in config.log_transform_columns and pd.api.types.is_numeric_dtype(X[col]):
            X[f"log1p_{col}"] = _safe_log1p(X[col])
            transformed_columns.append(f"log1p_{col}")

    if spec.log_transform_target:
        y = _safe_log1p(y)
        target_output_name = f"log1p_{target}"
    else:
        target_output_name = target

    categorical_used = []

    if encode_categoricals:
        categorical_cols = [
            col for col in X.columns
            if col in config.categorical_columns and col in X.columns
        ]

        categorical_used = categorical_cols.copy()

        if categorical_cols:
            X = pd.get_dummies(
                X,
                columns=categorical_cols,
                drop_first=True,
                dtype=float,
            )

    bool_cols = X.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(float)

    X = X.apply(pd.to_numeric, errors="ignore")

    metadata = {
        "target": target,
        "target_output_name": target_output_name,
        "problem_type": spec.problem_type,
        "rows": int(len(X)),
        "features_original": candidate_features,
        "features_final": list(X.columns),
        "categorical_encoded": categorical_used,
        "transformed_columns": transformed_columns,
        "log_transform_target": bool(spec.log_transform_target),
        "excluded_columns": sorted(excluded & set(df.columns)),
    }

    return X, y, metadata


def build_premodeling_datasets(
    df: pd.DataFrame,
    config: PremodelingConfig | None = None,
) -> dict[str, dict]:
    config = config or PremodelingConfig()
    datasets = {}

    for target in config.target_specs:
        if target not in df.columns:
            continue

        X, y, metadata = prepare_target_dataset(df, target, config=config)
        datasets[target] = {
            "X": X,
            "y": y,
            "metadata": metadata,
        }

    return datasets


def build_dataset_summary_table(datasets: dict[str, dict]) -> pd.DataFrame:
    rows = []

    for target, payload in datasets.items():
        X = payload["X"]
        y = payload["y"]
        metadata = payload["metadata"]

        rows.append({
            "target": target,
            "problem_type": metadata.get("problem_type"),
            "n_rows": len(X),
            "n_features": X.shape[1],
            "target_output_name": metadata.get("target_output_name"),
            "log_transform_target": metadata.get("log_transform_target"),
            "encoded_categoricals": ", ".join(metadata.get("categorical_encoded", [])),
            "transformed_columns": ", ".join(metadata.get("transformed_columns", [])),
        })

    return pd.DataFrame(rows)


def build_feature_summary_table(datasets: dict[str, dict]) -> pd.DataFrame:
    rows = []

    for target, payload in datasets.items():
        X = payload["X"]

        for col in X.columns:
            series = X[col]
            rows.append({
                "target": target,
                "feature": col,
                "dtype": str(series.dtype),
                "null_count": int(series.isna().sum()),
                "unique_count": int(series.nunique(dropna=True)),
                "mean": series.mean() if pd.api.types.is_numeric_dtype(series) else np.nan,
                "std": series.std() if pd.api.types.is_numeric_dtype(series) else np.nan,
            })

    return pd.DataFrame(rows)


def export_premodeling_evidence(
    datasets: dict[str, dict],
    output_dir: str | Path = "output/premodeling",
) -> dict[str, Path]:
    """Exporta datasets y metadata de premodeling.

    Se deja como evidencia auditable para la siguiente fase.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    for target, payload in datasets.items():
        X = payload["X"]
        y = payload["y"]
        metadata = payload["metadata"]

        target_dir = output_dir / target
        target_dir.mkdir(parents=True, exist_ok=True)

        X_path = target_dir / "X.csv"
        y_path = target_dir / "y.csv"
        metadata_path = target_dir / "metadata.json"

        X.to_csv(X_path, index=False)
        y.to_frame(metadata.get("target_output_name", target)).to_csv(y_path, index=False)
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

        exported[f"{target}_X"] = X_path
        exported[f"{target}_y"] = y_path
        exported[f"{target}_metadata"] = metadata_path

    return exported


def build_premodeling_tables(
    df: pd.DataFrame,
    config: PremodelingConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or PremodelingConfig()
    datasets = build_premodeling_datasets(df, config=config)

    return {
        "variable_plan": build_variable_plan_table(df, config=config),
        "dataset_summary": build_dataset_summary_table(datasets),
        "feature_summary": build_feature_summary_table(datasets),
    }
