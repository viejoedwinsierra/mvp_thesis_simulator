from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any

from premodeling_modeling.statistical_models import fit_statistical_models


def _params_table(target: str, model_type: str, result) -> pd.DataFrame:
    params = getattr(result, "params", pd.Series(dtype=float))
    pvalues = getattr(result, "pvalues", pd.Series(dtype=float))
    conf_int = None

    try:
        conf_int = result.conf_int()
    except Exception:
        conf_int = None

    rows = []

    for name, coef in params.items():
        rows.append({
            "target": target,
            "model_type": model_type,
            "term": name,
            "coefficient": coef,
            "p_value": pvalues.get(name, np.nan) if hasattr(pvalues, "get") else np.nan,
            "ci_lower": conf_int.loc[name, 0] if conf_int is not None and name in conf_int.index else np.nan,
            "ci_upper": conf_int.loc[name, 1] if conf_int is not None and name in conf_int.index else np.nan,
        })

    return pd.DataFrame(rows)


def build_model_summary_table(model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for target, payload in model_results.items():
        split = payload.get("split", {})

        for model_type, model_payload in payload.get("models", {}).items():
            if "error" in model_payload:
                rows.append({
                    "target": target,
                    "model_type": model_type,
                    "status": "error",
                    "train_rows": split.get("train_rows"),
                    "test_rows": split.get("test_rows"),
                    "metric_1": np.nan,
                    "metric_2": np.nan,
                    "metric_3": np.nan,
                    "notes": model_payload["error"],
                })
                continue

            metrics = model_payload.get("metrics", {})

            rows.append({
                "target": target,
                "model_type": model_type,
                "status": "ok",
                "train_rows": split.get("train_rows"),
                "test_rows": split.get("test_rows"),
                "metric_1": next(iter(metrics.values())) if metrics else np.nan,
                "metric_2": list(metrics.values())[1] if len(metrics) > 1 else np.nan,
                "metric_3": list(metrics.values())[2] if len(metrics) > 2 else np.nan,
                "metrics": ", ".join([
                    f"{k}={v:.4f}" if isinstance(v, (float, int)) else f"{k}={v}"
                    for k, v in metrics.items()
                ]),
                "notes": "",
            })

    return pd.DataFrame(rows)


def build_model_coefficients_table(model_results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    tables = []

    for target, payload in model_results.items():
        for model_type, model_payload in payload.get("models", {}).items():
            if "result" not in model_payload:
                continue

            tables.append(_params_table(target, model_type, model_payload["result"]))

    if not tables:
        return pd.DataFrame(columns=[
            "target", "model_type", "term", "coefficient", "p_value", "ci_lower", "ci_upper"
        ])

    return pd.concat(tables, ignore_index=True)


def build_modeling_tables(
    df: pd.DataFrame,
    model_results: Dict[str, Dict[str, Any]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Construye tablas de modelamiento sin reentrenar innecesariamente.

    Mejora clave:
    - Permite reutilizar `model_results` ya entrenados.
    - Mantiene compatibilidad total (si no se pasa, entrena como antes).
    """

    if model_results is None:
        model_results = fit_statistical_models(df)

    return {
        "model_summary": build_model_summary_table(model_results),
        "model_coefficients": build_model_coefficients_table(model_results),
    }
