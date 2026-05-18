from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)


CLASSIFICATION_TECHNIQUES = {
    "logistic_regression",
    "sgd_classifier",
    "random_forest_classifier",
    "gradient_boosting_classifier",
    "hist_gradient_boosting_classifier",
}

REGRESSION_TECHNIQUES = {
    "linear_regression",
    "ridge_regressor",
    "lasso_regressor",
    "elastic_net_regressor",
    "sgd_regressor",
    "random_forest_regressor",
    "gradient_boosting_regressor",
    "hist_gradient_boosting_regressor",
}


def infer_task_type(technique: str) -> str:
    """
    Inferencia robusta del tipo de tarea.

    Mantiene compatibilidad con nombres antiguos tipo *_classifier / *_regressor,
    pero también soporta modelos lineales como logistic_regression y linear_regression.
    """

    if technique in CLASSIFICATION_TECHNIQUES:
        return "classification"

    if technique in REGRESSION_TECHNIQUES:
        return "regression"

    if "classifier" in technique:
        return "classification"

    if "regressor" in technique:
        return "regression"

    if technique.startswith("logistic"):
        return "classification"

    if technique.startswith("linear"):
        return "regression"

    return "unknown"


def discover_ml_models(modeling_root: str | Path = "output/modeling/ml") -> list[dict[str, Any]]:
    """
    Descubre modelos guardados en la estructura:

    output/modeling/ml/<technique>/models/*.joblib

    Compatible con:
    - modelos ML no lineales;
    - modelos lineales;
    - modelos regularizados;
    - modelos SGD.
    """

    root = Path(modeling_root)

    if not root.exists():
        raise FileNotFoundError(f"No existe el directorio de modelos ML: {root}")

    models: list[dict[str, Any]] = []

    for model_file in sorted(root.rglob("models/*.joblib")):
        technique = model_file.parent.parent.name
        filename = model_file.stem

        if filename.startswith(technique + "_"):
            target = filename.replace(technique + "_", "", 1)
        else:
            target = "unknown"

        task_type = infer_task_type(technique)

        models.append({
            "family": "machine_learning",
            "technique": technique,
            "target": target,
            "task_type": task_type,
            "model_path": str(model_file),
        })

    if not models:
        raise FileNotFoundError(
            f"No se encontraron modelos .joblib en {root}/<modelo>/models/"
        )

    return models


def _infer_target_from_model_info(model_info: dict[str, Any]) -> str:
    target = model_info.get("target")
    if target and target != "unknown":
        return target

    task_type = model_info.get("task_type")
    technique = model_info.get("technique", "")

    if task_type == "classification" or technique in CLASSIFICATION_TECHNIQUES:
        return "has_error"

    return "storage_cost"


def _get_prediction_score(model, X, y_pred):
    """
    Obtiene score continuo para clasificación.

    Para pipelines de sklearn:
    - model.predict_proba(X)
    - model.decision_function(X)

    Si no existe score continuo, retorna y_pred.
    """

    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except Exception:
            pass

    if hasattr(model, "decision_function"):
        try:
            return model.decision_function(X)
        except Exception:
            pass

    return y_pred


def _safe_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _regression_metrics(y_true, y_pred, technique: str, target: str) -> dict[str, Any]:
    return {
        "family": "machine_learning",
        "technique": technique,
        "target": target,
        "task_type": "regression",
        "status": "ok",
        "n": len(y_true),
        "mae": mean_absolute_error(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": _safe_rmse(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def _classification_metrics(y_true, y_pred, y_score, technique: str, target: str) -> dict[str, Any]:
    row = {
        "family": "machine_learning",
        "technique": technique,
        "target": target,
        "task_type": "classification",
        "status": "ok",
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        row["roc_auc"] = roc_auc_score(y_true, y_score)
    except Exception:
        row["roc_auc"] = np.nan

    try:
        row["pr_auc"] = average_precision_score(y_true, y_score)
    except Exception:
        row["pr_auc"] = np.nan

    return row


def _prepare_evaluation_frame(df_new: pd.DataFrame, target: str):
    if target not in df_new.columns:
        raise ValueError(f"El target '{target}' no existe en el dataset nuevo.")

    X = df_new.drop(columns=[target], errors="ignore")
    y = df_new[target].copy()

    # Columnas de identificación o trazabilidad que no deben entrar a inferencia.
    X = X.drop(
        columns=[
            "source_file",
            "source_path",
            "content_hash",
            "hash_head",
            "hash_tail",
            "file_id",
            "run_id",
            "sequence",
        ],
        errors="ignore",
    )

    if target in {"has_error", "storage_cost"}:
        y = pd.to_numeric(y, errors="coerce")

    valid_mask = y.notna()

    X = X.loc[valid_mask].copy()
    y = y.loc[valid_mask].copy()

    return X, y


def evaluate_single_ml_model(
    df_new: pd.DataFrame,
    model_info: dict[str, Any],
    threshold: float = 0.5,
) -> tuple[dict[str, Any], pd.DataFrame]:
    technique = model_info.get("technique", "unknown")
    task_type = model_info.get("task_type", infer_task_type(technique))
    target = _infer_target_from_model_info(model_info)
    model_path = Path(model_info["model_path"])

    if task_type == "unknown":
        return {
            "family": "machine_learning",
            "technique": technique,
            "target": target,
            "task_type": task_type,
            "status": "error",
            "error_message": f"No se pudo inferir task_type para technique={technique}",
            "model_path": str(model_path),
        }, pd.DataFrame()

    try:
        model = joblib.load(model_path)
        X, y = _prepare_evaluation_frame(df_new, target)

        y_pred_raw = model.predict(X)

        if task_type == "classification":
            y_score = _get_prediction_score(model, X, y_pred_raw)

            if np.issubdtype(np.asarray(y_score).dtype, np.number):
                y_pred = (np.asarray(y_score) >= threshold).astype(int)
            else:
                y_pred = y_pred_raw

            y_eval = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

            metrics = _classification_metrics(
                y_true=y_eval,
                y_pred=y_pred,
                y_score=y_score,
                technique=technique,
                target=target,
            )

            metrics["threshold"] = threshold

            predictions = pd.DataFrame({
                "technique": technique,
                "target": target,
                "threshold": threshold,
                "y_true": y_eval.values,
                "y_pred": y_pred,
                "score": y_score,
            })

        elif task_type == "regression":
            y_eval = pd.to_numeric(y, errors="coerce")
            y_pred = pd.to_numeric(pd.Series(y_pred_raw, index=y_eval.index), errors="coerce")

            valid = y_eval.notna() & y_pred.notna()

            y_eval = y_eval[valid]
            y_pred = y_pred[valid]

            metrics = _regression_metrics(
                y_true=y_eval,
                y_pred=y_pred,
                technique=technique,
                target=target,
            )

            predictions = pd.DataFrame({
                "technique": technique,
                "target": target,
                "y_true": y_eval.values,
                "y_pred": y_pred.values,
                "residual": y_eval.values - y_pred.values,
            })

        else:
            raise ValueError(f"task_type no soportado: {task_type}")

        metrics["model_path"] = str(model_path)
        return metrics, predictions

    except Exception as exc:
        metrics = {
            "family": "machine_learning",
            "technique": technique,
            "target": target,
            "task_type": task_type,
            "status": "error",
            "error_message": str(exc),
            "model_path": str(model_path),
        }

        return metrics, pd.DataFrame()


def evaluate_single_ml_model_multi_threshold(
    df_new: pd.DataFrame,
    model_info: dict[str, Any],
    thresholds: list[float] | tuple[float, ...] = (0.05, 0.10, 0.20, 0.30, 0.50),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Evaluación opcional multi-threshold para modelos de clasificación.

    No reemplaza evaluate_single_ml_model; sirve cuando quieres analizar sensibilidad
    de precision/recall/f1 frente al threshold.
    """

    technique = model_info.get("technique", "unknown")
    task_type = model_info.get("task_type", infer_task_type(technique))

    if task_type != "classification":
        metrics, predictions = evaluate_single_ml_model(
            df_new=df_new,
            model_info=model_info,
            threshold=0.5,
        )
        return pd.DataFrame([metrics]), {f"{technique}_{model_info.get('target')}": predictions}

    rows = []
    predictions_by_threshold: dict[str, pd.DataFrame] = {}

    for threshold in thresholds:
        metrics, predictions = evaluate_single_ml_model(
            df_new=df_new,
            model_info=model_info,
            threshold=float(threshold),
        )

        rows.append(metrics)
        key = f"{technique}_{model_info.get('target')}_threshold_{threshold}"
        predictions_by_threshold[key] = predictions

    return pd.DataFrame(rows), predictions_by_threshold


def evaluate_saved_ml_models(
    df_new: pd.DataFrame,
    modeling_root: str | Path = "output/modeling/ml",
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    models = discover_ml_models(modeling_root)

    metrics_rows = []
    detailed_results: dict[str, pd.DataFrame] = {}

    for model_info in models:
        metrics, predictions = evaluate_single_ml_model(
            df_new=df_new,
            model_info=model_info,
            threshold=threshold,
        )

        metrics_rows.append(metrics)

        key = f"{model_info.get('technique')}_{model_info.get('target')}"
        detailed_results[key] = predictions

    metrics_df = pd.DataFrame(metrics_rows)

    return metrics_df, detailed_results


def export_ml_evaluation_results(
    metrics_df: pd.DataFrame,
    detailed_results: dict[str, pd.DataFrame],
    output_dir: str | Path = "output/ml_model_evaluation",
) -> dict[str, str]:
    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    predictions_dir = output_dir / "predictions"

    tables_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = tables_dir / "ml_evaluation_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    prediction_files = {}

    for name, df in detailed_results.items():
        if df is None or df.empty:
            continue

        safe_name = name.replace("\\", "_").replace("/", "_").replace(" ", "_")
        path = predictions_dir / f"{safe_name}_predictions.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        prediction_files[name] = str(path)

    manifest = {
        "metrics_csv": str(metrics_path),
        "prediction_files": prediction_files,
        "total_models": len(metrics_df),
        "ok": int((metrics_df.get("status") == "ok").sum()) if "status" in metrics_df else 0,
        "error": int((metrics_df.get("status") == "error").sum()) if "status" in metrics_df else 0,
    }

    manifest_path = output_dir / "manifest_ml_model_evaluation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "metrics_csv": str(metrics_path),
        "manifest_json": str(manifest_path),
        "predictions_dir": str(predictions_dir),
    }
