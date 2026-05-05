from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    explained_variance_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)

from premodeling_modeling.modeling_persistence import load_model_registry, load_saved_model
from premodeling_modeling.premodeling_tables import prepare_target_dataset


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .lower()
    )


def _extract_registry_items(registry: Any) -> list[dict[str, Any]]:
    """Soporta registry anterior tipo list y registry nuevo tipo dict.

    Registry anterior:
        [
            {"target": ..., "model_type": ...}
        ]

    Registry nuevo:
        {
            "registry_version": "1.0",
            "n_models": 5,
            "models": [...]
        }
    """
    if isinstance(registry, dict):
        items = registry.get("models", [])
    elif isinstance(registry, list):
        items = registry
    else:
        raise ValueError(
            "Formato de registry no soportado. "
            "Se esperaba una lista o un diccionario con clave 'models'."
        )

    if not isinstance(items, list):
        raise ValueError("La colección de modelos del registry debe ser una lista.")

    return [item for item in items if isinstance(item, dict)]


def _validate_threshold(threshold: float) -> None:
    if not 0 <= threshold <= 1:
        raise ValueError("threshold debe estar entre 0 y 1.")


def _validate_model_file(model_path: str | Path) -> Path:
    path = Path(model_path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de modelo guardado: {path}")

    return path


def _align_X_for_saved_model(
    X: pd.DataFrame,
    exog_names: list[str],
) -> pd.DataFrame:
    """Alinea columnas nuevas con las columnas usadas por statsmodels.

    Si en datos nuevos falta una dummy, se crea en 0.
    Si sobran columnas, se eliminan.
    """
    X_num = X.copy()

    for col in X_num.columns:
        if not pd.api.types.is_numeric_dtype(X_num[col]):
            X_num[col] = pd.to_numeric(X_num[col], errors="coerce")

    X_num = X_num.replace([np.inf, -np.inf], np.nan)

    medians = X_num.median(numeric_only=True)
    X_num = X_num.fillna(medians)
    X_num = X_num.fillna(0.0)

    X_sm = sm.add_constant(X_num, has_constant="add")

    for col in exog_names:
        if col not in X_sm.columns:
            X_sm[col] = 0.0

    return X_sm[exog_names]


def _residual_summary(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    abs_error = np.abs(residuals)

    return {
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_median": float(np.median(residuals)),
        "abs_error_p50": float(np.percentile(abs_error, 50)),
        "abs_error_p90": float(np.percentile(abs_error, 90)),
        "abs_error_p95": float(np.percentile(abs_error, 95)),
        "abs_error_max": float(np.max(abs_error)),
    }


def _evaluate_regression(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))

    metrics: dict[str, float] = {
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "rmse": rmse,
        "median_ae": float(median_absolute_error(y_true_arr, y_pred_arr)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
        "explained_variance": float(explained_variance_score(y_true_arr, y_pred_arr)),
    }

    try:
        metrics["mape"] = float(mean_absolute_percentage_error(y_true_arr, y_pred_arr))
    except Exception:
        metrics["mape"] = np.nan

    metrics.update(_residual_summary(y_true_arr, y_pred_arr))
    return metrics


def _evaluate_classification(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.clip(np.asarray(y_prob), 0, 1)
    y_pred = (y_prob_arr >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[0, 1]).ravel()

    metrics: dict[str, float | int] = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_arr, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred)),
        "precision": float(precision_score(y_true_arr, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred, zero_division=0)),
        "positive_rate_observed": float(np.mean(y_true_arr)),
        "positive_rate_predicted": float(np.mean(y_pred)),
        "probability_mean": float(np.mean(y_prob_arr)),
        "probability_p50": float(np.percentile(y_prob_arr, 50)),
        "probability_p90": float(np.percentile(y_prob_arr, 90)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    except Exception:
        metrics["roc_auc"] = np.nan

    try:
        metrics["average_precision"] = float(average_precision_score(y_true_arr, y_prob_arr))
    except Exception:
        metrics["average_precision"] = np.nan

    try:
        metrics["log_loss"] = float(log_loss(y_true_arr, y_prob_arr, labels=[0, 1]))
    except Exception:
        metrics["log_loss"] = np.nan

    return metrics


def _metric_delta(
    training_metrics: dict[str, Any],
    evaluation_metrics: dict[str, Any],
) -> dict[str, float]:
    """Calcula delta evaluación - entrenamiento para métricas comunes."""
    deltas: dict[str, float] = {}

    for metric_name, evaluation_value in evaluation_metrics.items():
        if metric_name not in training_metrics:
            continue

        training_value = training_metrics.get(metric_name)

        if isinstance(training_value, (int, float)) and isinstance(evaluation_value, (int, float)):
            deltas[f"delta_{metric_name}"] = float(evaluation_value) - float(training_value)

    return deltas


def _build_prediction_frame(
    y_true: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    is_classification: bool,
    threshold: float,
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "target": target,
        "model_type": model_type,
        "y_true": np.asarray(y_true),
        "prediction": np.asarray(pred),
    })

    if is_classification:
        frame["predicted_class"] = (frame["prediction"] >= threshold).astype(int)
        frame["error"] = frame["predicted_class"] - frame["y_true"]
    else:
        frame["residual"] = frame["y_true"] - frame["prediction"]
        frame["absolute_error"] = np.abs(frame["residual"])

    return frame


def evaluate_saved_models(
    df_new: pd.DataFrame,
    model_dir: str | Path = "output/models/statistical",
    threshold: float = 0.5,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Evalúa modelos guardados sobre datos nuevos.

    Este paso NO entrena.
    Solo carga modelos previamente guardados y calcula métricas.

    Retorna:
    - metrics_df: tabla resumen de métricas por modelo;
    - detailed_results: y_true, predicciones, métricas y metadata para plots/HTML.
    """
    _validate_threshold(threshold)

    if df_new is None or df_new.empty:
        raise ValueError("df_new no puede estar vacío para evaluar modelos guardados.")

    registry_payload = load_model_registry(model_dir)
    registry_items = _extract_registry_items(registry_payload)

    if not registry_items:
        raise ValueError(
            f"No hay modelos registrados para evaluar en: {Path(model_dir) / 'model_registry.json'}"
        )

    rows: list[dict[str, Any]] = []
    detailed_results: dict[str, dict[str, Any]] = {}

    for item in registry_items:
        target = item.get("target")
        model_type = item.get("model_type")
        model_path = item.get("model_path")
        exog_names = item.get("exog_names", [])

        base_row = {
            "target": target,
            "model_type": model_type,
            "model_path": model_path,
        }

        try:
            if not target or not model_type or not model_path:
                raise ValueError("Registro incompleto: target, model_type y model_path son obligatorios.")

            if not exog_names:
                raise ValueError(f"El modelo {target} / {model_type} no tiene exog_names en registry.")

            model_path_checked = _validate_model_file(model_path)

            X_new, y_new, metadata = prepare_target_dataset(df_new, target)
            X_eval = _align_X_for_saved_model(X_new, list(exog_names))

            model = load_saved_model(model_path_checked)
            pred = np.asarray(model.predict(X_eval))

            is_classification = metadata.get("problem_type") == "classification_binary"

            if is_classification:
                pred = np.clip(pred, 0, 1)
                metrics = _evaluate_classification(y_new, pred, threshold=threshold)
            else:
                metrics = _evaluate_regression(y_new, pred)

            training_metrics = item.get("metrics_training_run", {}) or {}
            metric_deltas = _metric_delta(training_metrics, metrics)

            row = {
                **base_row,
                "status": "ok",
                "n_rows_evaluated": int(len(X_new)),
                "n_features_evaluated": int(X_eval.shape[1]),
                "problem_type": metadata.get("problem_type"),
                **metrics,
                **metric_deltas,
            }

            result_key = f"{target}__{model_type}"
            prediction_frame = _build_prediction_frame(
                y_true=y_new,
                pred=pred,
                target=target,
                model_type=model_type,
                is_classification=is_classification,
                threshold=threshold,
            )

            detailed_results[result_key] = {
                "target": target,
                "model_type": model_type,
                "problem_type": metadata.get("problem_type"),
                "model_path": str(model_path_checked),
                "exog_names": list(exog_names),
                "metadata": metadata,
                "registry_item": item,
                "training_metrics": training_metrics,
                "metrics": metrics,
                "metric_deltas": metric_deltas,
                "y_true": y_new,
                "pred": pred,
                "prediction_frame": prediction_frame,
            }

        except Exception as exc:
            row = {
                **base_row,
                "status": "error",
                "n_rows_evaluated": 0,
                "n_features_evaluated": 0,
                "error": str(exc),
            }

        rows.append(row)

    return pd.DataFrame(rows), detailed_results


def build_evaluation_comparison_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Construye tabla compacta de diferencias entrenamiento vs evaluación."""
    if metrics_df.empty:
        return pd.DataFrame()

    delta_cols = [c for c in metrics_df.columns if c.startswith("delta_")]

    base_cols = [
        "target",
        "model_type",
        "status",
        "problem_type",
        "n_rows_evaluated",
    ]

    metric_cols = [
        "mae",
        "rmse",
        "r2",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
    ]

    selected_cols = [
        c for c in base_cols + metric_cols + delta_cols
        if c in metrics_df.columns
    ]

    return metrics_df[selected_cols].copy()


def build_evaluation_status_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Genera una tabla simple de estado por modelo."""
    if metrics_df.empty:
        return pd.DataFrame(columns=[
            "target", "model_type", "status", "recommendation", "notes"
        ])

    rows: list[dict[str, Any]] = []

    for _, row in metrics_df.iterrows():
        status = row.get("status")
        recommendation = "REVISAR"
        notes: list[str] = []

        if status != "ok":
            recommendation = "ERROR"
            notes.append(str(row.get("error", "")))
        else:
            problem_type = row.get("problem_type")

            if problem_type == "classification_binary":
                roc_auc = row.get("roc_auc", np.nan)
                f1 = row.get("f1", np.nan)

                if pd.notna(roc_auc) and roc_auc >= 0.75 and pd.notna(f1) and f1 >= 0.30:
                    recommendation = "OK"
                elif pd.notna(roc_auc) and roc_auc >= 0.70:
                    recommendation = "AJUSTAR_UMBRAL"
                    notes.append("ROC AUC aceptable, revisar F1/threshold.")
                else:
                    recommendation = "REENTRENAR_O_REVISAR"
                    notes.append("Desempeño bajo en clasificación.")
            else:
                r2 = row.get("r2", np.nan)

                if pd.notna(r2) and r2 >= 0.80:
                    recommendation = "OK"
                elif pd.notna(r2) and r2 >= 0.60:
                    recommendation = "MONITOREAR"
                    notes.append("R2 moderado en evaluación.")
                else:
                    recommendation = "REENTRENAR_O_REVISAR"
                    notes.append("R2 bajo en evaluación.")

            for delta_col in [c for c in metrics_df.columns if c.startswith("delta_")]:
                delta_value = row.get(delta_col, np.nan)
                if pd.notna(delta_value):
                    if delta_col in {"delta_r2", "delta_roc_auc", "delta_f1", "delta_accuracy"} and delta_value < -0.10:
                        notes.append(f"Caída relevante: {delta_col}={delta_value:.4f}")
                    if delta_col in {"delta_mae", "delta_rmse"} and delta_value > 0:
                        notes.append(f"Error aumentó: {delta_col}={delta_value:.4f}")

        rows.append({
            "target": row.get("target"),
            "model_type": row.get("model_type"),
            "status": status,
            "recommendation": recommendation,
            "notes": " | ".join(notes),
        })

    return pd.DataFrame(rows)


def export_evaluation_results(
    metrics_df: pd.DataFrame,
    output_dir: str | Path = "output/model_evaluation",
    detailed_results: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Exporta métricas, comparación, estado y predicciones de evaluación."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "model_evaluation_metrics.csv"
    json_path = output_dir / "model_evaluation_metrics.json"
    comparison_path = output_dir / "model_evaluation_comparison.csv"
    status_path = output_dir / "model_evaluation_status.csv"
    manifest_path = output_dir / "model_evaluation_export_manifest.json"

    metrics_df.to_csv(metrics_path, index=False)
    json_path.write_text(
        metrics_df.to_json(orient="records", indent=2, force_ascii=False),
        encoding="utf-8",
    )

    comparison_df = build_evaluation_comparison_table(metrics_df)
    status_df = build_evaluation_status_table(metrics_df)

    comparison_df.to_csv(comparison_path, index=False)
    status_df.to_csv(status_path, index=False)

    exported: dict[str, Path] = {
        "metrics_csv": metrics_path,
        "metrics_json": json_path,
        "comparison_csv": comparison_path,
        "status_csv": status_path,
    }

    prediction_files: list[str] = []

    if detailed_results:
        predictions_dir = output_dir / "predictions"
        predictions_dir.mkdir(parents=True, exist_ok=True)

        for key, payload in detailed_results.items():
            prediction_frame = payload.get("prediction_frame")
            if isinstance(prediction_frame, pd.DataFrame):
                pred_path = predictions_dir / f"{_safe_name(key)}_predictions.csv"
                prediction_frame.to_csv(pred_path, index=False)
                exported[f"predictions_{_safe_name(key)}"] = pred_path
                prediction_files.append(str(pred_path))

    manifest = {
        "created_at": datetime.utcnow().isoformat(),
        "n_models": int(len(metrics_df)),
        "n_success": int((metrics_df.get("status") == "ok").sum()) if "status" in metrics_df else 0,
        "n_error": int((metrics_df.get("status") == "error").sum()) if "status" in metrics_df else 0,
        "files": {k: str(v) for k, v in exported.items()},
        "prediction_files": prediction_files,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    exported["manifest_json"] = manifest_path
    return exported
