from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from premodeling_modeling.model_evaluation import evaluate_saved_models


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .lower()
    )


def _save(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def _plot_regression_observed_vs_predicted(
    y_true: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path:
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, pred, alpha=0.35)
    plt.xlabel("Valor observado")
    plt.ylabel("Valor estimado")
    plt.title(f"Evaluación - observado vs estimado - {target} - {model_type}")

    return _save(
        output_dir
        / f"evaluation_observed_vs_predicted_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_regression_residuals_vs_predicted(
    y_true: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path:
    residuals = np.asarray(y_true) - np.asarray(pred)

    plt.figure(figsize=(8, 5))
    plt.scatter(pred, residuals, alpha=0.35)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Valor estimado")
    plt.ylabel("Residuo")
    plt.title(f"Evaluación - residuos vs estimado - {target} - {model_type}")

    return _save(
        output_dir
        / f"evaluation_residuals_vs_predicted_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_regression_residual_histogram(
    y_true: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path:
    residuals = np.asarray(y_true) - np.asarray(pred)

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=50)
    plt.xlabel("Residuo")
    plt.ylabel("Frecuencia")
    plt.title(f"Evaluación - histograma de residuos - {target} - {model_type}")

    return _save(
        output_dir
        / f"evaluation_residual_histogram_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_regression_absolute_error_vs_predicted(
    y_true: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path:
    abs_error = np.abs(np.asarray(y_true) - np.asarray(pred))

    plt.figure(figsize=(8, 5))
    plt.scatter(pred, abs_error, alpha=0.35)
    plt.xlabel("Valor estimado")
    plt.ylabel("Error absoluto")
    plt.title(f"Evaluación - error absoluto vs estimado - {target} - {model_type}")

    return _save(
        output_dir
        / f"evaluation_absolute_error_vs_predicted_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_regression_error_boxplot(
    y_true: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path:
    abs_error = np.abs(np.asarray(y_true) - np.asarray(pred))

    plt.figure(figsize=(7, 5))
    plt.boxplot(abs_error, vert=True, showfliers=False)
    plt.ylabel("Error absoluto")
    plt.title(f"Evaluación - distribución de error absoluto - {target} - {model_type}")

    return _save(
        output_dir
        / f"evaluation_absolute_error_boxplot_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_regression_log_observed_vs_log_predicted(
    y_true: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path | None:
    y_true_arr = np.asarray(y_true)
    pred_arr = np.asarray(pred)

    if np.nanmin(y_true_arr) < 0:
        return None

    pred_arr = np.maximum(pred_arr, 0)

    plt.figure(figsize=(8, 5))
    plt.scatter(np.log1p(y_true_arr), np.log1p(pred_arr), alpha=0.35)
    plt.xlabel("log1p valor observado")
    plt.ylabel("log1p valor estimado")
    plt.title(f"Evaluación - log observado vs log estimado - {target} - {model_type}")

    return _save(
        output_dir
        / f"evaluation_log_observed_vs_predicted_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_classification_confusion_matrix(
    y_true: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
    threshold: float,
) -> Path:
    y_pred = (np.asarray(pred) >= threshold).astype(int)

    plt.figure(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.title(f"Evaluación - matriz de confusión - {target} - {model_type}")

    return _save(
        output_dir
        / f"evaluation_confusion_matrix_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_classification_roc_curve(
    y_true: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path | None:
    if pd.Series(y_true).nunique(dropna=True) < 2:
        return None

    plt.figure(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_true, pred)
    plt.title(f"Evaluación - curva ROC - {target} - {model_type}")

    return _save(
        output_dir
        / f"evaluation_roc_curve_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_classification_precision_recall_curve(
    y_true: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path | None:
    if pd.Series(y_true).nunique(dropna=True) < 2:
        return None

    plt.figure(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y_true, pred)
    plt.title(f"Evaluación - curva Precision-Recall - {target} - {model_type}")

    return _save(
        output_dir
        / f"evaluation_precision_recall_curve_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_classification_calibration_curve(
    y_true: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path | None:
    if pd.Series(y_true).nunique(dropna=True) < 2:
        return None

    plt.figure(figsize=(7, 6))
    CalibrationDisplay.from_predictions(y_true, pred, n_bins=10)
    plt.title(f"Evaluación - calibración - {target} - {model_type}")

    return _save(
        output_dir
        / f"evaluation_calibration_curve_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_classification_probability_distribution(
    y_true: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path:
    y_true_series = pd.Series(y_true).reset_index(drop=True)
    pred_arr = np.asarray(pred)

    plt.figure(figsize=(8, 5))

    if y_true_series.nunique(dropna=True) > 1:
        plt.hist(pred_arr[y_true_series == 0], bins=30, alpha=0.55, label="Clase 0")
        plt.hist(pred_arr[y_true_series == 1], bins=30, alpha=0.55, label="Clase 1")
        plt.legend()
    else:
        plt.hist(pred_arr, bins=30)

    plt.xlabel("Probabilidad estimada")
    plt.ylabel("Frecuencia")
    plt.title(f"Evaluación - probabilidades - {target} - {model_type}")

    return _save(
        output_dir
        / f"evaluation_probability_distribution_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_metrics_bar(
    detailed_results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    rows: list[dict[str, Any]] = []

    for _, payload in detailed_results.items():
        metrics = payload.get("metrics", {})
        target = payload.get("target")
        model_type = payload.get("model_type")
        problem_type = payload.get("problem_type")

        if problem_type == "classification_binary":
            for metric in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
                if metric in metrics and pd.notna(metrics[metric]):
                    rows.append({
                        "label": f"{target}\n{model_type}\n{metric}",
                        "value": float(metrics[metric]),
                    })
        else:
            for metric in ["r2", "explained_variance"]:
                if metric in metrics and pd.notna(metrics[metric]):
                    rows.append({
                        "label": f"{target}\n{model_type}\n{metric}",
                        "value": float(metrics[metric]),
                    })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    plt.figure(figsize=(max(10, len(df) * 0.65), 5))
    plt.bar(range(len(df)), df["value"])
    plt.xticks(range(len(df)), df["label"], rotation=75, ha="right")
    plt.ylabel("Valor de métrica")
    plt.title("Evaluación - resumen de métricas principales")

    return _save(output_dir / "evaluation_metrics_summary.png")


def _plot_metric_deltas(
    detailed_results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> Path | None:
    rows: list[dict[str, Any]] = []

    for _, payload in detailed_results.items():
        deltas = payload.get("metric_deltas", {})
        target = payload.get("target")
        model_type = payload.get("model_type")

        for metric, value in deltas.items():
            if isinstance(value, (int, float)) and pd.notna(value):
                rows.append({
                    "label": f"{target}\n{model_type}\n{metric.replace('delta_', '')}",
                    "delta": float(value),
                })

    if not rows:
        return None

    df = pd.DataFrame(rows)

    plt.figure(figsize=(max(10, len(df) * 0.65), 5))
    plt.bar(range(len(df)), df["delta"])
    plt.axhline(0, linestyle="--")
    plt.xticks(range(len(df)), df["label"], rotation=75, ha="right")
    plt.ylabel("Delta evaluación - entrenamiento")
    plt.title("Evaluación - cambio de métricas frente al entrenamiento")

    return _save(output_dir / "evaluation_metric_deltas.png")


def run_model_evaluation_plots(
    df_new: pd.DataFrame,
    model_dir: str | Path = "output/models/statistical",
    output_dir: str | Path = "output/plots/model_evaluation",
    threshold: float = 0.5,
    enabled: bool = True,
    detailed_results: dict[str, dict[str, Any]] | None = None,
) -> list[Path]:
    """Genera gráficos estadísticos de evaluación de modelos guardados.

    Mantiene compatibilidad:
    - conserva parámetros existentes;
    - respeta `enabled=False`;
    - agrega `detailed_results` opcional para evitar evaluar dos veces.

    Esta función NO entrena modelos.
    """

    if not enabled:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if detailed_results is None:
        _, detailed_results = evaluate_saved_models(
            df_new=df_new,
            model_dir=model_dir,
            threshold=threshold,
        )

    paths: list[Path] = []

    summary_path = _plot_metrics_bar(detailed_results, output_dir)
    if summary_path is not None:
        paths.append(summary_path)

    delta_path = _plot_metric_deltas(detailed_results, output_dir)
    if delta_path is not None:
        paths.append(delta_path)

    for _, payload in detailed_results.items():
        target = payload["target"]
        model_type = payload["model_type"]
        y_true = payload["y_true"]
        pred = np.asarray(payload["pred"])
        problem_type = payload.get("problem_type")

        is_classification = (
            problem_type == "classification_binary"
            or pd.Series(y_true).nunique(dropna=True) <= 2
        )

        if is_classification:
            pred = np.clip(pred, 0, 1)

            paths.append(
                _plot_classification_confusion_matrix(
                    y_true=y_true,
                    pred=pred,
                    target=target,
                    model_type=model_type,
                    output_dir=output_dir,
                    threshold=threshold,
                )
            )

            roc_path = _plot_classification_roc_curve(
                y_true=y_true,
                pred=pred,
                target=target,
                model_type=model_type,
                output_dir=output_dir,
            )
            if roc_path is not None:
                paths.append(roc_path)

            pr_path = _plot_classification_precision_recall_curve(
                y_true=y_true,
                pred=pred,
                target=target,
                model_type=model_type,
                output_dir=output_dir,
            )
            if pr_path is not None:
                paths.append(pr_path)

            calibration_path = _plot_classification_calibration_curve(
                y_true=y_true,
                pred=pred,
                target=target,
                model_type=model_type,
                output_dir=output_dir,
            )
            if calibration_path is not None:
                paths.append(calibration_path)

            paths.append(
                _plot_classification_probability_distribution(
                    y_true=y_true,
                    pred=pred,
                    target=target,
                    model_type=model_type,
                    output_dir=output_dir,
                )
            )

        else:
            paths.append(
                _plot_regression_observed_vs_predicted(
                    y_true=y_true,
                    pred=pred,
                    target=target,
                    model_type=model_type,
                    output_dir=output_dir,
                )
            )

            paths.append(
                _plot_regression_residuals_vs_predicted(
                    y_true=y_true,
                    pred=pred,
                    target=target,
                    model_type=model_type,
                    output_dir=output_dir,
                )
            )

            paths.append(
                _plot_regression_residual_histogram(
                    y_true=y_true,
                    pred=pred,
                    target=target,
                    model_type=model_type,
                    output_dir=output_dir,
                )
            )

            paths.append(
                _plot_regression_absolute_error_vs_predicted(
                    y_true=y_true,
                    pred=pred,
                    target=target,
                    model_type=model_type,
                    output_dir=output_dir,
                )
            )

            paths.append(
                _plot_regression_error_boxplot(
                    y_true=y_true,
                    pred=pred,
                    target=target,
                    model_type=model_type,
                    output_dir=output_dir,
                )
            )

            log_path = _plot_regression_log_observed_vs_log_predicted(
                y_true=y_true,
                pred=pred,
                target=target,
                model_type=model_type,
                output_dir=output_dir,
            )
            if log_path is not None:
                paths.append(log_path)

    return paths
