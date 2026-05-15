from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, output_dir: str | Path, filename: str) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_metric_bar(metrics_df: pd.DataFrame, metric: str, output_dir: str | Path) -> Path | None:
    if metrics_df is None or metrics_df.empty or metric not in metrics_df.columns:
        return None

    data = metrics_df[metrics_df.get("status", "ok") == "ok"].dropna(subset=[metric]).copy()

    if data.empty:
        return None

    data["label"] = data["technique"].astype(str) + " | " + data["target"].astype(str)
    data = data.sort_values(metric)

    fig, ax = plt.subplots(figsize=(11, max(5, len(data) * 0.38)))
    ax.barh(data["label"], data[metric])
    ax.set_title(f"Evaluación de modelos guardados por {metric}")
    ax.set_xlabel(metric)
    ax.grid(True, axis="x", alpha=0.3)

    return _save(fig, output_dir, f"evaluation_{metric}.png")


def plot_regression_predictions(predictions: pd.DataFrame, technique: str, output_dir: str | Path) -> list[Path]:
    paths = []

    if predictions is None or predictions.empty:
        return paths

    if "residual" not in predictions.columns:
        return paths

    y_true = predictions["y_true"]
    y_pred = predictions["y_pred"]
    residual = predictions["residual"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.45)
    min_value = np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)])
    max_value = np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)])
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    ax.set_title(f"Real vs predicho - {technique}")
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicho")
    ax.grid(True, alpha=0.3)
    paths.append(_save(fig, output_dir, f"{technique}_actual_vs_predicted.png"))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residual, alpha=0.45)
    ax.axhline(0, linestyle="--")
    ax.set_title(f"Residuos vs predicción - {technique}")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Residuo")
    ax.grid(True, alpha=0.3)
    paths.append(_save(fig, output_dir, f"{technique}_residuals.png"))

    return paths


def plot_classification_confusion(predictions: pd.DataFrame, technique: str, output_dir: str | Path) -> Path | None:
    if predictions is None or predictions.empty:
        return None

    if "score" not in predictions.columns:
        return None

    cm = pd.crosstab(
        predictions["y_true"],
        predictions["y_pred"],
        rownames=["Real"],
        colnames=["Predicho"],
        dropna=False,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm.values)

    ax.set_title(f"Matriz de confusión - {technique}")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xticklabels(cm.columns)
    ax.set_yticklabels(cm.index)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm.values[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)

    return _save(fig, output_dir, f"{technique}_confusion_matrix.png")


def run_ml_evaluation_plots(
    metrics_df: pd.DataFrame,
    detailed_results: dict[str, pd.DataFrame],
    output_dir: str | Path = "output/plots/ml_model_evaluation",
) -> list[Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_files: list[Path] = []

    for metric in ["r2", "rmse", "mae", "f1", "roc_auc", "recall", "precision"]:
        plot = plot_metric_bar(metrics_df, metric, output_dir)
        if plot:
            plot_files.append(plot)

    for key, predictions in detailed_results.items():
        if predictions is None or predictions.empty:
            continue

        technique = str(predictions["technique"].iloc[0]) if "technique" in predictions.columns else key

        if "residual" in predictions.columns:
            plot_files.extend(plot_regression_predictions(predictions, technique, output_dir))
        else:
            plot = plot_classification_confusion(predictions, technique, output_dir)
            if plot:
                plot_files.append(plot)

    return plot_files
