from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml_modeling.common import ensure_dir


def _save(fig, output_dir: str | Path, filename: str) -> Path:
    output_dir = ensure_dir(output_dir)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_actual_vs_predicted(y_true, y_pred, output_dir: str | Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.45)
    min_value = np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)])
    max_value = np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)])
    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    ax.set_title("Real vs Predicho")
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicho")
    ax.grid(True, alpha=0.3)
    return _save(fig, output_dir, "actual_vs_predicted.png")


def plot_residuals(y_true, y_pred, output_dir: str | Path) -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.45)
    ax.axhline(0, linestyle="--")
    ax.set_title("Residuos vs Predicción")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Residuo")
    ax.grid(True, alpha=0.3)
    return _save(fig, output_dir, "residuals_vs_prediction.png")


def plot_residual_histogram(y_true, y_pred, output_dir: str | Path) -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=30)
    ax.set_title("Distribución de residuos")
    ax.set_xlabel("Residuo")
    ax.set_ylabel("Frecuencia")
    ax.grid(True, alpha=0.3)
    return _save(fig, output_dir, "residual_histogram.png")


def plot_feature_importance(feature_importance: pd.DataFrame, output_dir: str | Path, top_n: int = 20) -> Path | None:
    if feature_importance is None or feature_importance.empty:
        return None

    data = feature_importance.copy()

    value_col = "importance" if "importance" in data.columns else "coefficient"
    if value_col not in data.columns:
        return None

    data = data.sort_values("abs_value", ascending=False).head(top_n)
    data = data.sort_values(value_col)

    fig, ax = plt.subplots(figsize=(10, max(5, len(data) * 0.35)))
    ax.barh(data["feature"], data[value_col])
    ax.set_title(f"Top {top_n} variables")
    ax.set_xlabel(value_col)
    ax.grid(True, axis="x", alpha=0.3)
    return _save(fig, output_dir, "feature_importance.png")


def plot_confusion_matrix(y_true, y_pred, output_dir: str | Path) -> Path:
    labels = sorted(set(list(y_true) + list(y_pred)))
    matrix = pd.crosstab(
        pd.Series(y_true, name="Real"),
        pd.Series(y_pred, name="Predicho"),
        dropna=False,
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix.values)

    ax.set_title("Matriz de confusión")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticklabels(matrix.index)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix.values[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    return _save(fig, output_dir, "confusion_matrix.png")


def plot_metric_comparison(metrics: pd.DataFrame, output_dir: str | Path, metric: str) -> Path | None:
    if metrics is None or metrics.empty or metric not in metrics.columns:
        return None

    data = metrics.dropna(subset=[metric]).copy()
    if data.empty:
        return None

    data["label"] = data["technique"].astype(str) + " | " + data["target"].astype(str)
    data = data.sort_values(metric)

    fig, ax = plt.subplots(figsize=(11, max(5, len(data) * 0.35)))
    ax.barh(data["label"], data[metric])
    ax.set_title(f"Comparación de modelos por {metric}")
    ax.set_xlabel(metric)
    ax.grid(True, axis="x", alpha=0.3)
    return _save(fig, output_dir, f"comparison_{metric}.png")
