from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from premodeling_modeling.common import ensure_dir


def _save(fig, output_dir: str | Path, filename: str) -> Path:
    output_dir = ensure_dir(output_dir)
    path = output_dir / filename
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_actual_vs_predicted(y_true, y_pred, output_dir: str | Path, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(y_true, y_pred, alpha=0.45)

    min_value = np.nanmin([np.nanmin(y_true), np.nanmin(y_pred)])
    max_value = np.nanmax([np.nanmax(y_true), np.nanmax(y_pred)])

    ax.plot([min_value, max_value], [min_value, max_value], linestyle="--")
    ax.set_title("Real vs Predicho")
    ax.set_xlabel("Real")
    ax.set_ylabel("Predicho")
    ax.grid(True, alpha=0.3)

    return _save(fig, output_dir, filename)


def plot_residuals(y_true, y_pred, output_dir: str | Path, filename: str) -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_pred, residuals, alpha=0.45)
    ax.axhline(0, linestyle="--")
    ax.set_title("Residuos vs Predicción")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Residuo")
    ax.grid(True, alpha=0.3)

    return _save(fig, output_dir, filename)


def plot_residual_histogram(y_true, y_pred, output_dir: str | Path, filename: str) -> Path:
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(residuals, bins=30)
    ax.set_title("Distribución de residuos")
    ax.set_xlabel("Residuo")
    ax.set_ylabel("Frecuencia")
    ax.grid(True, alpha=0.3)

    return _save(fig, output_dir, filename)


def plot_coefficients(coefficients: pd.DataFrame, output_dir: str | Path, filename: str, top_n: int = 20) -> Path | None:
    if coefficients is None or coefficients.empty:
        return None

    coef = coefficients.copy()

    if "abs_coefficient" not in coef.columns and "coefficient" in coef.columns:
        coef["abs_coefficient"] = coef["coefficient"].abs()

    coef = coef.sort_values("abs_coefficient", ascending=False).head(top_n)
    coef = coef.sort_values("coefficient")

    fig, ax = plt.subplots(figsize=(10, max(5, len(coef) * 0.35)))
    ax.barh(coef["feature"], coef["coefficient"])
    ax.set_title(f"Top {top_n} coeficientes")
    ax.set_xlabel("Coeficiente")
    ax.grid(True, axis="x", alpha=0.3)

    return _save(fig, output_dir, filename)


def plot_confusion_matrix(cm: pd.DataFrame, output_dir: str | Path, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(6, 5))
    matrix = cm.values

    im = ax.imshow(matrix)
    ax.set_title("Matriz de confusión")
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))
    ax.set_xticklabels(cm.columns)
    ax.set_yticklabels(cm.index)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)

    return _save(fig, output_dir, filename)


def plot_metrics_by_technique(summary: pd.DataFrame, metric: str, output_dir: str | Path, filename: str) -> Path | None:
    if summary is None or summary.empty or metric not in summary.columns:
        return None

    data = summary.dropna(subset=[metric]).copy()

    if data.empty:
        return None

    label_col = "technique_target"
    data[label_col] = data["technique"].astype(str) + " | " + data["target"].astype(str)
    data = data.sort_values(metric)

    fig, ax = plt.subplots(figsize=(11, max(5, len(data) * 0.35)))
    ax.barh(data[label_col], data[metric])
    ax.set_title(f"Comparación de técnicas por {metric}")
    ax.set_xlabel(metric)
    ax.grid(True, axis="x", alpha=0.3)

    return _save(fig, output_dir, filename)
