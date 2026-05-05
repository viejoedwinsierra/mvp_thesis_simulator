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

from premodeling_modeling.statistical_models import fit_statistical_models


try:
    import scipy.stats as stats
except Exception:  # pragma: no cover
    stats = None


def _save(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .lower()
    )


def _as_numeric_array(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _has_enough_points(y_test: pd.Series, pred: np.ndarray) -> bool:
    return len(y_test) > 1 and len(pred) > 1


def _plot_regression_observed_vs_predicted(
    y_test: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path:
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, pred, alpha=0.35)
    plt.xlabel("Valor observado")
    plt.ylabel("Valor estimado")
    plt.title(f"Observado vs estimado - {target} - {model_type}")

    return _save(
        output_dir
        / f"modeling_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_regression_residuals_vs_predicted(
    y_test: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path:
    residuals = _as_numeric_array(y_test) - _as_numeric_array(pred)

    plt.figure(figsize=(8, 5))
    plt.scatter(pred, residuals, alpha=0.35)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Valor estimado")
    plt.ylabel("Residuo")
    plt.title(f"Residuos vs estimado - {target} - {model_type}")

    return _save(
        output_dir
        / f"modeling_residuals_vs_predicted_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_regression_residual_histogram(
    y_test: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path:
    residuals = _as_numeric_array(y_test) - _as_numeric_array(pred)

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=50)
    plt.xlabel("Residuo")
    plt.ylabel("Frecuencia")
    plt.title(f"Distribución de residuos - {target} - {model_type}")

    return _save(
        output_dir
        / f"modeling_residual_histogram_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_regression_absolute_error_vs_predicted(
    y_test: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path:
    absolute_error = np.abs(_as_numeric_array(y_test) - _as_numeric_array(pred))

    plt.figure(figsize=(8, 5))
    plt.scatter(pred, absolute_error, alpha=0.35)
    plt.xlabel("Valor estimado")
    plt.ylabel("Error absoluto")
    plt.title(f"Error absoluto vs estimado - {target} - {model_type}")

    return _save(
        output_dir
        / f"modeling_absolute_error_vs_predicted_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_regression_qq_residuals(
    y_test: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path | None:
    if stats is None:
        return None

    residuals = _as_numeric_array(y_test) - _as_numeric_array(pred)
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) < 3:
        return None

    plt.figure(figsize=(7, 6))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f"QQ Plot de residuos - {target} - {model_type}")

    return _save(
        output_dir
        / f"modeling_qq_residuals_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_regression_log_observed_vs_log_predicted(
    y_test: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path | None:
    y_values = _as_numeric_array(y_test)
    pred_values = _as_numeric_array(pred)

    finite_mask = np.isfinite(y_values) & np.isfinite(pred_values)
    y_values = y_values[finite_mask]
    pred_values = pred_values[finite_mask]

    if len(y_values) < 2:
        return None

    pred_values = np.maximum(pred_values, 0)
    y_values = np.maximum(y_values, 0)

    plt.figure(figsize=(8, 5))
    plt.scatter(np.log1p(y_values), np.log1p(pred_values), alpha=0.35)
    plt.xlabel("log1p observado")
    plt.ylabel("log1p estimado")
    plt.title(f"log observado vs log estimado - {target} - {model_type}")

    return _save(
        output_dir
        / f"modeling_log_observed_vs_predicted_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_glm_deviance_residuals(
    result: Any,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path | None:
    if result is None:
        return None

    residuals = getattr(result, "resid_deviance", None)

    if residuals is None:
        return None

    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals[np.isfinite(residuals)]

    if len(residuals) < 2:
        return None

    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=50)
    plt.xlabel("Residuo deviance")
    plt.ylabel("Frecuencia")
    plt.title(f"Distribución de residuos deviance - {target} - {model_type}")

    return _save(
        output_dir
        / f"modeling_deviance_residuals_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_classification_probability_distribution(
    y_test: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path:
    plt.figure(figsize=(8, 5))
    plt.hist(pred, bins=30)
    plt.xlabel("Probabilidad estimada")
    plt.ylabel("Frecuencia")
    plt.title(f"Distribución de probabilidades - {target} - {model_type}")

    return _save(
        output_dir
        / f"modeling_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_classification_confusion_matrix(
    y_test: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
    threshold: float = 0.5,
) -> Path:
    y_pred = (_as_numeric_array(pred) >= threshold).astype(int)

    plt.figure(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"Matriz de confusión - {target} - {model_type}")

    return _save(
        output_dir
        / f"modeling_confusion_matrix_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_classification_roc_curve(
    y_test: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path | None:
    if pd.Series(y_test).nunique(dropna=True) < 2:
        return None

    plt.figure(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_test, pred)
    plt.title(f"Curva ROC - {target} - {model_type}")

    return _save(
        output_dir
        / f"modeling_roc_curve_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_classification_precision_recall_curve(
    y_test: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path | None:
    if pd.Series(y_test).nunique(dropna=True) < 2:
        return None

    plt.figure(figsize=(7, 6))
    PrecisionRecallDisplay.from_predictions(y_test, pred)
    plt.title(f"Curva Precision-Recall - {target} - {model_type}")

    return _save(
        output_dir
        / f"modeling_precision_recall_curve_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _plot_classification_calibration_curve(
    y_test: pd.Series,
    pred: np.ndarray,
    target: str,
    model_type: str,
    output_dir: Path,
) -> Path | None:
    if pd.Series(y_test).nunique(dropna=True) < 2:
        return None

    pred_values = np.clip(_as_numeric_array(pred), 0, 1)

    plt.figure(figsize=(7, 6))
    CalibrationDisplay.from_predictions(y_test, pred_values, n_bins=10)
    plt.title(f"Curva de calibración - {target} - {model_type}")

    return _save(
        output_dir
        / f"modeling_calibration_curve_{_safe_name(target)}_{_safe_name(model_type)}.png"
    )


def _append_if_not_none(paths: list[Path], path: Path | None) -> None:
    if path is not None:
        paths.append(path)


def _is_classification_target(y_test: pd.Series, model_type: str) -> bool:
    if "logistic" in str(model_type).lower():
        return True

    return pd.Series(y_test).nunique(dropna=True) <= 2


def _is_gamma_model(model_type: str) -> bool:
    normalized = str(model_type).lower()
    return "gamma" in normalized or "glm_gamma" in normalized


def run_modeling_plots(
    df: pd.DataFrame,
    output_dir: str | Path = "output/plots/modeling",
    enabled: bool = False,
    model_results: dict[str, dict] | None = None,
) -> list[Path]:
    """Genera gráficos de validación de modelos estadísticos.

    Mantiene compatibilidad:
    - conserva la firma pública original: df, output_dir y enabled;
    - agrega model_results como parámetro opcional para evitar reentrenamiento;
    - mantiene el mismo comportamiento cuando enabled=False;
    - conserva el gráfico base `modeling_<target>_<model>.png`.

    Gráficos por tipo de modelo:
    - regresión: observado vs estimado, residuos vs estimado, histograma de residuos,
      error absoluto vs estimado y QQ plot de residuos cuando scipy está disponible;
    - GLM Gamma: agrega residuos deviance y observado vs estimado en escala log1p;
    - clasificación: distribución de probabilidades, matriz de confusión, ROC,
      Precision-Recall y calibración.
    """
    if not enabled:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = model_results if model_results is not None else fit_statistical_models(df)
    paths: list[Path] = []

    for target, payload in results.items():
        for model_type, model_payload in payload.get("models", {}).items():
            if "predictions" not in model_payload or "y_test" not in model_payload:
                continue

            y_test = model_payload["y_test"]
            pred = np.asarray(model_payload["predictions"])

            if not _has_enough_points(y_test, pred):
                continue

            is_classification = _is_classification_target(y_test, model_type)

            if is_classification:
                paths.append(
                    _plot_classification_probability_distribution(
                        y_test=y_test,
                        pred=pred,
                        target=target,
                        model_type=model_type,
                        output_dir=output_dir,
                    )
                )

                paths.append(
                    _plot_classification_confusion_matrix(
                        y_test=y_test,
                        pred=pred,
                        target=target,
                        model_type=model_type,
                        output_dir=output_dir,
                    )
                )

                _append_if_not_none(
                    paths,
                    _plot_classification_roc_curve(
                        y_test=y_test,
                        pred=pred,
                        target=target,
                        model_type=model_type,
                        output_dir=output_dir,
                    ),
                )

                _append_if_not_none(
                    paths,
                    _plot_classification_precision_recall_curve(
                        y_test=y_test,
                        pred=pred,
                        target=target,
                        model_type=model_type,
                        output_dir=output_dir,
                    ),
                )

                _append_if_not_none(
                    paths,
                    _plot_classification_calibration_curve(
                        y_test=y_test,
                        pred=pred,
                        target=target,
                        model_type=model_type,
                        output_dir=output_dir,
                    ),
                )

            else:
                paths.append(
                    _plot_regression_observed_vs_predicted(
                        y_test=y_test,
                        pred=pred,
                        target=target,
                        model_type=model_type,
                        output_dir=output_dir,
                    )
                )

                paths.append(
                    _plot_regression_residuals_vs_predicted(
                        y_test=y_test,
                        pred=pred,
                        target=target,
                        model_type=model_type,
                        output_dir=output_dir,
                    )
                )

                paths.append(
                    _plot_regression_residual_histogram(
                        y_test=y_test,
                        pred=pred,
                        target=target,
                        model_type=model_type,
                        output_dir=output_dir,
                    )
                )

                paths.append(
                    _plot_regression_absolute_error_vs_predicted(
                        y_test=y_test,
                        pred=pred,
                        target=target,
                        model_type=model_type,
                        output_dir=output_dir,
                    )
                )

                _append_if_not_none(
                    paths,
                    _plot_regression_qq_residuals(
                        y_test=y_test,
                        pred=pred,
                        target=target,
                        model_type=model_type,
                        output_dir=output_dir,
                    ),
                )

                if _is_gamma_model(model_type):
                    _append_if_not_none(
                        paths,
                        _plot_glm_deviance_residuals(
                            result=model_payload.get("result"),
                            target=target,
                            model_type=model_type,
                            output_dir=output_dir,
                        ),
                    )

                    _append_if_not_none(
                        paths,
                        _plot_regression_log_observed_vs_log_predicted(
                            y_test=y_test,
                            pred=pred,
                            target=target,
                            model_type=model_type,
                            output_dir=output_dir,
                        ),
                    )

    return paths
