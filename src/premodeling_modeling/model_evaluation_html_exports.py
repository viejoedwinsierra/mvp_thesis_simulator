from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from premodeling_modeling.model_evaluation_exporter import ModelEvaluationExporter


def _ensure_parent_dir(path: str | Path) -> Path:
    """Crea el directorio padre del archivo de salida y retorna el Path normalizado."""
    normalized_path = Path(path)
    normalized_path.parent.mkdir(parents=True, exist_ok=True)
    return normalized_path


def _ensure_dir(path: str | Path) -> Path:
    """Crea un directorio si no existe y retorna el Path normalizado."""
    normalized_path = Path(path)
    normalized_path.mkdir(parents=True, exist_ok=True)
    return normalized_path


def _load_registry_payload(registry_path: Path) -> Any:
    """Carga el registry JSON de modelos persistidos."""
    try:
        return json.loads(registry_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"El registry existe, pero no es un JSON válido: {registry_path}"
        ) from exc


def _extract_registry_models(registry_payload: Any) -> list[dict[str, Any]]:
    """Soporta registry anterior tipo list y registry nuevo tipo dict."""
    if isinstance(registry_payload, dict):
        models = registry_payload.get("models", [])
    elif isinstance(registry_payload, list):
        models = registry_payload
    else:
        raise ValueError(
            "Formato de registry no soportado. "
            "Se esperaba una lista o un diccionario con la clave 'models'."
        )

    if not isinstance(models, list):
        raise ValueError("La clave 'models' del registry debe ser una lista.")

    return [m for m in models if isinstance(m, dict)]


def _validate_model_registry(model_dir: str | Path) -> tuple[Path, list[dict[str, Any]]]:
    """Valida que exista el registry y que apunte a modelos persistidos.

    Este wrapper NO entrena modelos.
    """
    normalized_model_dir = Path(model_dir)
    registry_path = normalized_model_dir / "model_registry.json"

    if not registry_path.exists():
        raise FileNotFoundError(
            f"No existe el registro de modelos persistidos: {registry_path}. "
            "Primero ejecuta la fase de modelamiento con persistencia habilitada."
        )

    registry_payload = _load_registry_payload(registry_path)
    models = _extract_registry_models(registry_payload)

    if not models:
        raise ValueError(
            f"El registry existe, pero no contiene modelos evaluables: {registry_path}"
        )

    missing_files: list[str] = []

    for model_info in models:
        model_path = model_info.get("model_path")
        if model_path and not Path(model_path).exists():
            missing_files.append(str(model_path))

    if missing_files:
        missing_preview = "\n".join(missing_files[:5])
        raise FileNotFoundError(
            "El registry referencia modelos que no existen en disco. "
            f"Primeros faltantes:\n{missing_preview}"
        )

    return registry_path, models


def _validate_inputs(
    df_new: pd.DataFrame,
    threshold: float,
) -> None:
    """Valida entradas mínimas para construir el reporte de evaluación."""
    if df_new is None or df_new.empty:
        raise ValueError("df_new no puede estar vacío para generar la evaluación de modelos.")

    if not 0 <= threshold <= 1:
        raise ValueError("threshold debe estar entre 0 y 1.")


def _write_report_manifest(
    evidence_dir: Path,
    output_path: Path,
    model_dir: str | Path,
    registry_path: Path,
    models: list[dict[str, Any]],
    plot_files: list[Path],
    threshold: float,
    df_new: pd.DataFrame,
) -> Path:
    """Guarda un manifiesto simple de trazabilidad del reporte HTML."""
    manifest_path = evidence_dir / "model_evaluation_report_manifest.json"

    manifest = {
        "created_at": datetime.utcnow().isoformat(),
        "report_type": "model_evaluation_html",
        "report_path": str(output_path),
        "model_dir": str(model_dir),
        "registry_path": str(registry_path),
        "threshold": float(threshold),
        "n_rows_evaluated": int(len(df_new)),
        "n_columns_evaluated": int(df_new.shape[1]),
        "n_models_in_registry": int(len(models)),
        "models": [
            {
                "target": model.get("target"),
                "model_type": model.get("model_type"),
                "model_name": model.get("model_name"),
                "model_path": model.get("model_path"),
            }
            for model in models
        ],
        "plot_files": [str(p) for p in plot_files],
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return manifest_path


def build_model_evaluation_report(
    df_new: pd.DataFrame,
    plot_files: list[Path] | None = None,
    output_path: str | Path = "output/html/report_model_evaluation.html",
    model_dir: str | Path = "output/models/statistical",
    evidence_dir: str | Path = "output/model_evaluation",
    threshold: float = 0.5,
    enabled: bool = False,
    evaluation_metrics: pd.DataFrame | None = None,
    detailed_results: dict[str, dict[str, Any]] | None = None,
) -> Path | None:
    """Construye el reporte HTML de evaluación posterior con datos nuevos.

    Mantiene compatibilidad:
    - conserva la función pública `build_model_evaluation_report`;
    - respeta `enabled=False`;
    - conserva parámetros existentes;
    - agrega `evaluation_metrics` y `detailed_results` como opcionales;
    - no reentrena modelos.

    Uso compatible anterior:
        build_model_evaluation_report(df_new, plot_files, enabled=True)

    Uso optimizado:
        metrics_df, detailed = evaluate_saved_models(...)
        plots = run_model_evaluation_plots(..., detailed_results=detailed)
        build_model_evaluation_report(
            df_new=df_new,
            plot_files=plots,
            evaluation_metrics=metrics_df,
            detailed_results=detailed,
            enabled=True,
        )
    """

    if not enabled:
        return None

    _validate_inputs(df_new=df_new, threshold=threshold)

    output_path = _ensure_parent_dir(output_path)
    evidence_dir = _ensure_dir(evidence_dir)

    plot_files = plot_files or []

    registry_path, models = _validate_model_registry(model_dir)

    exporter_kwargs: dict[str, Any] = {
        "df_new": df_new,
        "plot_files": plot_files,
        "output_path": output_path,
        "model_dir": model_dir,
        "evidence_dir": evidence_dir,
        "threshold": threshold,
    }

    # Compatible con el exporter mejorado.
    # Si el exporter local aún no soporta estos parámetros nuevos,
    # se hace fallback sin romper el flujo.
    try:
        exporter = ModelEvaluationExporter(
            **exporter_kwargs,
            evaluation_metrics=evaluation_metrics,
            detailed_results=detailed_results,
        )
    except TypeError:
        exporter = ModelEvaluationExporter(**exporter_kwargs)

    report_path = exporter.build_report()

    _write_report_manifest(
        evidence_dir=evidence_dir,
        output_path=Path(report_path),
        model_dir=model_dir,
        registry_path=registry_path,
        models=models,
        plot_files=plot_files,
        threshold=threshold,
        df_new=df_new,
    )

    return report_path
