"""Runner para evaluar modelos guardados sobre el dataset actual.

Uso:
    python src/run_evaluate_saved_models.py

Flujo recomendado:
    1. Entrenar con dataset A:
       python src/run_train_statistical_models.py
       o ejecutar modeling con persistencia habilitada.

    2. Regenerar dataset B con run_simulation.py
       usando la misma configuración experimental.

    3. Evaluar modelos guardados:
       python src/run_evaluate_saved_models.py

Este runner NO reentrena modelos.
Solo carga modelos persistidos desde `MODEL_DIR`, evalúa sobre datos nuevos,
genera métricas, gráficos estadísticos, evidencia tabular y reporte HTML.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from analysis_pipeline.config import AnalysisConfig
from analysis_pipeline.data_loader import load_datasets_from_folder
from analysis_pipeline.data_preparation import prepare_dataframe

from premodeling_modeling.model_evaluation import (
    evaluate_saved_models,
    export_evaluation_results,
)
from premodeling_modeling.model_evaluation_html_exports import build_model_evaluation_report
from premodeling_modeling.model_evaluation_plots import run_model_evaluation_plots
from premodeling_modeling.modeling_persistence import load_model_registry


THRESHOLD = 0.5

MODEL_DIR = Path("output/models/statistical")
EVALUATION_PLOT_DIR = Path("output/plots/model_evaluation")
EVIDENCE_DIR = Path("output/model_evaluation")


def _extract_registry_models(registry: Any) -> list[dict[str, Any]]:
    """Soporta registry viejo tipo list y registry nuevo tipo dict."""
    if isinstance(registry, dict):
        models = registry.get("models", [])
    elif isinstance(registry, list):
        models = registry
    else:
        raise ValueError(
            "Formato de registry no soportado. "
            "Se esperaba una lista o un diccionario con clave 'models'."
        )

    if not isinstance(models, list):
        raise ValueError("La colección de modelos del registry debe ser una lista.")

    return [model for model in models if isinstance(model, dict)]


def validate_evaluation_inputs(model_dir: Path, threshold: float) -> list[dict[str, Any]]:
    """Valida condiciones mínimas para evaluar modelos guardados.

    No entrena modelos. Solo verifica que exista la persistencia requerida.
    """

    if not 0 <= threshold <= 1:
        raise ValueError("THRESHOLD debe estar entre 0 y 1.")

    registry_path = model_dir / "model_registry.json"

    if not registry_path.exists():
        raise FileNotFoundError(
            f"No existe el registry de modelos: {registry_path}. "
            "Primero debes entrenar y guardar modelos estadísticos."
        )

    registry = load_model_registry(model_dir)
    models = _extract_registry_models(registry)

    if not models:
        raise ValueError(
            f"El registry existe, pero no contiene modelos evaluables: {registry_path}"
        )

    missing_models: list[str] = []

    for model in models:
        model_path = model.get("model_path")
        if model_path and not Path(model_path).exists():
            missing_models.append(str(model_path))

    if missing_models:
        preview = "\n".join(missing_models[:5])
        raise FileNotFoundError(
            "El registry referencia modelos que no existen en disco. "
            f"Primeros faltantes:\n{preview}"
        )

    return models


def main():
    config = AnalysisConfig()
    config.ensure_dirs()

    models = validate_evaluation_inputs(
        model_dir=MODEL_DIR,
        threshold=THRESHOLD,
    )

    df_new = load_datasets_from_folder(config.dataset_dir)
    df_new = prepare_dataframe(df_new)

    if df_new.empty:
        raise ValueError("El dataset preparado para evaluación está vacío.")

    print("✅ Datos nuevos cargados y preparados para evaluación")
    print(f"✅ Filas para evaluación: {len(df_new)}")
    print(f"✅ Modelos persistidos encontrados: {len(models)}")
    print(f"✅ Directorio de modelos: {MODEL_DIR}")

    # Evaluación única: métricas + detalle reutilizable para plots y HTML.
    metrics_df, detailed_results = evaluate_saved_models(
        df_new=df_new,
        model_dir=MODEL_DIR,
        threshold=THRESHOLD,
    )

    exported = export_evaluation_results(
        metrics_df=metrics_df,
        output_dir=EVIDENCE_DIR,
        detailed_results=detailed_results,
    )

    plot_files = run_model_evaluation_plots(
        df_new=df_new,
        model_dir=MODEL_DIR,
        output_dir=EVALUATION_PLOT_DIR,
        threshold=THRESHOLD,
        enabled=True,
        detailed_results=detailed_results,
    )

    report_path = build_model_evaluation_report(
        df_new=df_new,
        plot_files=plot_files,
        output_path=config.html_dir / "report_model_evaluation.html",
        model_dir=MODEL_DIR,
        evidence_dir=EVIDENCE_DIR,
        threshold=THRESHOLD,
        enabled=True,
        evaluation_metrics=metrics_df,
        detailed_results=detailed_results,
    )

    n_ok = int((metrics_df.get("status") == "ok").sum()) if "status" in metrics_df else 0
    n_error = int((metrics_df.get("status") == "error").sum()) if "status" in metrics_df else 0

    print(f"✅ Modelos evaluados correctamente: {n_ok}")
    print(f"⚠️ Modelos con error: {n_error}")
    print(f"✅ Gráficos de evaluación generados: {len(plot_files)}")
    print(f"✅ Reporte de evaluación generado: {report_path}")
    print(f"✅ Métricas exportadas: {exported.get('metrics_csv')}")
    print(f"✅ Manifiesto exportado: {exported.get('manifest_json')}")
    print("✅ Evaluación completada sin reentrenar modelos")


if __name__ == "__main__":
    main()
