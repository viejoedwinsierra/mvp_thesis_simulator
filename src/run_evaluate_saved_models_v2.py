from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from analysis_pipeline.config import AnalysisConfig
from analysis_pipeline.data_loader import load_datasets_from_folder
from analysis_pipeline.data_preparation import prepare_dataframe

# Evaluación estadística existente
from premodeling_modeling.model_evaluation import (
    evaluate_saved_models,
    export_evaluation_results,
)
from premodeling_modeling.model_evaluation_html_exports import build_model_evaluation_report
from premodeling_modeling.model_evaluation_plots import run_model_evaluation_plots
from premodeling_modeling.modeling_persistence import load_model_registry

# Nueva evaluación ML
from ml_modeling.evaluation import (
    discover_ml_models,
    evaluate_saved_ml_models,
    export_ml_evaluation_results,
)
from ml_modeling.evaluation_plots import run_ml_evaluation_plots
from ml_modeling.evaluation_reports import build_ml_evaluation_report


# ============================================================
# AUTO DATASET DISCOVERY PATCH
# ============================================================

def discover_dataset_dir(config):
    """
    Intenta descubrir automáticamente dónde quedaron los datasets finales.
    """

    dataset_dir = Path(config.dataset_dir)

    standard = list(dataset_dir.glob("*/blob_inventory_*.csv"))

    if standard:
        print(f"[DATASET] Estructura estándar detectada: {dataset_dir}")
        return dataset_dir

    recursive = list(dataset_dir.rglob("blob_inventory*.csv"))

    if recursive:
        print("[WARN] No se encontró estructura estándar.")
        print("[WARN] Se activó descubrimiento recursivo de datasets.")
        print(f"[DATASET] CSV encontrado: {recursive[0]}")

        return recursive[0].parent

    raise FileNotFoundError(
        f"No se encontraron datasets CSV dentro de: {dataset_dir}"
    )



def parse_args():
    parser = argparse.ArgumentParser(
        description="Evalúa modelos guardados estadísticos y/o ML sobre nuevos datasets."
    )

    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Carpeta de datasets. Por defecto usa AnalysisConfig.dataset_dir.",
    )

    parser.add_argument(
        "--family",
        choices=["statistical", "ml", "all"],
        default="ml",
        help="Familia de modelos a evaluar.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold para clasificación binaria.",
    )

    parser.add_argument(
        "--statistical-model-dir",
        default="output/models/statistical",
        help="Directorio de modelos estadísticos legacy.",
    )

    parser.add_argument(
        "--ml-model-root",
        default="output/modeling/ml",
        help="Raíz donde están los modelos ML por técnica.",
    )

    parser.add_argument(
        "--output-dir",
        default="output/model_evaluation_v2",
        help="Directorio raíz de evidencia de evaluación.",
    )

    return parser.parse_args()


def _extract_registry_models(registry):
    if isinstance(registry, dict):
        models = registry.get("models", [])
    elif isinstance(registry, list):
        models = registry
    else:
        raise ValueError("Formato de registry no soportado.")

    if not isinstance(models, list):
        raise ValueError("La colección de modelos del registry debe ser una lista.")

    return [model for model in models if isinstance(model, dict)]


def validate_statistical_inputs(model_dir: Path, threshold: float):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold debe estar entre 0 y 1.")

    registry_path = model_dir / "model_registry.json"

    if not registry_path.exists():
        raise FileNotFoundError(
            f"No existe el registry estadístico: {registry_path}. "
            "Ejecuta primero run_premodeling_modeling_legacy.py o el flujo estadístico compatible."
        )

    registry = load_model_registry(model_dir)
    models = _extract_registry_models(registry)

    if not models:
        raise ValueError(f"El registry no contiene modelos evaluables: {registry_path}")

    missing = []

    for model in models:
        model_path = model.get("model_path")
        if model_path and not Path(model_path).exists():
            missing.append(str(model_path))

    if missing:
        raise FileNotFoundError("Modelos faltantes:\n" + "\n".join(missing[:5]))

    return models


def evaluate_statistical_family(df_new, args, output_root: Path) -> dict:
    model_dir = Path(args.statistical_model_dir)
    evidence_dir = output_root / "statistical"
    plot_dir = evidence_dir / "plots"
    html_dir = evidence_dir / "html"

    validate_statistical_inputs(model_dir, args.threshold)

    metrics_df, detailed_results = evaluate_saved_models(
        df_new=df_new,
        model_dir=model_dir,
        threshold=args.threshold,
    )

    exported = export_evaluation_results(
        metrics_df=metrics_df,
        output_dir=evidence_dir,
        detailed_results=detailed_results,
    )

    plot_files = run_model_evaluation_plots(
        df_new=df_new,
        model_dir=model_dir,
        output_dir=plot_dir,
        threshold=args.threshold,
        enabled=True,
        detailed_results=detailed_results,
    )

    report_path = build_model_evaluation_report(
        df_new=df_new,
        plot_files=plot_files,
        output_path=html_dir / "report_statistical_model_evaluation.html",
        model_dir=model_dir,
        evidence_dir=evidence_dir,
        threshold=args.threshold,
        enabled=True,
        evaluation_metrics=metrics_df,
        detailed_results=detailed_results,
    )

    return {
        "family": "statistical",
        "metrics_rows": len(metrics_df),
        "report": str(report_path),
        "metrics_csv": exported.get("metrics_csv"),
        "manifest_json": exported.get("manifest_json"),
        "plots": len(plot_files),
    }


def evaluate_ml_family(df_new, args, output_root: Path) -> dict:
    model_root = Path(args.ml_model_root)
    evidence_dir = output_root / "ml"
    plot_dir = evidence_dir / "plots"
    html_dir = evidence_dir / "html"

    discover_ml_models(model_root)

    metrics_df, detailed_results = evaluate_saved_ml_models(
        df_new=df_new,
        modeling_root=model_root,
        threshold=args.threshold,
    )

    exported = export_ml_evaluation_results(
        metrics_df=metrics_df,
        detailed_results=detailed_results,
        output_dir=evidence_dir,
    )

    plot_files = run_ml_evaluation_plots(
        metrics_df=metrics_df,
        detailed_results=detailed_results,
        output_dir=plot_dir,
    )

    report_path = build_ml_evaluation_report(
        metrics_df=metrics_df,
        detailed_results=detailed_results,
        plot_files=plot_files,
        output_path=html_dir / "report_ml_model_evaluation.html",
        model_root=model_root,
        evidence_dir=evidence_dir,
        threshold=args.threshold,
    )

    return {
        "family": "ml",
        "metrics_rows": len(metrics_df),
        "report": str(report_path),
        "metrics_csv": exported.get("metrics_csv"),
        "manifest_json": exported.get("manifest_json"),
        "plots": len(plot_files),
    }


def main():
    args = parse_args()

    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold debe estar entre 0 y 1.")

    config = AnalysisConfig()
    config.ensure_dirs()

    dataset_dir = (
        Path(args.dataset_dir)
        if args.dataset_dir
        else discover_dataset_dir(config)
    )
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    df_new = load_datasets_from_folder(dataset_dir)
    df_new = prepare_dataframe(df_new)

    if df_new.empty:
        raise ValueError("Dataset preparado vacío.")

    print("Datos nuevos cargados y preparados")
    print(f"Filas evaluación: {len(df_new)}")
    print(f"Familia: {args.family}")
    print(f"Threshold: {args.threshold}")

    results = []

    if args.family in {"statistical", "all"}:
        print("[EVAL] Modelos estadísticos")
        results.append(evaluate_statistical_family(df_new, args, output_root))

    if args.family in {"ml", "all"}:
        print("[EVAL] Modelos ML")
        results.append(evaluate_ml_family(df_new, args, output_root))

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "output_root": str(output_root),
        "family": args.family,
        "threshold": args.threshold,
        "results": results,
    }

    manifest_path = output_root / "manifest_evaluate_saved_models_v2.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Evaluación completada sin reentrenar modelos")
    print(f"Manifest: {manifest_path}")

    for item in results:
        print(f"[{item['family']}] Reporte: {item['report']}")
        print(f"[{item['family']}] Métricas: {item['metrics_csv']}")
        print(f"[{item['family']}] Plots: {item['plots']}")


if __name__ == "__main__":
    main()
