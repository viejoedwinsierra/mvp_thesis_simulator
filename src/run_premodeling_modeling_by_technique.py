"""Runner unificado de premodeling y modeling estadistico.

Objetivo:
- Mantener un solo punto de entrada para iteraciones con datasets generados automaticamente.
- Permitir premodeling opcional.
- Permitir modeling por tecnica.
- Mantener trazabilidad por experimento mediante manifest.
- Evitar duplicidad entre runners anteriores.

Uso sugerido:
    python run_premodeling_modeling_unified.py \
        --dataset-dir output/datasets/01_base_controlada \
        --output-dir output/experiments/exp_001_base \
        --techniques linear_simple linear_multiple log_log logistic_error \
        --target-regression storage_cost \
        --simple-feature size_mb \
        --target-classification has_error \
        --run-premodeling \
        --run-modeling
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from analysis_pipeline.config import AnalysisConfig
from analysis_pipeline.data_loader import load_datasets_from_folder
from analysis_pipeline.data_preparation import prepare_dataframe

from premodeling_modeling.premodeling_plots import run_premodeling_plots
from premodeling_modeling.premodeling_html_exports import build_premodeling_report

from premodeling_modeling.linear_regression_simple import run_linear_regression_simple
from premodeling_modeling.linear_regression_multiple import run_linear_regression_multiple
from premodeling_modeling.log_log_regression import run_log_log_regression
from premodeling_modeling.logistic_regression_error import run_logistic_regression_error
from premodeling_modeling.technique_reports import (
    build_master_modeling_report,
    build_technique_report,
)


DEFAULT_TECHNIQUES = [
    "linear_simple",
    "linear_multiple",
    "log_log",
    "logistic_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta premodeling y modeling estadistico en un flujo unificado "
            "para iteraciones con multiples datasets."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Carpeta de datasets. Por defecto usa AnalysisConfig.dataset_dir.",
    )

    parser.add_argument(
        "--output-dir",
        default="output/experiments/default_experiment",
        help="Carpeta raiz de salida del experimento.",
    )

    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Nombre logico del experimento. Si no se envia, usa el nombre de output-dir.",
    )

    parser.add_argument(
        "--run-premodeling",
        action="store_true",
        help="Ejecuta analisis exploratorio/premodeling.",
    )

    parser.add_argument(
        "--run-modeling",
        action="store_true",
        help="Ejecuta modeling por tecnica.",
    )

    parser.add_argument(
        "--techniques",
        nargs="*",
        default=DEFAULT_TECHNIQUES,
        help=(
            "Tecnicas a ejecutar: "
            "linear_simple linear_multiple log_log logistic_error"
        ),
    )

    parser.add_argument(
        "--target-regression",
        default="storage_cost",
        help="Target para regresiones lineales.",
    )

    parser.add_argument(
        "--simple-feature",
        default="size_mb",
        help="Variable explicativa para regresion lineal simple.",
    )

    parser.add_argument(
        "--target-classification",
        default="has_error",
        help="Target para regresion logistica.",
    )

    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Carga el dataset sin aplicar prepare_dataframe.",
    )

    return parser.parse_args()


def resolve_paths(args: argparse.Namespace, config: AnalysisConfig) -> dict[str, Path]:
    output_root = Path(args.output_dir)
    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else config.dataset_dir

    return {
        "dataset_dir": dataset_dir,
        "output_root": output_root,
        "premodeling_root": output_root / "premodeling",
        "modeling_root": output_root / "modeling" / "statistical",
        "manifest_path": output_root / "manifest_experiment.json",
    }


def ensure_experiment_dirs(paths: dict[str, Path]) -> None:
    paths["output_root"].mkdir(parents=True, exist_ok=True)
    paths["premodeling_root"].mkdir(parents=True, exist_ok=True)
    paths["modeling_root"].mkdir(parents=True, exist_ok=True)


def load_and_prepare_dataset(
    dataset_dir: Path,
    skip_prepare: bool = False,
):
    df = load_datasets_from_folder(dataset_dir)

    if skip_prepare:
        return df

    return prepare_dataframe(df)


def run_premodeling_stage(
    df,
    output_dir: Path,
    enabled: bool,
) -> dict[str, Any] | None:
    if not enabled:
        return None

    plots_dir = output_dir / "plots"
    evidence_dir = output_dir / "evidence"
    report_path = output_dir / "report_premodeling.html"

    plots = run_premodeling_plots(
        df=df,
        output_dir=plots_dir,
        enabled=True,
    )

    report = build_premodeling_report(
        df=df,
        plot_files=plots,
        output_path=report_path,
        evidence_dir=str(evidence_dir),
        enabled=True,
    )

    return {
        "report": str(report),
        "plots": [str(path) for path in plots],
        "plots_count": len(plots),
        "output_dir": str(output_dir),
    }


def run_selected_modeling_techniques(
    df,
    output_root: Path,
    args: argparse.Namespace,
) -> list:
    results = []

    selected = set(args.techniques)

    if "linear_simple" in selected:
        result = run_linear_regression_simple(
            df=df,
            output_dir=output_root / "linear_regression_simple",
            target=args.target_regression,
            feature=args.simple_feature,
        )
        build_technique_report(result)
        results.append(result)

    if "linear_multiple" in selected:
        result = run_linear_regression_multiple(
            df=df,
            output_dir=output_root / "linear_regression_multiple",
            target=args.target_regression,
        )
        build_technique_report(result)
        results.append(result)

    if "log_log" in selected:
        result = run_log_log_regression(
            df=df,
            output_dir=output_root / "log_log_regression",
            target=args.target_regression,
        )
        build_technique_report(result)
        results.append(result)

    if "logistic_error" in selected:
        result = run_logistic_regression_error(
            df=df,
            output_dir=output_root / "logistic_regression_error",
            target=args.target_classification,
        )
        build_technique_report(result)
        results.append(result)

    unknown = sorted(selected - set(DEFAULT_TECHNIQUES))
    if unknown:
        raise ValueError(
            "Tecnicas no soportadas: "
            + ", ".join(unknown)
            + ". Tecnicas validas: "
            + ", ".join(DEFAULT_TECHNIQUES)
        )

    return results


def run_modeling_stage(
    df,
    output_dir: Path,
    args: argparse.Namespace,
    enabled: bool,
) -> dict[str, Any] | None:
    if not enabled:
        return None

    results = run_selected_modeling_techniques(
        df=df,
        output_root=output_dir,
        args=args,
    )

    master_report = build_master_modeling_report(results, output_dir)

    return {
        "master_report": str(master_report),
        "total_models": len(results),
        "techniques": [
            {
                "technique": result.technique,
                "target": result.target,
                "report": str(result.report_path) if result.report_path else None,
                "model": str(result.model_path) if result.model_path else None,
                "output_dir": str(result.output_dir),
            }
            for result in results
        ],
        "output_dir": str(output_dir),
    }


def build_manifest(
    args: argparse.Namespace,
    paths: dict[str, Path],
    df,
    premodeling_outputs: dict[str, Any] | None,
    modeling_outputs: dict[str, Any] | None,
) -> dict[str, Any]:
    experiment_name = args.experiment_name or paths["output_root"].name

    return {
        "experiment_name": experiment_name,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(paths["dataset_dir"]),
        "output_root": str(paths["output_root"]),
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "run_premodeling": bool(args.run_premodeling),
        "run_modeling": bool(args.run_modeling),
        "skip_prepare": bool(args.skip_prepare),
        "modeling_parameters": {
            "techniques": args.techniques,
            "target_regression": args.target_regression,
            "simple_feature": args.simple_feature,
            "target_classification": args.target_classification,
        },
        "premodeling": premodeling_outputs,
        "modeling": modeling_outputs,
    }


def write_manifest(manifest: dict[str, Any], manifest_path: Path) -> Path:
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return manifest_path


def main() -> None:
    args = parse_args()

    if not args.run_premodeling and not args.run_modeling:
        args.run_modeling = True

    config = AnalysisConfig()
    config.ensure_dirs()

    paths = resolve_paths(args, config)
    ensure_experiment_dirs(paths)

    df = load_and_prepare_dataset(
        dataset_dir=paths["dataset_dir"],
        skip_prepare=args.skip_prepare,
    )

    print("Datos cargados y preparados")
    print(f"Dataset dir: {paths['dataset_dir']}")
    print(f"Output dir: {paths['output_root']}")
    print(f"Filas: {len(df)} | Columnas: {df.shape[1]}")

    premodeling_outputs = run_premodeling_stage(
        df=df,
        output_dir=paths["premodeling_root"],
        enabled=args.run_premodeling,
    )

    if premodeling_outputs is not None:
        print(f"Reporte premodeling: {premodeling_outputs['report']}")

    modeling_outputs = run_modeling_stage(
        df=df,
        output_dir=paths["modeling_root"],
        args=args,
        enabled=args.run_modeling,
    )

    if modeling_outputs is not None:
        print(f"Reporte maestro modeling: {modeling_outputs['master_report']}")
        print(f"Modelos generados: {modeling_outputs['total_models']}")

    manifest = build_manifest(
        args=args,
        paths=paths,
        df=df,
        premodeling_outputs=premodeling_outputs,
        modeling_outputs=modeling_outputs,
    )

    manifest_path = write_manifest(manifest, paths["manifest_path"])

    print(f"Manifest del experimento: {manifest_path}")
    print("Proceso unificado premodeling/modeling completado")


if __name__ == "__main__":
    main()
