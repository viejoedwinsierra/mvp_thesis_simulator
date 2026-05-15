from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from analysis_pipeline.config import AnalysisConfig
from analysis_pipeline.data_loader import load_datasets_from_folder
from analysis_pipeline.data_preparation import prepare_dataframe

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta premodelado/modelamiento estadistico separado por tecnica."
    )

    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Carpeta de datasets. Por defecto usa AnalysisConfig.dataset_dir.",
    )

    parser.add_argument(
        "--output-dir",
        default="output/modeling/statistical",
        help="Carpeta raiz de salida por tecnica.",
    )

    parser.add_argument(
        "--techniques",
        nargs="*",
        default=DEFAULT_TECHNIQUES,
        help="Tecnicas a ejecutar: linear_simple linear_multiple log_log logistic_error",
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

    return parser.parse_args()


def run_selected_techniques(df, output_root: Path, args) -> list:
    results = []

    if "linear_simple" in args.techniques:
        result = run_linear_regression_simple(
            df=df,
            output_dir=output_root / "linear_regression_simple",
            target=args.target_regression,
            feature=args.simple_feature,
        )
        build_technique_report(result)
        results.append(result)

    if "linear_multiple" in args.techniques:
        result = run_linear_regression_multiple(
            df=df,
            output_dir=output_root / "linear_regression_multiple",
            target=args.target_regression,
        )
        build_technique_report(result)
        results.append(result)

    if "log_log" in args.techniques:
        result = run_log_log_regression(
            df=df,
            output_dir=output_root / "log_log_regression",
            target=args.target_regression,
        )
        build_technique_report(result)
        results.append(result)

    if "logistic_error" in args.techniques:
        result = run_logistic_regression_error(
            df=df,
            output_dir=output_root / "logistic_regression_error",
            target=args.target_classification,
        )
        build_technique_report(result)
        results.append(result)

    return results


def main():
    args = parse_args()

    config = AnalysisConfig()
    config.ensure_dirs()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else config.dataset_dir
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    df = load_datasets_from_folder(dataset_dir)
    df = prepare_dataframe(df)

    print("Datos cargados y preparados")
    print(f"Filas: {len(df)} | Columnas: {df.shape[1]}")
    print(f"Tecnicas: {', '.join(args.techniques)}")

    results = run_selected_techniques(df, output_root, args)

    master_report = build_master_modeling_report(results, output_root)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "output_root": str(output_root),
        "techniques": args.techniques,
        "total_models": len(results),
        "master_report": str(master_report),
        "results": [
            {
                "technique": r.technique,
                "target": r.target,
                "report": str(r.report_path) if r.report_path else None,
                "model": str(r.model_path) if r.model_path else None,
                "output_dir": str(r.output_dir),
            }
            for r in results
        ],
    }

    manifest_path = output_root / "manifest_modeling_by_technique.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Modelamiento por tecnica completado")
    print(f"Reporte maestro: {master_report}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
