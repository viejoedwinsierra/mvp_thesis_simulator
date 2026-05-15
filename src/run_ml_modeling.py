from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from analysis_pipeline.config import AnalysisConfig
from analysis_pipeline.data_loader import load_datasets_from_folder
from analysis_pipeline.data_preparation import prepare_dataframe

from ml_modeling.models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    run_classification_model,
    run_regression_model,
)
from ml_modeling.reports import build_ml_master_report, build_ml_model_report


DEFAULT_REGRESSION_TECHNIQUES = [
    "random_forest_regressor",
    "gradient_boosting_regressor",
    "hist_gradient_boosting_regressor",
]

DEFAULT_CLASSIFICATION_TECHNIQUES = [
    "random_forest_classifier",
    "gradient_boosting_classifier",
    "hist_gradient_boosting_classifier",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ejecuta modelos de Machine Learning sobre el universo simulado."
    )

    parser.add_argument(
        "--dataset-dir",
        default=None,
        help="Carpeta de datasets. Por defecto usa AnalysisConfig.dataset_dir.",
    )

    parser.add_argument(
        "--output-dir",
        default="output/modeling/ml",
        help="Carpeta raiz de salida para modelos ML.",
    )

    parser.add_argument(
        "--target-regression",
        default="storage_cost",
        help="Target para modelos de regresion.",
    )

    parser.add_argument(
        "--target-classification",
        default="has_error",
        help="Target para modelos de clasificacion.",
    )

    parser.add_argument(
        "--regression-techniques",
        nargs="*",
        default=DEFAULT_REGRESSION_TECHNIQUES,
        help=f"Modelos de regresion disponibles: {', '.join(REGRESSION_MODELS.keys())}",
    )

    parser.add_argument(
        "--classification-techniques",
        nargs="*",
        default=DEFAULT_CLASSIFICATION_TECHNIQUES,
        help=f"Modelos de clasificacion disponibles: {', '.join(CLASSIFICATION_MODELS.keys())}",
    )

    parser.add_argument(
        "--include-sgd",
        action="store_true",
        help="Incluye SGDRegressor y SGDClassifier como aproximacion incremental.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.include_sgd:
        if "sgd_regressor" not in args.regression_techniques:
            args.regression_techniques.append("sgd_regressor")
        if "sgd_classifier" not in args.classification_techniques:
            args.classification_techniques.append("sgd_classifier")

    config = AnalysisConfig()
    config.ensure_dirs()

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else config.dataset_dir
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    df = load_datasets_from_folder(dataset_dir)
    df = prepare_dataframe(df)

    print("Datos cargados y preparados para ML")
    print(f"Filas: {len(df)} | Columnas: {df.shape[1]}")
    print(f"Regresion: {', '.join(args.regression_techniques)}")
    print(f"Clasificacion: {', '.join(args.classification_techniques)}")

    results = []

    for technique in args.regression_techniques:
        print(f"[ML][REG] {technique}")
        result = run_regression_model(
            df=df,
            output_dir=output_root / technique,
            technique=technique,
            target=args.target_regression,
        )
        build_ml_model_report(result)
        results.append(result)

    for technique in args.classification_techniques:
        print(f"[ML][CLF] {technique}")
        result = run_classification_model(
            df=df,
            output_dir=output_root / technique,
            technique=technique,
            target=args.target_classification,
        )
        build_ml_model_report(result)
        results.append(result)

    master_report = build_ml_master_report(results, output_root)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(dataset_dir),
        "output_root": str(output_root),
        "target_regression": args.target_regression,
        "target_classification": args.target_classification,
        "total_models": len(results),
        "master_report": str(master_report),
        "results": [
            {
                "technique": r.technique,
                "target": r.target,
                "task_type": r.task_type,
                "report": str(r.report_path) if r.report_path else None,
                "model": str(r.model_path) if r.model_path else None,
                "output_dir": str(r.output_dir),
            }
            for r in results
        ],
    }

    manifest_path = output_root / "manifest_ml_modeling.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Modelamiento ML completado")
    print(f"Reporte maestro ML: {master_report}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
