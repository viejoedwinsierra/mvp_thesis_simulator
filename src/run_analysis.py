from __future__ import annotations

import argparse
from pathlib import Path

from analysis_pipeline.config import AnalysisConfig
from analysis_pipeline.data_loader import load_datasets_from_folder
from analysis_pipeline.data_preparation import prepare_dataframe
from analysis_pipeline.data_dictionary import build_data_dictionary_table
from analysis_pipeline.descriptive_tables import run_descriptive_analysis
from analysis_pipeline.descriptive_plots import generate_descriptive_plots
from analysis_pipeline.html_report import build_descriptive_report

from analysis_pipeline.advanced_plots import run_advanced_plots
from analysis_pipeline.advanced_html_exports import build_advanced_report


# ==========================================================
# CONFIGURACIÓN DE EJECUCIÓN
# ==========================================================

RUN_ADVANCED_DEFAULT = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ejecuta análisis descriptivo y, opcionalmente, análisis avanzado multivariado."
    )

    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Activa la generación del análisis avanzado multivariado.",
    )

    parser.add_argument(
        "--no-advanced",
        action="store_true",
        help="Desactiva la generación del análisis avanzado multivariado.",
    )

    return parser.parse_args()


def should_run_advanced(args: argparse.Namespace) -> bool:
    if args.no_advanced:
        return False

    if args.advanced:
        return True

    return RUN_ADVANCED_DEFAULT


def run_descriptive_stage(df, config: AnalysisConfig):
    """Ejecuta la fase descriptiva existente sin mezclar lógica avanzada."""
    data_dictionary_df = build_data_dictionary_table(df)
    descriptive_results = run_descriptive_analysis(df)

    descriptive_plots = generate_descriptive_plots(
        df=df,
        output_dir=config.descriptive_plot_dir,
    )

    descriptive_report = build_descriptive_report(
        df=df,
        results=descriptive_results,
        data_dictionary_df=data_dictionary_df,
        key_tables=descriptive_results,
        plots=descriptive_plots,
        output_path=config.html_dir / config.report_name,
    )

    return descriptive_report, descriptive_plots, descriptive_results


def run_advanced_stage(df, config: AnalysisConfig) -> Path | None:
    """Ejecuta la fase avanzada multivariada.

    Esta fase:
    - No calcula modelos predictivos.
    - No calcula regresiones.
    - No calcula VIF.
    - No ejecuta inferencia causal.
    - Solo genera análisis multivariado exploratorio previo al modelado.
    """
    advanced_plot_files = run_advanced_plots(
        df=df,
        output_dir=config.advanced_plot_dir,
        enabled=True,
    )

    advanced_report = build_advanced_report(
        df=df,
        plot_files=advanced_plot_files,
        output_path=config.html_dir / config.advanced_report_name,
        enabled=True,
    )

    return advanced_report


def main():
    args = parse_args()
    run_advanced = should_run_advanced(args)

    config = AnalysisConfig()
    config.ensure_dirs()

    # ======================================================
    # 1. CARGA Y PREPARACIÓN
    # ======================================================
    df = load_datasets_from_folder(config.dataset_dir)
    df = prepare_dataframe(df)

    # ======================================================
    # 2. FASE DESCRIPTIVA
    # ======================================================
    descriptive_report, _, _ = run_descriptive_stage(df, config)

    # ======================================================
    # 3. FASE AVANZADA MULTIVARIADA
    # ======================================================
    advanced_report = None

    if run_advanced:
        advanced_report = run_advanced_stage(df, config)

    # ======================================================
    # 4. LOG FINAL
    # ======================================================
    print("\n✅ Análisis completado")
    print(f"✅ Reporte descriptivo: {descriptive_report}")
    print(f"✅ Gráficos descriptivos: {config.descriptive_plot_dir}")

    if advanced_report:
        print(f"✅ Reporte avanzado multivariado: {advanced_report}")
        print(f"✅ Gráficos avanzados: {config.advanced_plot_dir}")
    else:
        print("ℹ️ Reporte avanzado multivariado no generado.")


if __name__ == "__main__":
    main()
