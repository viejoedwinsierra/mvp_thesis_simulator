"""Runner de premodeling y modeling estadístico.

Orquesta el flujo completo manteniendo compatibilidad:
- premodeling opcional;
- modeling opcional;
- un solo entrenamiento estadístico por ejecución;
- plots, HTML y persistencia reutilizan el mismo `model_results`;
- no rompe comportamiento de funciones existentes.
"""

from __future__ import annotations

from pathlib import Path

from analysis_pipeline.config import AnalysisConfig
from analysis_pipeline.data_loader import load_datasets_from_folder
from analysis_pipeline.data_preparation import prepare_dataframe

from premodeling_modeling.premodeling_plots import run_premodeling_plots
from premodeling_modeling.premodeling_html_exports import build_premodeling_report

from premodeling_modeling.statistical_models import fit_statistical_models
from premodeling_modeling.modeling_plots import run_modeling_plots
from premodeling_modeling.modeling_html_exports import build_modeling_report
from premodeling_modeling.modeling_persistence import train_and_save_statistical_models


RUN_PREMODELING = False
RUN_MODELING = True

# Persistencia desactivada por defecto para mantener el comportamiento anterior.
SAVE_MODELING_MODELS = True


def run_premodeling_stage(df, config):
    premodeling_plots = run_premodeling_plots(
        df=df,
        output_dir=getattr(config, "premodeling_plot_dir", "output/plots/premodeling"),
        enabled=True,
    )

    return build_premodeling_report(
        df=df,
        plot_files=premodeling_plots,
        output_path=config.html_dir / "report_premodeling.html",
        evidence_dir="output/premodeling",
        enabled=True,
    )


def run_modeling_stage(
    df,
    config,
    save_models: bool = SAVE_MODELING_MODELS,
):
    """Ejecuta modeling evitando reentrenamiento duplicado.

    Flujo correcto:
    1. Entrenar una sola vez con `fit_statistical_models`.
    2. Reutilizar `model_results` en plots.
    3. Reutilizar `model_results` en HTML.
    4. Guardar modelos, si `save_models=True`, usando los mismos resultados.
    """

    # Fuente única de entrenamiento para toda la fase de modeling.
    model_results = fit_statistical_models(df)

    modeling_plots = run_modeling_plots(
        df=df,
        output_dir=getattr(config, "modeling_plot_dir", "output/plots/modeling"),
        enabled=True,
        model_results=model_results,
    )

    modeling_report = build_modeling_report(
        df=df,
        plot_files=modeling_plots,
        output_path=config.html_dir / "report_modeling.html",
        enabled=True,
        model_results=model_results,
    )

    exported_models = None

    if save_models:
        model_output_dir = getattr(
            config,
            "model_output_dir",
            Path("output/models/statistical"),
        )

        exported_models = train_and_save_statistical_models(
            df=df,
            output_dir=model_output_dir,
            model_results=model_results,
        )

    return {
        "report": modeling_report,
        "plots": modeling_plots,
        "models": exported_models,
        "model_results": model_results,
    }


def main():
    config = AnalysisConfig()
    config.ensure_dirs()

    df = load_datasets_from_folder(config.dataset_dir)
    df = prepare_dataframe(df)

    print("✅ Datos cargados y preparados")

    if RUN_PREMODELING:
        premodeling_report = run_premodeling_stage(df, config)
        print(f"✅ Reporte premodeling: {premodeling_report}")

    if RUN_MODELING:
        modeling_outputs = run_modeling_stage(
            df=df,
            config=config,
            save_models=SAVE_MODELING_MODELS,
        )

        print(f"✅ Reporte modeling: {modeling_outputs['report']}")
        print(f"✅ Gráficos modeling generados: {len(modeling_outputs['plots'])}")

        if modeling_outputs["models"] is not None:
            print(f"✅ Modelos guardados: {modeling_outputs['models'].get('registry')}")

    print("✅ Proceso premodeling/modeling completado")


if __name__ == "__main__":
    main()
