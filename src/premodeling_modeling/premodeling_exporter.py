from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis_pipeline.html_report import build_html_page, plot_grid_html, to_html_table
from premodeling_modeling.premodeling_tables import (
    build_premodeling_datasets,
    build_premodeling_tables,
    export_premodeling_evidence,
)


class PremodelingExporter:
    """Genera reporte de preparación para modelamiento.

    No entrena modelos. Solo documenta datasets, variables, transformaciones
    y exporta evidencia estructurada.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        plot_files: list[Path] | None = None,
        output_path: str | Path = "output/html/report_premodeling.html",
        evidence_dir: str | Path = "output/premodeling",
        export_evidence: bool = True,
    ):
        self.df = df.copy()
        self.plot_files = plot_files or []
        self.output_path = Path(output_path)
        self.evidence_dir = Path(evidence_dir)
        self.export_evidence = export_evidence

    def build_tables(self) -> dict[str, pd.DataFrame]:
        return build_premodeling_tables(self.df)

    @staticmethod
    def _intro() -> str:
        return """
        <section class='card'>
            <h2>Objetivo del reporte de pre-modelamiento</h2>
            <p>
                Este reporte documenta la construcción de datasets por variable
                objetivo. No entrena modelos ni reporta métricas predictivas.
                Su finalidad es dejar evidencia reproducible de variables,
                transformaciones y estructura final de entrada para la fase
                de modelamiento estadístico.
            </p>
        </section>
        """

    def _plots_section(self) -> str:
        plots = [
            (p.stem.replace("_", " ").title(), p.name)
            for p in self.plot_files
            if p.suffix.lower() == ".png"
        ]

        if not plots:
            return ""

        return plot_grid_html(plots, relative_plot_dir="../plots/premodeling")

    def build_report(self) -> Path:
        if self.export_evidence:
            datasets = build_premodeling_datasets(self.df)
            export_premodeling_evidence(datasets, self.evidence_dir)

        tables = self.build_tables()

        body = self._intro()
        body += to_html_table(tables.get("variable_plan"), "1. Plan de variables por target")
        body += to_html_table(tables.get("dataset_summary"), "2. Resumen de datasets preparados")
        body += to_html_table(tables.get("feature_summary"), "3. Resumen de variables finales")
        body += self._plots_section()

        html = build_html_page("Reporte de pre-modelamiento - Blob Storage Monte Carlo", body)

        html = html.replace(
            "Fase descriptiva univariada para simulación Monte Carlo de Blob Storage.",
            "Fase de preparación de datos para modelamiento estadístico.",
        )

        html = html.replace(
            "Este reporte es estrictamente univariado: no contiene correlaciones,\n"
            "            regresiones, VIF, cruces entre variables, scatter plots, heatmaps ni análisis bivariado.\n"
            "            El conteo temporal de muestras se incluye únicamente como control de frecuencia.",
            "Este reporte documenta datasets preparados para modelamiento estadístico. "
            "No contiene entrenamiento de modelos ni métricas finales.",
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(html, encoding="utf-8")

        return self.output_path
