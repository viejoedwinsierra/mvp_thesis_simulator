from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import pandas as pd

from analysis_pipeline.html_report import build_html_page, plot_grid_html, to_html_table
from premodeling_modeling.modeling_tables import build_modeling_tables


class ModelingExporter:
    """Reporte final de modelos estadísticos interpretables.

    Mejora clave:
    - Permite reutilizar `model_results` ya entrenados.
    - Evita reentrenamiento implícito en tablas.
    - Mantiene compatibilidad total con implementación anterior.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        plot_files: list[Path] | None = None,
        output_path: str | Path = "output/html/report_modeling.html",
        model_results: Dict[str, Dict[str, Any]] | None = None,
    ):
        self.df = df.copy()
        self.plot_files = plot_files or []
        self.output_path = Path(output_path)
        self.model_results = model_results

    def build_tables(self) -> dict[str, pd.DataFrame]:
        return build_modeling_tables(
            self.df,
            model_results=self.model_results,
        )

    @staticmethod
    def _intro() -> str:
        return """
        <section class='card'>
            <h2>Objetivo del reporte de modelamiento estadístico</h2>
            <p>
                Este reporte presenta modelos estadísticos interpretables para los
                objetivos definidos: duración de transferencia, costo de almacenamiento
                y ocurrencia de error. No incluye algoritmos de machine learning
                complejos ni procesos de tuning.
            </p>
        </section>
        """

    def _plots_section(self) -> str:
        plots = [
            (p.stem.replace("_", " ").title(), p.name)
            for p in sorted(self.plot_files)
            if p.suffix.lower() == ".png"
        ]

        if not plots:
            return ""

        return plot_grid_html(plots, relative_plot_dir="../plots/modeling")

    def build_report(self) -> Path:
        tables = self.build_tables()

        body = self._intro()
        body += to_html_table(tables.get("model_summary"), "1. Resumen de modelos")
        body += to_html_table(tables.get("model_coefficients"), "2. Coeficientes de modelos")
        body += self._plots_section()

        html = build_html_page(
            "Reporte de modelamiento estadístico - Blob Storage Monte Carlo",
            body,
        )

        html = html.replace(
            "Fase descriptiva univariada para simulación Monte Carlo de Blob Storage.",
            "Fase de modelamiento estadístico para simulación Monte Carlo de Blob Storage.",
        )

        html = html.replace(
            "Este reporte es estrictamente univariado: no contiene correlaciones,\n"
            "            regresiones, VIF, cruces entre variables, scatter plots, heatmaps ni análisis bivariado.\n"
            "            El conteo temporal de muestras se incluye únicamente como control de frecuencia.",
            "Este reporte contiene modelos estadísticos interpretables. "
            "No incluye machine learning complejo, optimización ni tuning.",
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(html, encoding="utf-8")

        return self.output_path
