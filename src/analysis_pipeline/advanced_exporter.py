from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis_pipeline.advanced_tables import build_advanced_tables
from analysis_pipeline.html_report import build_html_page, plot_grid_html, to_html_table


ALLOWED_ADVANCED_PLOT_PREFIXES = (
    "correlation_matrix_",
    "scatter_",
    "log_scatter_",
    "target_relationships_",
)


class AdvancedExporter:
    """Genera el informe avanzado multivariado HTML.

    Responsabilidad única:
    - Recibe dataframe preparado.
    - Recibe lista de gráficos avanzados ya generados.
    - Calcula tablas multivariadas.
    - Construye un único HTML avanzado.

    No exporta CSV.
    No mezcla salidas descriptivas.
    No calcula modelos predictivos, VIF ni inferencia causal.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        plot_files: list[Path] | None = None,
        output_path: str | Path = "output/html/report_advanced.html",
    ):
        self.df = df.copy()
        self.plot_files = plot_files or []
        self.output_path = Path(output_path)

    def build_tables(self) -> dict[str, pd.DataFrame]:
        return build_advanced_tables(self.df)

    @staticmethod
    def _is_valid_advanced_plot(path: Path) -> bool:
        return (
            path.suffix.lower() == ".png"
            and path.name.startswith(ALLOWED_ADVANCED_PLOT_PREFIXES)
        )

    def _filtered_plot_files(self) -> list[Path]:
        """Evita que el reporte avanzado renderice gráficos de otra fase.

        Aunque advanced_plots.py ya filtre, esta validación protege el HTML
        contra imágenes viejas o residuos en la carpeta output/plots/advanced.
        """
        return [
            Path(plot_file)
            for plot_file in self.plot_files
            if self._is_valid_advanced_plot(Path(plot_file))
        ]

    @staticmethod
    def _section_intro() -> str:
        return """
        <section class='card'>
            <h2>Objetivo del informe avanzado multivariado</h2>
            <p>
                Este informe documenta el análisis multivariado exploratorio previo
                a la fase posterior. Se enfoca en relaciones entre variables,
                redundancias, fuga de información, relación con variables objetivo,
                posibles relaciones no lineales y transformaciones sugeridas.
            </p>
            <p>
                No incluye modelos predictivos, regresiones, machine learning,
                VIF, optimización, tuning ni inferencia causal.
            </p>
        </section>
        """

    @staticmethod
    def _methodological_note() -> str:
        return """
        <section class='card'>
            <h2>Nota metodológica</h2>
            <p>
                Spearman se usa como referencia principal porque permite detectar
                relaciones monotónicas que no necesariamente son lineales. Pearson
                se conserva como referencia lineal para comparar diferencias entre
                relaciones lineales y monotónicas.
            </p>
            <p>
                Las matrices completas se ubican al final del reporte para no
                sobrecargar la lectura inicial. Las tablas priorizadas resumen los
                hallazgos más relevantes para revisión previa a la fase posterior.
            </p>
        </section>
        """

    def _plots_section(self) -> str:
        filtered_plots = self._filtered_plot_files()

        if not filtered_plots:
            return """
            <section class='card'>
                <h2>12. Gráficos multivariados exploratorios</h2>
                <p class='section-desc'>
                    No se encontraron gráficos multivariados válidos para renderizar.
                </p>
            </section>
            """

        plots = [
            (plot_file.stem.replace("_", " ").title(), plot_file.name)
            for plot_file in filtered_plots
        ]

        section = """
        <section class='card'>
            <h2>12. Gráficos multivariados exploratorios</h2>
            <p class='section-desc'>
                Incluye heatmaps de correlación, scatter plots de relaciones
                principales, relaciones con variables objetivo y vistas logarítmicas
                cuando aplican. No incluye histogramas, distribuciones univariadas
                ni boxplots descriptivos simples.
            </p>
        </section>
        """

        section += plot_grid_html(
            plots,
            relative_plot_dir="../plots/advanced",
        )

        return section

    def build_report(self) -> Path:
        tables = self.build_tables()
        body = ""

        body += self._section_intro()
        body += self._methodological_note()

        body += to_html_table(
            tables.get("executive_summary"),
            "1. Resumen ejecutivo",
        )

        body += to_html_table(
            tables.get("target_relationships"),
            "2. Relación con variables objetivo",
        )

        body += to_html_table(
            tables.get("top_relationships"),
            "3. Top relaciones multivariadas - Spearman",
        )

        body += to_html_table(
            tables.get("redundancy"),
            "4. Variables redundantes",
        )

        body += to_html_table(
            tables.get("derived_variables"),
            "5. Variables derivadas",
        )

        body += to_html_table(
            tables.get("leakage_and_exclusions"),
            "6. Fuga de información y exclusiones",
        )

        body += to_html_table(
            tables.get("nonlinear_relationships"),
            "7. Posibles relaciones no lineales",
        )

        body += to_html_table(
            tables.get("transformation_candidates"),
            "8. Candidatas a transformación",
        )

        body += to_html_table(
            tables.get("predictor_spearman_correlation"),
            "9. Matriz Spearman solo predictoras",
        )

        body += to_html_table(
            tables.get("spearman_correlation"),
            "10. Matriz de correlación Spearman general",
        )

        body += to_html_table(
            tables.get("pearson_correlation"),
            "11. Matriz de correlación Pearson general",
        )

        body += self._plots_section()

        html = build_html_page(
            "Reporte avanzado multivariado - Blob Storage Monte Carlo",
            body,
        )

        # Corrección defensiva:
        # Si build_html_page todavía conserva el subtítulo/nota del reporte
        # descriptivo, se reemplaza solo en este HTML avanzado.
        html = html.replace(
            "Fase descriptiva univariada para simulación Monte Carlo de Blob Storage.",
            "Fase avanzada multivariada exploratoria para simulación Monte Carlo de Blob Storage.",
        )

        html = html.replace(
            """
            Este reporte es estrictamente univariado: no contiene correlaciones,
            regresiones, VIF, cruces entre variables, scatter plots, heatmaps ni análisis bivariado.
            El conteo temporal de muestras se incluye únicamente como control de frecuencia.
            """,
            """
            Este reporte es estrictamente multivariado exploratorio: contiene
            correlaciones, relaciones con objetivos, redundancias, fuga de información
            y posibles relaciones no lineales. No contiene modelos predictivos,
            regresiones, VIF, optimización ni inferencia causal.
            """,
        )

        html = html.replace(
            "Gráficos univariados",
            "Gráficos multivariados exploratorios",
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(html, encoding="utf-8")

        return self.output_path
