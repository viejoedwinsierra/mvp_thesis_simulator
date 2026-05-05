from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from analysis_pipeline.html_report import build_html_page, plot_grid_html, to_html_table
from premodeling_modeling.model_evaluation import (
    build_evaluation_comparison_table,
    build_evaluation_status_table,
    evaluate_saved_models,
    export_evaluation_results,
)


class ModelEvaluationExporter:
    """Reporte de evaluación de modelos guardados.

    Esta fase NO entrena modelos. Carga modelos entrenados previamente y los
    evalúa contra el dataset disponible.

    Mantiene compatibilidad:
    - conserva los parámetros originales;
    - agrega `evaluation_metrics` y `detailed_results` como opcionales;
    - evita recalcular evaluación cuando ya se tienen resultados;
    - exporta evidencia tabular para trazabilidad.
    """

    def __init__(
        self,
        df_new: pd.DataFrame,
        plot_files: list[Path] | None = None,
        output_path: str | Path = "output/html/report_model_evaluation.html",
        model_dir: str | Path = "output/models/statistical",
        evidence_dir: str | Path = "output/model_evaluation",
        threshold: float = 0.5,
        evaluation_metrics: pd.DataFrame | None = None,
        detailed_results: dict[str, dict[str, Any]] | None = None,
    ):
        self.df_new = df_new.copy()
        self.plot_files = plot_files or []
        self.output_path = Path(output_path)
        self.model_dir = Path(model_dir)
        self.evidence_dir = Path(evidence_dir)
        self.threshold = threshold
        self.evaluation_metrics = evaluation_metrics
        self.detailed_results = detailed_results

    def _get_evaluation_results(self) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        """Obtiene resultados de evaluación sin duplicar cálculos si ya existen."""
        if self.evaluation_metrics is not None:
            metrics_df = self.evaluation_metrics.copy()
            detailed = self.detailed_results or {}
            return metrics_df, detailed

        metrics_df, detailed = evaluate_saved_models(
            df_new=self.df_new,
            model_dir=self.model_dir,
            threshold=self.threshold,
        )

        self.evaluation_metrics = metrics_df
        self.detailed_results = detailed

        return metrics_df, detailed

    def build_tables(self) -> dict[str, pd.DataFrame]:
        metrics_df, detailed = self._get_evaluation_results()

        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        export_evaluation_results(
            metrics_df=metrics_df,
            output_dir=self.evidence_dir,
            detailed_results=detailed,
        )

        return {
            "evaluation_metrics": metrics_df,
            "evaluation_comparison": build_evaluation_comparison_table(metrics_df),
            "evaluation_status": build_evaluation_status_table(metrics_df),
        }

    @staticmethod
    def _intro() -> str:
        return """
        <section class='card'>
            <h2>Objetivo del reporte de evaluación</h2>
            <p>
                Este reporte evalúa modelos estadísticos previamente entrenados
                sobre un dataset nuevo o regenerado. A diferencia del reporte de
                modelamiento, esta fase no reentrena; solo mide desempeño.
            </p>
            <p>
                Su objetivo es validar generalización, estabilidad, degradación de
                métricas y utilidad del modelo sobre escenarios no usados durante
                el entrenamiento.
            </p>
        </section>
        """

    def _evaluation_context(self, tables: dict[str, pd.DataFrame]) -> str:
        metrics_df = tables.get("evaluation_metrics", pd.DataFrame())
        status_df = tables.get("evaluation_status", pd.DataFrame())

        n_models = int(len(metrics_df)) if metrics_df is not None else 0
        n_ok = int((metrics_df.get("status") == "ok").sum()) if not metrics_df.empty and "status" in metrics_df else 0
        n_error = int((metrics_df.get("status") == "error").sum()) if not metrics_df.empty and "status" in metrics_df else 0

        recommendations = ""

        if not status_df.empty and "recommendation" in status_df:
            counts = status_df["recommendation"].value_counts(dropna=False)
            recommendations = "<ul>"
            for name, count in counts.items():
                recommendations += f"<li><strong>{name}</strong>: {count}</li>"
            recommendations += "</ul>"

        return f"""
        <section class='card summary-card'>
            <h2>Resumen ejecutivo de evaluación</h2>
            <p>
                Se evaluaron <strong>{n_models}</strong> modelos guardados sobre
                el dataset actual, usando umbral de clasificación
                <strong>{self.threshold}</strong>.
            </p>
            <p>
                Modelos evaluados correctamente: <strong>{n_ok}</strong>.
                Modelos con error: <strong>{n_error}</strong>.
            </p>
            <p>
                La evaluación se realizó usando modelos persistidos desde:
                <code>{self.model_dir}</code>.
            </p>
            {recommendations}
        </section>
        """

    @staticmethod
    def _evidence_section() -> str:
        return """
        <section class='card'>
            <h2>Evidencia exportada</h2>
            <p>
                Además del HTML, esta fase exporta archivos de evidencia para
                auditoría y trazabilidad:
            </p>
            <ul>
                <li><code>model_evaluation_metrics.csv</code></li>
                <li><code>model_evaluation_metrics.json</code></li>
                <li><code>model_evaluation_comparison.csv</code></li>
                <li><code>model_evaluation_status.csv</code></li>
                <li><code>model_evaluation_export_manifest.json</code></li>
                <li><code>predictions/*.csv</code> con predicciones por modelo</li>
            </ul>
        </section>
        """

    def _plots_section(self) -> str:
        plots = [
            (p.stem.replace("_", " ").title(), p.name)
            for p in sorted(self.plot_files)
            if p.suffix.lower() == ".png"
        ]

        if not plots:
            return """
            <section class='card'>
                <h2>Gráficos de evaluación del modelo</h2>
                <p>No se recibieron gráficos para incluir en este reporte.</p>
            </section>
            """

        return plot_grid_html(
            plots,
            relative_plot_dir="../plots/model_evaluation",
        )

    @staticmethod
    def _interpretation_notes() -> str:
        return """
        <section class='card'>
            <h2>Guía de interpretación</h2>
            <p>
                Para modelos de regresión, valores altos de <strong>R²</strong> y
                errores bajos en <strong>MAE/RMSE</strong> indican mejor
                generalización. Los residuos deben estar centrados cerca de cero
                y sin patrones fuertes frente al valor estimado.
            </p>
            <p>
                Para clasificación, <strong>ROC AUC</strong> mide capacidad de
                discriminación, mientras que <strong>F1</strong>,
                <strong>precision</strong> y <strong>recall</strong> son más
                sensibles al desbalance de clases. Si ROC AUC es aceptable pero
                F1 es bajo, conviene revisar el umbral de decisión.
            </p>
            <p>
                Los deltas comparan evaluación contra entrenamiento. Una caída
                relevante en R², ROC AUC o F1 puede indicar drift, cambio en la
                simulación o necesidad de reentrenamiento.
            </p>
        </section>
        """

    def build_report(self) -> Path:
        tables = self.build_tables()

        body = self._intro()
        body += self._evaluation_context(tables)

        body += to_html_table(
            tables.get("evaluation_status"),
            "1. Estado y recomendación por modelo",
        )

        body += to_html_table(
            tables.get("evaluation_comparison"),
            "2. Comparación de métricas de evaluación",
        )

        body += to_html_table(
            tables.get("evaluation_metrics"),
            "3. Métricas completas de evaluación sobre datos nuevos",
        )

        body += self._plots_section()
        body += self._interpretation_notes()
        body += self._evidence_section()

        html = build_html_page(
            "Reporte de evaluación de modelos - Blob Storage Monte Carlo",
            body,
        )

        html = html.replace(
            "Fase descriptiva univariada para simulación Monte Carlo de Blob Storage.",
            "Fase de evaluación de modelos entrenados sobre datos nuevos.",
        )

        html = html.replace(
            "Este reporte es estrictamente univariado: no contiene correlaciones,\n"
            "            regresiones, VIF, cruces entre variables, scatter plots, heatmaps ni análisis bivariado.\n"
            "            El conteo temporal de muestras se incluye únicamente como control de frecuencia.",
            "Este reporte evalúa modelos previamente entrenados. No entrena de nuevo.",
        )

        html = html.replace(
            "8. Gráficos univariados",
            "4. Gráficos estadísticos de evaluación",
        )

        html = html.replace(
            "Histogramas, histogramas log1p, boxplots, boxplots recortados,\n"
            "            barras de frecuencia y conteo temporal de muestras. No se incluyen\n"
            "            cruces entre variables.",
            "Incluye observado vs estimado, residuos, error absoluto, matrices de confusión, "
            "curvas ROC, Precision-Recall, calibración y distribuciones de probabilidad.",
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(html, encoding="utf-8")

        return self.output_path
