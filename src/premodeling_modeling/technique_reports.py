from __future__ import annotations

from pathlib import Path
import os
import math

import pandas as pd

from premodeling_modeling.common import (
    TechniqueResult,
    save_table,
)


# ============================================================
# Helpers HTML
# ============================================================

def _safe_relative_link(target_path: Path, from_dir: Path) -> str:
    target_path = Path(target_path)
    from_dir = Path(from_dir)
    return os.path.relpath(target_path, start=from_dir).replace("\\", "/")


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass

    if isinstance(value, (int, float)):
        if math.isfinite(value):
            return f"{value:,.{digits}f}"
        return "N/A"

    return str(value)


def _get_metric(metrics: pd.DataFrame | None, name: str):
    if metrics is None or metrics.empty or name not in metrics.columns:
        return None
    return metrics.iloc[0].get(name)


def _metric_card(label: str, value, hint: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{_fmt(value)}</div>
        <div class="metric-hint">{hint}</div>
    </div>
    """


def _table(df: pd.DataFrame | None, title: str, description: str = "", max_rows: int = 50) -> str:
    if df is None or df.empty:
        return f"""
        <section class="card">
            <h2>{title}</h2>
            <p class="muted">No disponible.</p>
        </section>
        """

    table = df.copy()

    for col in table.select_dtypes(include="number").columns:
        table[col] = table[col].round(5)

    if len(table) > max_rows:
        table = table.head(max_rows)
        description += f" Se muestran las primeras {max_rows} filas."

    return f"""
    <section class="card">
        <h2>{title}</h2>
        <p class="section-desc">{description}</p>
        <div class="table-container">
            {table.to_html(index=False, border=0, classes="data-table")}
        </div>
    </section>
    """


def _top_coefficients(coefficients: pd.DataFrame | None, n: int = 12) -> pd.DataFrame:
    if coefficients is None or coefficients.empty:
        return pd.DataFrame()

    coef = coefficients.copy()

    if "coefficient" not in coef.columns:
        return coef.head(n)

    if "abs_coefficient" not in coef.columns:
        coef["abs_coefficient"] = coef["coefficient"].abs()

    return (
        coef.sort_values("abs_coefficient", ascending=False)
            .head(n)
            .reset_index(drop=True)
    )


def _predictions_sample(predictions: pd.DataFrame | None, n: int = 25) -> pd.DataFrame:
    if predictions is None or predictions.empty:
        return pd.DataFrame()

    cols = [c for c in ["target", "y_true", "y_pred", "residual", "probability_error"] if c in predictions.columns]
    if not cols:
        cols = list(predictions.columns)

    return predictions[cols].head(n).copy()


def _plot_grid(plot_paths: list[Path] | None, html_dir: Path) -> str:
    if not plot_paths:
        return """
        <section class="card">
            <h2>Gráficos del modelo</h2>
            <p class="muted">No se generaron gráficos.</p>
        </section>
        """

    items = ""

    for path in plot_paths:
        path = Path(path)
        rel = _safe_relative_link(path, html_dir)
        title = path.stem.replace("_", " ").title()

        items += f"""
        <figure class="plot-card">
            <img src="{rel}" alt="{title}">
            <figcaption>{title}</figcaption>
        </figure>
        """

    return f"""
    <section class="card">
        <h2>Gráficos del modelo</h2>
        <p class="section-desc">
            Los gráficos se cargan con rutas relativas desde este HTML hacia la carpeta de plots de la técnica.
        </p>
        <div class="plot-grid">
            {items}
        </div>
    </section>
    """


def _professional_html(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        :root {{
            --bg: #f3f6fa;
            --card: #ffffff;
            --ink: #17212b;
            --muted: #667085;
            --primary: #17324d;
            --accent: #2c7be5;
            --border: #e5edf3;
            --good: #0f766e;
            --warn: #b45309;
        }}

        body {{
            font-family: Arial, Helvetica, sans-serif;
            margin: 0;
            color: var(--ink);
            background: var(--bg);
        }}

        .page {{
            max-width: 1320px;
            margin: 0 auto;
            padding: 32px;
        }}

        .hero {{
            background: linear-gradient(135deg, #17324d, #245a89);
            color: white;
            padding: 28px;
            border-radius: 18px;
            margin-bottom: 24px;
            box-shadow: 0 10px 30px rgba(23,50,77,0.18);
        }}

        .hero h1 {{
            margin: 0 0 8px 0;
            font-size: 30px;
        }}

        .hero p {{
            margin: 0;
            opacity: 0.92;
            font-size: 15px;
        }}

        .card {{
            background: var(--card);
            padding: 22px;
            margin-bottom: 22px;
            border-radius: 16px;
            border: 1px solid var(--border);
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        .note {{
            background: #eef6ff;
            border-left: 6px solid var(--accent);
            padding: 16px 18px;
            border-radius: 12px;
            margin-bottom: 22px;
            line-height: 1.5;
        }}

        h2 {{
            color: var(--primary);
            margin-top: 0;
            border-bottom: 1px solid var(--border);
            padding-bottom: 9px;
        }}

        .section-desc, .muted {{
            color: var(--muted);
            font-size: 14px;
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 16px;
            margin-bottom: 22px;
        }}

        .metric-card {{
            background: white;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 18px;
            box-shadow: 0 1px 6px rgba(0,0,0,0.05);
        }}

        .metric-label {{
            font-size: 13px;
            color: var(--muted);
            margin-bottom: 8px;
        }}

        .metric-value {{
            font-size: 25px;
            font-weight: 700;
            color: var(--primary);
        }}

        .metric-hint {{
            font-size: 12px;
            color: var(--muted);
            margin-top: 8px;
        }}

        .table-container {{
            overflow-x: auto;
        }}

        table.data-table, table.dataframe {{
            border-collapse: collapse;
            width: 100%;
            font-size: 13px;
        }}

        table.data-table th, table.dataframe th {{
            background-color: var(--primary);
            color: white;
            padding: 8px;
            text-align: left;
            position: sticky;
            top: 0;
        }}

        table.data-table td, table.dataframe td {{
            border: 1px solid #dde5ed;
            padding: 7px;
            vertical-align: top;
        }}

        table.data-table tr:nth-child(even), table.dataframe tr:nth-child(even) {{
            background-color: #f8fafc;
        }}

        .plot-grid {{
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 20px;
        }}

        .plot-card {{
            margin: 0;
            background: #fbfdff;
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px;
        }}

        .plot-card img {{
            width: 100%;
            max-width: 100%;
            display: block;
            border-radius: 10px;
            border: 1px solid #d9e2ec;
            background: white;
        }}

        .plot-card figcaption {{
            margin-top: 8px;
            font-size: 13px;
            color: var(--muted);
            text-align: center;
        }}

        .two-col {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}

        a {{
            color: var(--accent);
            font-weight: 600;
            text-decoration: none;
        }}

        a:hover {{
            text-decoration: underline;
        }}
    </style>
</head>
<body>
    <main class="page">
        <section class="hero">
            <h1>{title}</h1>
            <p>Reporte estadístico generado para documentar técnica, métricas, coeficientes, predicciones, residuos y evidencia visual.</p>
        </section>
        {body}
    </main>
</body>
</html>"""


# ============================================================
# Interpretaciones
# ============================================================

def _technique_intro(result: TechniqueResult) -> str:
    if result.technique == "linear_regression_simple":
        return (
            "Regresión lineal simple usada como baseline estadístico. "
            "Evalúa si una variable principal explica el comportamiento del target."
        )

    if result.technique == "linear_regression_multiple":
        return (
            "Regresión lineal múltiple usada para explicar el target mediante varias variables operacionales. "
            "Permite comparar magnitud y sentido de coeficientes, pero puede sufrir por no linealidad y multicolinealidad."
        )

    if result.technique == "log_log_regression":
        return (
            "Modelo log-log usado para capturar relaciones multiplicativas y elasticidades. "
            "Es útil cuando los costos o duraciones tienen alta asimetría y cambios porcentuales."
        )

    if result.technique == "logistic_regression_error":
        return (
            "Regresión logística usada para modelar la probabilidad de error. "
            "Los coeficientes positivos aumentan los log-odds del evento."
        )

    return "Modelo estadístico aplicado como parte de la fase de premodelamiento."


def _automatic_findings(result: TechniqueResult) -> pd.DataFrame:
    rows = []

    metrics = result.metrics

    if metrics is not None and not metrics.empty:
        row = metrics.iloc[0].to_dict()

        if "r2" in row and pd.notna(row.get("r2")):
            r2 = row["r2"]
            if r2 >= 0.90:
                lectura = "Ajuste muy alto; posible relación fuerte o variable transformada adecuada."
            elif r2 >= 0.60:
                lectura = "Ajuste medio-alto; el modelo explica una parte importante de la variabilidad."
            elif r2 >= 0.30:
                lectura = "Ajuste moderado; hay patrones, pero puede faltar no linealidad o segmentación."
            else:
                lectura = "Ajuste bajo; se recomienda revisar transformación o modelos no lineales."

            rows.append({
                "hallazgo": "capacidad_explicativa",
                "valor": round(float(r2), 5),
                "lectura": lectura,
            })

        if "rmse" in row and pd.notna(row.get("rmse")):
            rows.append({
                "hallazgo": "error_promedio_cuadratico",
                "valor": round(float(row["rmse"]), 5),
                "lectura": "RMSE permite comparar error entre modelos de regresión sobre el mismo target.",
            })

        if "f1" in row and pd.notna(row.get("f1")):
            f1 = row["f1"]
            rows.append({
                "hallazgo": "desempeno_clasificacion",
                "valor": round(float(f1), 5),
                "lectura": "F1 resume equilibrio entre precisión y recall.",
            })

        if "roc_auc" in row and pd.notna(row.get("roc_auc")):
            roc = row["roc_auc"]
            rows.append({
                "hallazgo": "separabilidad_clases",
                "valor": round(float(roc), 5),
                "lectura": "ROC-AUC mide capacidad de separar eventos con y sin error.",
            })

    top_coef = _top_coefficients(result.coefficients, n=5)
    if not top_coef.empty and "feature" in top_coef.columns:
        rows.append({
            "hallazgo": "variables_mas_influyentes",
            "valor": ", ".join(top_coef["feature"].astype(str).head(5)),
            "lectura": "Estas variables dominan el comportamiento del modelo según magnitud absoluta del coeficiente.",
        })

    if not rows:
        return pd.DataFrame(columns=["hallazgo", "valor", "lectura"])

    return pd.DataFrame(rows)


# ============================================================
# Reporte por técnica
# ============================================================

def build_technique_report(result: TechniqueResult) -> Path:
    output_dir = result.output_dir
    html_dir = output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    metrics = result.metrics

    metric_cards = ""

    if metrics is not None and not metrics.empty:
        metric_cards += _metric_card("R²", _get_metric(metrics, "r2"), "Capacidad explicativa")
        metric_cards += _metric_card("RMSE", _get_metric(metrics, "rmse"), "Error promedio cuadrático")
        metric_cards += _metric_card("MAE", _get_metric(metrics, "mae"), "Error absoluto medio")
        metric_cards += _metric_card("F1 / Accuracy", _get_metric(metrics, "f1") or _get_metric(metrics, "accuracy"), "Clasificación")

    findings = _automatic_findings(result)

    body = f"""
    <section class="note">
        <strong>Técnica:</strong> {result.technique}<br>
        <strong>Target:</strong> {result.target}<br>
        <strong>Modelo persistido:</strong> {result.model_path}<br><br>
        {_technique_intro(result)}
    </section>

    <section class="metrics-grid">
        {metric_cards}
    </section>
    """

    body += _table(findings, "1. Lectura automática del modelo", "Interpretación ejecutiva generada a partir de métricas y coeficientes.", max_rows=20)
    body += _table(result.metrics, "2. Métricas técnicas", "Métricas completas exportadas para esta técnica.", max_rows=20)
    body += _table(_top_coefficients(result.coefficients, n=25), "3. Top coeficientes", "Coeficientes ordenados por magnitud absoluta.", max_rows=25)
    body += _plot_grid(result.plots or [], html_dir=html_dir)
    body += _table(_predictions_sample(result.predictions, n=40), "4. Muestra de predicciones", "Primeras predicciones para revisar comportamiento del modelo.", max_rows=40)
    body += _table(result.diagnostics, "5. Diagnósticos adicionales", "Espacio reservado para VIF, supuestos, normalidad de residuos o pruebas posteriores.", max_rows=50)

    html = _professional_html(
        title=f"Reporte por técnica - {result.technique}",
        body=body,
    )

    report_path = html_dir / f"report_{result.technique}_{result.target}.html"
    report_path.write_text(html, encoding="utf-8")
    result.report_path = report_path

    return report_path


# ============================================================
# Reporte maestro
# ============================================================

def _comparison_findings(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics is None or metrics.empty:
        return pd.DataFrame(columns=["hallazgo", "tecnica", "target", "valor", "lectura"])

    rows = []

    reg = metrics.dropna(subset=["r2"], how="all") if "r2" in metrics.columns else pd.DataFrame()
    if not reg.empty:
        best_r2 = reg.sort_values("r2", ascending=False).iloc[0]
        rows.append({
            "hallazgo": "mejor_r2",
            "tecnica": best_r2.get("technique"),
            "target": best_r2.get("target"),
            "valor": best_r2.get("r2"),
            "lectura": "Mejor capacidad explicativa entre modelos de regresión.",
        })

    if "rmse" in metrics.columns:
        reg_rmse = metrics.dropna(subset=["rmse"])
        if not reg_rmse.empty:
            best_rmse = reg_rmse.sort_values("rmse", ascending=True).iloc[0]
            rows.append({
                "hallazgo": "menor_rmse",
                "tecnica": best_rmse.get("technique"),
                "target": best_rmse.get("target"),
                "valor": best_rmse.get("rmse"),
                "lectura": "Menor error absoluto en escala del target evaluado.",
            })

    if "roc_auc" in metrics.columns:
        clf = metrics.dropna(subset=["roc_auc"])
        if not clf.empty:
            best_auc = clf.sort_values("roc_auc", ascending=False).iloc[0]
            rows.append({
                "hallazgo": "mejor_roc_auc",
                "tecnica": best_auc.get("technique"),
                "target": best_auc.get("target"),
                "valor": best_auc.get("roc_auc"),
                "lectura": "Mejor separabilidad para clasificación.",
            })

    return pd.DataFrame(rows)


def build_master_modeling_report(results: list[TechniqueResult], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    html_dir = output_dir / "html"
    table_dir = output_dir / "tables"
    html_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.concat(
        [r.metrics for r in results if r.metrics is not None and not r.metrics.empty],
        ignore_index=True,
    ) if results else pd.DataFrame()

    registry_rows = []

    for result in results:
        report_link = ""
        if result.report_path:
            rel_report = _safe_relative_link(result.report_path, html_dir)
            report_link = f"<a href='{rel_report}'>Abrir reporte</a>"

        registry_rows.append({
            "technique": result.technique,
            "target": result.target,
            "report": str(result.report_path) if result.report_path else "",
            "model": str(result.model_path) if result.model_path else "",
            "plots": len(result.plots or []),
            "report_link": report_link,
        })

    registry = pd.DataFrame(registry_rows)
    findings = _comparison_findings(metrics)

    save_table(metrics, table_dir / "metrics_all_techniques.csv")
    save_table(registry, table_dir / "model_registry.csv")
    save_table(findings, table_dir / "modeling_findings.csv")

    metric_cards = ""
    metric_cards += _metric_card("Técnicas", len(results), "Modelos ejecutados")
    metric_cards += _metric_card("Mejor R²", findings[findings["hallazgo"] == "mejor_r2"]["valor"].iloc[0] if not findings[findings["hallazgo"] == "mejor_r2"].empty else None, "Regresión")
    metric_cards += _metric_card("Menor RMSE", findings[findings["hallazgo"] == "menor_rmse"]["valor"].iloc[0] if not findings[findings["hallazgo"] == "menor_rmse"].empty else None, "Regresión")
    metric_cards += _metric_card("Mejor ROC-AUC", findings[findings["hallazgo"] == "mejor_roc_auc"]["valor"].iloc[0] if not findings[findings["hallazgo"] == "mejor_roc_auc"].empty else None, "Clasificación")

    body = f"""
    <section class="note">
        Este reporte maestro compara técnicas estadísticas iniciales.
        Su objetivo es servir como puente entre analítica avanzada, premodelamiento y modelos de Machine Learning.
        El foco no es solo obtener métricas, sino documentar qué técnica explica mejor el comportamiento operacional.
    </section>

    <section class="metrics-grid">
        {metric_cards}
    </section>
    """

    body += _table(findings, "1. Hallazgos comparativos", "Resumen ejecutivo del desempeño entre técnicas.", max_rows=20)
    body += _table(registry, "2. Registro de técnicas", "Links a reportes individuales, modelos persistidos y cantidad de gráficos.", max_rows=50)
    body += _table(metrics, "3. Métricas comparativas", "Métricas consolidadas de regresión y clasificación.", max_rows=50)

    html = _professional_html(
        title="Reporte maestro de modelamiento estadístico por técnica",
        body=body,
    )

    html = (
        html.replace("&lt;a href=", "<a href=")
            .replace("&lt;/a&gt;", "</a>")
            .replace("&gt;Abrir reporte", ">Abrir reporte")
    )

    report_path = html_dir / "report_modeling_master.html"
    report_path.write_text(html, encoding="utf-8")

    return report_path
