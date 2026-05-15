from __future__ import annotations

from pathlib import Path
import os
import math

import pandas as pd

from ml_modeling.common import MLModelResult, fmt, safe_rel, save_table
from ml_modeling.plots import plot_metric_comparison


def _card(label: str, value, hint: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{fmt(value)}</div>
        <div class="metric-hint">{hint}</div>
    </div>
    """


def _table(df: pd.DataFrame | None, title: str, description: str = "", max_rows: int = 50) -> str:
    if df is None or df.empty:
        return f"<section class='card'><h2>{title}</h2><p class='muted'>No disponible.</p></section>"

    data = df.copy()

    for col in data.select_dtypes(include="number").columns:
        data[col] = data[col].round(5)

    if len(data) > max_rows:
        data = data.head(max_rows)
        description += f" Se muestran las primeras {max_rows} filas."

    return f"""
    <section class="card">
        <h2>{title}</h2>
        <p class="section-desc">{description}</p>
        <div class="table-container">
            {data.to_html(index=False, border=0, classes="data-table")}
        </div>
    </section>
    """


def _html(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
:root {{
  --bg:#f3f6fa; --card:#fff; --ink:#17212b; --muted:#667085;
  --primary:#17324d; --accent:#2c7be5; --border:#e5edf3;
}}
body {{ font-family: Arial, sans-serif; margin:0; background:var(--bg); color:var(--ink); }}
.page {{ max-width:1320px; margin:0 auto; padding:32px; }}
.hero {{ background:linear-gradient(135deg,#17324d,#245a89); color:white; padding:28px; border-radius:18px; margin-bottom:24px; }}
.hero h1 {{ margin:0 0 8px 0; font-size:30px; }}
.hero p {{ margin:0; opacity:.92; }}
.card {{ background:white; padding:22px; margin-bottom:22px; border-radius:16px; border:1px solid var(--border); box-shadow:0 2px 10px rgba(0,0,0,.05); }}
.note {{ background:#eef6ff; border-left:6px solid var(--accent); padding:16px 18px; border-radius:12px; margin-bottom:22px; line-height:1.5; }}
h2 {{ color:var(--primary); margin-top:0; border-bottom:1px solid var(--border); padding-bottom:9px; }}
.section-desc,.muted {{ color:var(--muted); font-size:14px; }}
.metrics-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin-bottom:22px; }}
.metric-card {{ background:white; border:1px solid var(--border); border-radius:14px; padding:18px; box-shadow:0 1px 6px rgba(0,0,0,.05); }}
.metric-label {{ font-size:13px; color:var(--muted); margin-bottom:8px; }}
.metric-value {{ font-size:25px; font-weight:700; color:var(--primary); }}
.metric-hint {{ font-size:12px; color:var(--muted); margin-top:8px; }}
.table-container {{ overflow-x:auto; }}
table.data-table, table.dataframe {{ border-collapse:collapse; width:100%; font-size:13px; }}
table.data-table th, table.dataframe th {{ background-color:var(--primary); color:white; padding:8px; text-align:left; }}
table.data-table td, table.dataframe td {{ border:1px solid #dde5ed; padding:7px; vertical-align:top; }}
table.data-table tr:nth-child(even), table.dataframe tr:nth-child(even) {{ background-color:#f8fafc; }}
.plot-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px; }}
.plot-card {{ margin:0; background:#fbfdff; border:1px solid var(--border); border-radius:14px; padding:12px; }}
.plot-card img {{ width:100%; max-width:100%; display:block; border-radius:10px; border:1px solid #d9e2ec; background:white; }}
.plot-card figcaption {{ margin-top:8px; font-size:13px; color:var(--muted); text-align:center; }}
a {{ color:var(--accent); font-weight:600; text-decoration:none; }}
</style>
</head>
<body>
<main class="page">
<section class="hero"><h1>{title}</h1><p>Modelamiento de Machine Learning aplicado al universo simulado de Blob Storage.</p></section>
{body}
</main>
</body>
</html>"""


def _plot_grid(paths: list[Path] | None, html_dir: Path) -> str:
    if not paths:
        return "<section class='card'><h2>Gráficos</h2><p class='muted'>No se generaron gráficos.</p></section>"

    items = ""
    for p in paths:
        rel = safe_rel(Path(p), html_dir)
        title = Path(p).stem.replace("_", " ").title()
        items += f"<figure class='plot-card'><img src='{rel}' alt='{title}'><figcaption>{title}</figcaption></figure>"

    return f"<section class='card'><h2>Gráficos</h2><div class='plot-grid'>{items}</div></section>"


def build_ml_model_report(result: MLModelResult) -> Path:
    html_dir = result.output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    metric = result.metrics.iloc[0].to_dict() if result.metrics is not None and not result.metrics.empty else {}

    cards = ""
    cards += _card("R²", metric.get("r2"), "Regresión")
    cards += _card("RMSE", metric.get("rmse"), "Error")
    cards += _card("F1", metric.get("f1"), "Clasificación")
    cards += _card("ROC-AUC", metric.get("roc_auc"), "Separabilidad")

    intro = f"""
    <section class="note">
      <strong>Técnica:</strong> {result.technique}<br>
      <strong>Target:</strong> {result.target}<br>
      <strong>Tipo:</strong> {result.task_type}<br>
      <strong>Modelo:</strong> {result.model_path}<br><br>
      Este reporte permite comparar modelos ML contra los modelos estadísticos previos.
      Los modelos de árbol y boosting capturan no linealidades y relaciones de interacción que las regresiones clásicas no representan bien.
    </section>
    <section class="metrics-grid">{cards}</section>
    """

    body = intro
    body += _table(result.metrics, "1. Métricas")
    body += _table(result.feature_importance.head(25) if result.feature_importance is not None else None, "2. Top variables", "Variables más influyentes según importancia o coeficiente.")
    body += _plot_grid(result.plots, html_dir)
    body += _table(result.predictions, "3. Muestra de predicciones", max_rows=40)

    report_path = html_dir / f"report_{result.technique}_{result.target}.html"
    report_path.write_text(_html(f"Reporte ML - {result.technique}", body), encoding="utf-8")
    result.report_path = report_path
    return report_path


def _findings(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if metrics is None or metrics.empty:
        return pd.DataFrame(columns=["hallazgo", "technique", "target", "value", "lectura"])

    if "r2" in metrics.columns:
        reg = metrics.dropna(subset=["r2"])
        if not reg.empty:
            best = reg.sort_values("r2", ascending=False).iloc[0]
            rows.append({"hallazgo":"mejor_r2", "technique":best["technique"], "target":best["target"], "value":best["r2"], "lectura":"Mejor capacidad explicativa en regresión."})

    if "rmse" in metrics.columns:
        reg = metrics.dropna(subset=["rmse"])
        if not reg.empty:
            best = reg.sort_values("rmse", ascending=True).iloc[0]
            rows.append({"hallazgo":"menor_rmse", "technique":best["technique"], "target":best["target"], "value":best["rmse"], "lectura":"Menor error de predicción en regresión."})

    if "roc_auc" in metrics.columns:
        clf = metrics.dropna(subset=["roc_auc"])
        if not clf.empty:
            best = clf.sort_values("roc_auc", ascending=False).iloc[0]
            rows.append({"hallazgo":"mejor_roc_auc", "technique":best["technique"], "target":best["target"], "value":best["roc_auc"], "lectura":"Mejor separabilidad de clases."})

    if "f1" in metrics.columns:
        clf = metrics.dropna(subset=["f1"])
        if not clf.empty:
            best = clf.sort_values("f1", ascending=False).iloc[0]
            rows.append({"hallazgo":"mejor_f1", "technique":best["technique"], "target":best["target"], "value":best["f1"], "lectura":"Mejor equilibrio precisión/recall."})

    return pd.DataFrame(rows)


def build_ml_master_report(results: list[MLModelResult], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    html_dir = output_dir / "html"
    table_dir = output_dir / "tables"
    plot_dir = output_dir / "plots" / "comparison"
    html_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    metrics = pd.concat([r.metrics for r in results], ignore_index=True) if results else pd.DataFrame()
    findings = _findings(metrics)

    registry_rows = []
    for r in results:
        link = safe_rel(r.report_path, html_dir) if r.report_path else ""
        registry_rows.append({
            "technique": r.technique,
            "target": r.target,
            "task_type": r.task_type,
            "report": str(r.report_path) if r.report_path else "",
            "model": str(r.model_path) if r.model_path else "",
            "report_link": f"<a href='{link}'>Abrir reporte</a>" if link else "",
        })

    registry = pd.DataFrame(registry_rows)

    save_table(metrics, table_dir / "metrics_all_ml_models.csv")
    save_table(findings, table_dir / "ml_findings.csv")
    save_table(registry, table_dir / "ml_model_registry.csv")

    comparison_plots = []
    for metric in ["r2", "rmse", "f1", "roc_auc"]:
        p = plot_metric_comparison(metrics, plot_dir, metric)
        if p:
            comparison_plots.append(p)

    cards = ""
    cards += _card("Modelos ML", len(results), "Técnicas ejecutadas")
    cards += _card("Mejor R²", findings[findings["hallazgo"] == "mejor_r2"]["value"].iloc[0] if not findings[findings["hallazgo"] == "mejor_r2"].empty else None, "Regresión")
    cards += _card("Menor RMSE", findings[findings["hallazgo"] == "menor_rmse"]["value"].iloc[0] if not findings[findings["hallazgo"] == "menor_rmse"].empty else None, "Regresión")
    cards += _card("Mejor ROC-AUC", findings[findings["hallazgo"] == "mejor_roc_auc"]["value"].iloc[0] if not findings[findings["hallazgo"] == "mejor_roc_auc"].empty else None, "Clasificación")

    body = f"""
    <section class="note">
      Este reporte consolida modelos de Machine Learning y permite compararlos con la fase estadística.
      La finalidad es identificar cuándo los modelos no lineales mejoran la capacidad predictiva del simulador.
    </section>
    <section class="metrics-grid">{cards}</section>
    """

    body += _table(findings, "1. Hallazgos ML")
    body += _table(registry, "2. Registro de modelos ML")
    body += _table(metrics, "3. Métricas comparativas")
    body += _plot_grid(comparison_plots, html_dir)

    html = _html("Reporte maestro de Machine Learning", body)
    html = html.replace("&lt;a href=", "<a href=").replace("&lt;/a&gt;", "</a>").replace("&gt;Abrir reporte", ">Abrir reporte")

    report_path = html_dir / "report_ml_master.html"
    report_path.write_text(html, encoding="utf-8")
    return report_path
