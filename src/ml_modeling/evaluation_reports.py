from __future__ import annotations

from pathlib import Path
import os
import math

import pandas as pd


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


def _rel(path: str | Path, start: str | Path) -> str:
    return os.path.relpath(Path(path), start=Path(start)).replace("\\", "/")


def _card(label: str, value, hint: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{_fmt(value)}</div>
        <div class="metric-hint">{hint}</div>
    </div>
    """


def _table(df: pd.DataFrame | None, title: str, description: str = "", max_rows: int = 80) -> str:
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


def _plot_grid(plot_files: list[Path], html_dir: Path) -> str:
    if not plot_files:
        return "<section class='card'><h2>Gráficos</h2><p class='muted'>No se generaron gráficos.</p></section>"

    items = ""

    for plot in plot_files:
        rel = _rel(plot, html_dir)
        title = Path(plot).stem.replace("_", " ").title()
        items += f"""
        <figure class="plot-card">
            <img src="{rel}" alt="{title}">
            <figcaption>{title}</figcaption>
        </figure>
        """

    return f"""
    <section class="card">
        <h2>Gráficos de evaluación</h2>
        <div class="plot-grid">{items}</div>
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
.page {{ max-width:1360px; margin:0 auto; padding:32px; }}
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
</style>
</head>
<body>
<main class="page">
<section class="hero">
<h1>{title}</h1>
<p>Evaluación de modelos guardados sobre nuevos datasets generados por el simulador, sin reentrenamiento.</p>
</section>
{body}
</main>
</body>
</html>"""


def build_evaluation_findings(metrics_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if metrics_df is None or metrics_df.empty:
        return pd.DataFrame(columns=["hallazgo", "technique", "target", "value", "lectura"])

    ok = metrics_df[metrics_df.get("status", "ok") == "ok"].copy()

    if "r2" in ok.columns:
        reg = ok.dropna(subset=["r2"])
        if not reg.empty:
            best = reg.sort_values("r2", ascending=False).iloc[0]
            rows.append({
                "hallazgo": "mejor_r2",
                "technique": best.get("technique"),
                "target": best.get("target"),
                "value": best.get("r2"),
                "lectura": "Modelo con mejor capacidad explicativa sobre nuevos datos.",
            })

    if "rmse" in ok.columns:
        reg = ok.dropna(subset=["rmse"])
        if not reg.empty:
            best = reg.sort_values("rmse", ascending=True).iloc[0]
            rows.append({
                "hallazgo": "menor_rmse",
                "technique": best.get("technique"),
                "target": best.get("target"),
                "value": best.get("rmse"),
                "lectura": "Modelo con menor error sobre nuevos datos.",
            })

    if "roc_auc" in ok.columns:
        clf = ok.dropna(subset=["roc_auc"])
        if not clf.empty:
            best = clf.sort_values("roc_auc", ascending=False).iloc[0]
            rows.append({
                "hallazgo": "mejor_roc_auc",
                "technique": best.get("technique"),
                "target": best.get("target"),
                "value": best.get("roc_auc"),
                "lectura": "Modelo con mejor separación entre clases en nuevos datos.",
            })

    if "recall" in ok.columns:
        clf = ok.dropna(subset=["recall"])
        if not clf.empty:
            best = clf.sort_values("recall", ascending=False).iloc[0]
            rows.append({
                "hallazgo": "mejor_recall",
                "technique": best.get("technique"),
                "target": best.get("target"),
                "value": best.get("recall"),
                "lectura": "Modelo que recupera más errores reales.",
            })

    return pd.DataFrame(rows)


def build_ml_evaluation_report(
    metrics_df: pd.DataFrame,
    detailed_results: dict[str, pd.DataFrame],
    plot_files: list[Path],
    output_path: str | Path,
    model_root: str | Path,
    evidence_dir: str | Path,
    threshold: float = 0.5,
) -> Path:
    output_path = Path(output_path)
    html_dir = output_path.parent
    html_dir.mkdir(parents=True, exist_ok=True)

    findings = build_evaluation_findings(metrics_df)

    ok_count = int((metrics_df.get("status") == "ok").sum()) if "status" in metrics_df else len(metrics_df)
    error_count = int((metrics_df.get("status") == "error").sum()) if "status" in metrics_df else 0

    best_r2 = None
    if not findings[findings["hallazgo"] == "mejor_r2"].empty:
        best_r2 = findings[findings["hallazgo"] == "mejor_r2"]["value"].iloc[0]

    best_auc = None
    if not findings[findings["hallazgo"] == "mejor_roc_auc"].empty:
        best_auc = findings[findings["hallazgo"] == "mejor_roc_auc"]["value"].iloc[0]

    cards = ""
    cards += _card("Modelos OK", ok_count, "Evaluados sin reentrenar")
    cards += _card("Modelos con error", error_count, "Revisar registry/model path")
    cards += _card("Mejor R²", best_r2, "Regresión")
    cards += _card("Mejor ROC-AUC", best_auc, "Clasificación")

    body = f"""
    <section class="note">
        <strong>Directorio de modelos:</strong> {model_root}<br>
        <strong>Evidencia exportada:</strong> {evidence_dir}<br>
        <strong>Threshold clasificación:</strong> {threshold}<br><br>
        Este reporte representa la fase tipo producción: se generan nuevos datasets y los modelos guardados se evalúan sin reentrenar.
        Esto permite medir generalización, robustez, degradación y sensibilidad ante nueva variabilidad simulada.
    </section>

    <section class="metrics-grid">
        {cards}
    </section>
    """

    body += _table(findings, "1. Hallazgos de evaluación", "Resumen ejecutivo sobre el desempeño de modelos guardados.")
    body += _table(metrics_df, "2. Métricas completas", "Resultados sobre el dataset nuevo.")
    body += _plot_grid(plot_files, html_dir)
    body += _table(
        pd.DataFrame([
            {"modelo": key, "predicciones": len(value)}
            for key, value in detailed_results.items()
        ]),
        "3. Predicciones exportadas",
        "Cantidad de predicciones generadas por modelo.",
    )

    output_path.write_text(
        _html("Evaluación de modelos guardados sobre nuevos datasets", body),
        encoding="utf-8",
    )

    return output_path
