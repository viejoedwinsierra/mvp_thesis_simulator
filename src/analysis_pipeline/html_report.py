from pathlib import Path

import pandas as pd


def format_table(df: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    table = df.copy()

    for col in table.select_dtypes(include="number").columns:
        table[col] = table[col].round(decimals)

    for col in table.columns:
        if str(table[col].dtype) == "category":
            table[col] = table[col].astype("object")

    table = table.astype("object")
    return table.where(pd.notnull(table), "")


def get_result_table(results: dict, key_tables: dict, key: str):
    value = results.get(key)
    if value is not None:
        return value
    return key_tables.get(key)


def build_general_info(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame([{
        "n_rows": len(df),
        "n_columns": df.shape[1],
        "numeric_columns": df.select_dtypes(include="number").shape[1],
        "categorical_columns": df.select_dtypes(include=["object", "category", "bool"]).shape[1],
        "duplicated_rows": df.duplicated().sum(),
        "total_nulls": df.isna().sum().sum(),
        "total_cells": df.size,
        "null_pct_dataset": (df.isna().sum().sum() / df.size * 100) if df.size > 0 else 0,
    }])


def build_executive_summary(
    numeric_summary: pd.DataFrame | None,
    categorical_summary: pd.DataFrame | None,
    data_quality: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []

    if numeric_summary is not None and not numeric_summary.empty:
        needs_log = numeric_summary[
            numeric_summary["modeling_recommendation"].eq("NEEDS_LOG")
        ]["variable"].tolist()

        high_outliers = numeric_summary[
            numeric_summary["riesgo_outliers"].eq("alto")
        ]["variable"].tolist()

        heavy_tail = numeric_summary[
            numeric_summary.get("distribution_type", "").eq("heavy_tail")
        ]["variable"].tolist() if "distribution_type" in numeric_summary.columns else []

        rows.append({
            "aspecto": "Variables numéricas a transformar",
            "resultado": ", ".join(needs_log) if needs_log else "Sin alertas relevantes",
        })

        rows.append({
            "aspecto": "Variables con outliers altos",
            "resultado": ", ".join(high_outliers) if high_outliers else "Sin outliers altos",
        })

        rows.append({
            "aspecto": "Variables con colas pesadas",
            "resultado": ", ".join(heavy_tail) if heavy_tail else "Sin colas pesadas detectadas",
        })

    if categorical_summary is not None and not categorical_summary.empty:
        exclude_vars = categorical_summary[
            categorical_summary["recommendation"].eq("excluir")
        ]["variable"].tolist()

        rows.append({
            "aspecto": "Variables categóricas a excluir",
            "resultado": ", ".join(exclude_vars) if exclude_vars else "Sin exclusiones categóricas",
        })

    if data_quality is not None and not data_quality.empty:
        critical = data_quality[
            data_quality.get("quality_severity", "").eq("CRITICAL")
        ]["variable"].tolist() if "quality_severity" in data_quality.columns else []

        warning = data_quality[
            data_quality.get("quality_severity", "").eq("WARNING")
        ]["variable"].tolist() if "quality_severity" in data_quality.columns else []

        rows.append({
            "aspecto": "Variables críticas de calidad",
            "resultado": ", ".join(critical) if critical else "Sin variables críticas",
        })

        rows.append({
            "aspecto": "Variables con advertencia",
            "resultado": ", ".join(warning) if warning else "Sin advertencias",
        })

    return pd.DataFrame(rows)


def to_html_table(
    df,
    title: str,
    description: str | None = None,
    css_class: str = "",
) -> str:
    if df is None:
        return f"""
        <section class="card {css_class}">
            <h2>{title}</h2>
            <p>No disponible.</p>
        </section>
        """

    if isinstance(df, dict):
        df = pd.DataFrame([df])

    if isinstance(df, list):
        df = pd.DataFrame({"conclusion": df})

    if not isinstance(df, pd.DataFrame) or df.empty:
        return f"""
        <section class="card {css_class}">
            <h2>{title}</h2>
            <p>No disponible.</p>
        </section>
        """

    desc_html = f"<p class='section-desc'>{description}</p>" if description else ""
    table = format_table(df)

    return f"""
    <section class="card {css_class}">
        <h2>{title}</h2>
        {desc_html}
        <div class="table-container">
            {table.to_html(index=False, border=0)}
        </div>
    </section>
    """


def plot_grid_html(
    plots: list[tuple[str, str]],
    relative_plot_dir: str = "../plots/descriptive",
) -> str:
    if not plots:
        return """
        <section class="card">
            <h2>8. Gráficos univariados</h2>
            <p>No se generaron gráficos.</p>
        </section>
        """

    body = """
    <section class="card">
        <h2>8. Gráficos univariados</h2>
        <p class="section-desc">
            Histogramas, histogramas log1p, boxplots, boxplots recortados,
            barras de frecuencia y conteo temporal de muestras. No se incluyen
            cruces entre variables.
        </p>
        <div class="grid">
    """

    for title, file_name in plots:
        body += f"""
        <div class="plot-card">
            <h3>{title}</h3>
            <img src="{relative_plot_dir}/{file_name}" alt="{title}">
        </div>
        """

    body += """
        </div>
    </section>
    """
    return body


def build_html_page(title: str, body: str) -> str:
    return f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <title>{title}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 32px;
                color: #2c3e50;
                background: #f8f9fa;
            }}

            h1 {{
                margin-bottom: 6px;
                color: #1f3a56;
            }}

            h2 {{
                color: #1f3a56;
                margin-top: 0;
                border-bottom: 2px solid #e8eef3;
                padding-bottom: 8px;
            }}

            h3 {{
                margin-top: 0;
                color: #2c3e50;
                font-size: 15px;
            }}

            .subtitle {{
                color: #666;
                margin-bottom: 24px;
            }}

            .card {{
                background: white;
                padding: 20px;
                margin-bottom: 24px;
                border-radius: 10px;
                box-shadow: 0 1px 5px rgba(0,0,0,0.08);
            }}

            .note {{
                background: #eef6ff;
                border-left: 5px solid #2c7be5;
                padding: 14px;
                border-radius: 6px;
                margin-bottom: 24px;
                font-size: 14px;
            }}

            .summary-card {{
                border-left: 5px solid #198754;
            }}

            .section-desc {{
                color: #555;
                font-size: 14px;
                margin-top: -4px;
                margin-bottom: 14px;
            }}

            .table-container {{
                overflow-x: auto;
            }}

            table {{
                border-collapse: collapse;
                width: 100%;
                font-size: 13px;
            }}

            th {{
                background-color: #2c3e50;
                color: white;
                padding: 8px;
                text-align: left;
                white-space: nowrap;
            }}

            td {{
                border: 1px solid #ddd;
                padding: 7px;
                white-space: nowrap;
            }}

            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}

            .grid {{
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 20px;
            }}

            .plot-card {{
                background: #ffffff;
                padding: 16px;
                border-radius: 8px;
                border: 1px solid #e4e4e4;
            }}

            img {{
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 4px;
                background: white;
            }}

            @media (max-width: 900px) {{
                .grid {{
                    grid-template-columns: 1fr;
                }}
            }}
        </style>
    </head>
    <body>
        <h1>{title}</h1>
        <p class="subtitle">
            Fase descriptiva univariada para simulación Monte Carlo de Blob Storage.
        </p>

        <div class="note">
            Este reporte es estrictamente univariado: no contiene correlaciones,
            regresiones, VIF, cruces entre variables, scatter plots, heatmaps ni análisis bivariado.
            El conteo temporal de muestras se incluye únicamente como control de frecuencia.
        </div>

        {body}
    </body>
    </html>
    """


def build_descriptive_report(
    df: pd.DataFrame,
    results: dict,
    data_dictionary_df: pd.DataFrame | None,
    key_tables: dict | None,
    plots: list[tuple[str, str]],
    output_path: str | Path,
) -> Path:
    key_tables = key_tables or {}
    results = results or {}

    numeric_summary = get_result_table(results, key_tables, "numeric_summary")
    categorical_summary = get_result_table(results, key_tables, "categorical_summary")
    outlier_summary = get_result_table(results, key_tables, "outlier_summary")
    data_quality = get_result_table(results, key_tables, "data_quality")
    variable_classification = get_result_table(results, key_tables, "variable_classification")
    automatic_conclusions = get_result_table(results, key_tables, "automatic_conclusions")

    executive_summary = build_executive_summary(
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        data_quality=data_quality,
    )

    body = ""

    body += to_html_table(
        build_general_info(df),
        "1. Información general",
        "Resumen estructural del dataset usado en la fase descriptiva.",
    )

    body += to_html_table(
        executive_summary,
        "2. Resumen ejecutivo univariado",
        "Síntesis automática de alertas relevantes para la siguiente fase.",
        css_class="summary-card",
    )

    if data_dictionary_df is not None:
        body += to_html_table(
            data_dictionary_df,
            "3. Diccionario de variables",
            "Definición funcional de las variables disponibles.",
        )

    body += to_html_table(
        data_quality,
        "4. Calidad de datos",
        "Nulos, valores únicos, flags de calidad y severidad.",
    )

    body += to_html_table(
        numeric_summary,
        "5. Análisis univariado numérico",
        "Incluye tendencia central, dispersión, percentiles, asimetría, curtosis, tipo de distribución, escala y recomendación de modelamiento.",
    )

    body += to_html_table(
        categorical_summary,
        "6. Variables categóricas",
        "Número de categorías, categoría dominante, cardinalidad y recomendación de uso.",
    )

    body += to_html_table(
        outlier_summary,
        "7. Outliers univariados",
        "Detección por método IQR. En simulación Monte Carlo no se recomienda eliminar automáticamente valores extremos.",
    )

    body += plot_grid_html(plots)

    body += to_html_table(
        variable_classification,
        "9. Clasificación de variables",
        "Rol sugerido de cada variable para fases posteriores de modelamiento.",
    )

    body += to_html_table(
        automatic_conclusions,
        "10. Interpretación automática",
        "Conclusiones generadas a partir de asimetría, distribución, outliers, cardinalidad, nulos, derivadas y posible fuga de información.",
    )

    html = build_html_page(
        "Reporte descriptivo univariado - Blob Storage Monte Carlo",
        body,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    return output_path