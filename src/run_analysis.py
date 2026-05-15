from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from analysis_pipeline.config import AnalysisConfig
from analysis_pipeline.data_preparation import prepare_dataframe
from analysis_pipeline.data_dictionary import build_data_dictionary_table
from analysis_pipeline.descriptive_tables import run_descriptive_analysis
from analysis_pipeline.descriptive_plots import generate_descriptive_plots
from analysis_pipeline.html_report import (
    build_descriptive_report,
    to_html_table,
    build_html_page,
)

# Fase avanzada multivariada por escenario
from analysis_pipeline.advanced_plots import run_advanced_plots
from analysis_pipeline.advanced_html_exports import build_advanced_report
from analysis_pipeline.advanced_tables import build_advanced_tables


# ==========================================================
# CONFIGURACION GENERAL
# ==========================================================

RUN_ADVANCED_DEFAULT = True

DATASET_FILE_PATTERN = "blob_inventory*.csv"
SCENARIO_CONFIG_ROOT = Path("config/escenarios")
RUNTIME_CONFIG_ROOT = Path("output/runtime_configs")


# ==========================================================
# ARGUMENTOS
# ==========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera reportes descriptivos y avanzados personalizados por escenario. "
            "Lee datasets desde output/dataset/<escenario> y exporta imagenes, "
            "HTML y tablas por caso."
        )
    )

    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="output/dataset",
        help="Carpeta raiz donde estan los datasets por escenario.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/analysis",
        help="Carpeta raiz donde se guardaran reportes, imagenes y tablas.",
    )

    parser.add_argument(
        "--scenario-filter",
        type=str,
        default=None,
        help="Ejecuta solo escenarios cuyo nombre contenga este texto.",
    )

    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Limita la cantidad de escenarios a analizar.",
    )

    parser.add_argument(
        "--include-global",
        action="store_true",
        help="Tambien genera un reporte global uniendo todos los escenarios.",
    )

    parser.add_argument(
        "--advanced",
        action="store_true",
        help="Activa la analitica avanzada multivariada por escenario.",
    )

    parser.add_argument(
        "--no-advanced",
        action="store_true",
        help="Desactiva la analitica avanzada multivariada por escenario.",
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="No elimina reportes anteriores del escenario antes de generar.",
    )

    return parser.parse_args()


def should_run_advanced(args: argparse.Namespace) -> bool:
    if args.no_advanced:
        return False

    if args.advanced:
        return True

    return RUN_ADVANCED_DEFAULT


# ==========================================================
# DESCUBRIMIENTO Y CARGA DE DATASETS
# ==========================================================

def discover_scenario_dataset_folders(dataset_root: Path) -> list[Path]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"No existe dataset_root: {dataset_root}")

    scenario_dirs = [
        folder
        for folder in sorted(dataset_root.iterdir())
        if folder.is_dir() and list(folder.glob(DATASET_FILE_PATTERN))
    ]

    # Compatibilidad: si no hay subcarpetas, pero hay CSV directamente en dataset_root.
    if not scenario_dirs and list(dataset_root.glob(DATASET_FILE_PATTERN)):
        scenario_dirs = [dataset_root]

    if not scenario_dirs:
        raise FileNotFoundError(
            "No se encontraron escenarios con archivos "
            f"{DATASET_FILE_PATTERN} dentro de {dataset_root}"
        )

    return scenario_dirs


def load_dataset_files(files: list[Path], scenario_name: str) -> pd.DataFrame:
    frames = []

    for file in sorted(files):
        df_part = pd.read_csv(file)
        df_part["source_file"] = file.name
        df_part["source_path"] = str(file)
        df_part["scenario_name"] = scenario_name
        frames.append(df_part)

    if not frames:
        raise FileNotFoundError(f"No hay CSV para el escenario {scenario_name}")

    return pd.concat(frames, ignore_index=True)


def load_scenario_dataset(scenario_dir: Path) -> pd.DataFrame:
    scenario_name = scenario_dir.name
    files = sorted(scenario_dir.glob(DATASET_FILE_PATTERN))
    return load_dataset_files(files, scenario_name=scenario_name)


# ==========================================================
# VALIDACION DE CONFIGURACION VS DATASET
# ==========================================================

def safe_load_json(path: Path) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_scenario_config(scenario_name: str) -> tuple[Path | None, Path | None]:
    scenario_folder = SCENARIO_CONFIG_ROOT / scenario_name

    if not scenario_folder.exists():
        return None, None

    simulation_config = next(iter(sorted(scenario_folder.glob("simulation_config*.json"))), None)
    time_config = next(iter(sorted(scenario_folder.glob("time_distribution_config*.json"))), None)

    return simulation_config, time_config


def build_runtime_config_index(scenario_name: str) -> pd.DataFrame:
    runtime_folder = RUNTIME_CONFIG_ROOT / scenario_name
    rows = []

    if not runtime_folder.exists():
        return pd.DataFrame(columns=[
            "runtime_date",
            "runtime_config",
            "simulation_date",
            "max_valid_files_per_day",
            "output_dir",
            "seed",
        ])

    for config_file in sorted(runtime_folder.rglob("simulation_config*.json")):
        config = safe_load_json(config_file) or {}
        simulation = config.get("simulation", {})
        scenario_execution = config.get("scenario_execution", {})

        rows.append({
            "runtime_date": scenario_execution.get("current_simulation_date") or simulation.get("simulation_date"),
            "runtime_config": str(config_file),
            "simulation_date": simulation.get("simulation_date"),
            "max_valid_files_per_day": simulation.get("max_valid_files_per_day"),
            "output_dir": simulation.get("output_dir"),
            "seed": simulation.get("seed"),
        })

    return pd.DataFrame(rows)


def build_config_summary(scenario_name: str) -> pd.DataFrame:
    simulation_config_path, time_config_path = find_scenario_config(scenario_name)
    config = safe_load_json(simulation_config_path) if simulation_config_path else None
    time_config = safe_load_json(time_config_path) if time_config_path else None

    if not config:
        return pd.DataFrame([{
            "scenario": scenario_name,
            "config_status": "NO_CONFIG_FOUND",
        }])

    simulation = config.get("simulation", {})
    capacity = config.get("capacity", {})
    outliers = config.get("outliers", {})
    arrival = config.get("arrival_process", {})
    transfer = config.get("transfer", {}).get("speed_mbps", {})
    errors = config.get("errors", {})
    lifecycle = config.get("lifecycle", {})
    storage = config.get("storage_tier", {})

    weekly_sum = None
    if time_config:
        weekly_sum = sum(
            float(item.get("base_load", 0))
            for item in time_config.get("weekly_time_distribution", [])
        )

    file_distribution = config.get("file_types", {}).get("distribution", {})
    cost_by_tier = storage.get("cost_per_mb_per_month", {})

    return pd.DataFrame([{
        "scenario": scenario_name,
        "config_status": "OK",
        "simulation_config": str(simulation_config_path),
        "time_distribution_config": str(time_config_path),
        "max_valid_files_per_day": simulation.get("max_valid_files_per_day"),
        "base_simulation_date_in_config": simulation.get("simulation_date"),
        "arrival_strategy": arrival.get("strategy"),
        "hourly_noise": arrival.get("hourly_noise"),
        "capacity_enabled": capacity.get("enabled"),
        "files_per_hour": capacity.get("files_per_hour"),
        "duration_penalty_factor": capacity.get("duration_penalty_factor"),
        "error_penalty_factor": capacity.get("error_penalty_factor"),
        "outliers_enabled": outliers.get("enabled"),
        "outlier_probability": outliers.get("probability"),
        "outlier_multiplier_min": outliers.get("size_multiplier_min"),
        "outlier_multiplier_max": outliers.get("size_multiplier_max"),
        "transfer_mean_mbps": transfer.get("mean"),
        "transfer_sigma": transfer.get("sigma"),
        "base_error_probability": errors.get("base_error_probability"),
        "movement_storage_probability": lifecycle.get("movement_storage_probability"),
        "storage_strategy": storage.get("strategy"),
        "weekly_base_load_sum": weekly_sum,
        "file_type_distribution_json": json.dumps(file_distribution, ensure_ascii=False),
        "tier_cost_json": json.dumps(cost_by_tier, ensure_ascii=False),
    }])


def build_file_inventory_table(scenario_dir: Path) -> pd.DataFrame:
    rows = []

    for file in sorted(scenario_dir.glob(DATASET_FILE_PATTERN)):
        try:
            df = pd.read_csv(file)
            n_rows = len(df)
            dates = (
                sorted(pd.to_datetime(df["simulation_date"], errors="coerce").dt.date.dropna().unique())
                if "simulation_date" in df.columns
                else []
            )
        except Exception as exc:
            n_rows = None
            dates = []
            error = str(exc)
        else:
            error = ""

        rows.append({
            "file": file.name,
            "path": str(file),
            "rows": n_rows,
            "simulation_dates_detected": ", ".join(str(d) for d in dates),
            "read_error": error,
        })

    return pd.DataFrame(rows)


def build_execution_validation_table(df: pd.DataFrame, scenario_dir: Path) -> pd.DataFrame:
    rows = []

    scenario_name = scenario_dir.name
    config_summary = build_config_summary(scenario_name)
    expected_rows = None

    if not config_summary.empty and "max_valid_files_per_day" in config_summary.columns:
        expected_rows = config_summary.iloc[0].get("max_valid_files_per_day")

    if "simulation_date" in df.columns:
        grouped = (
            df.assign(simulation_date=pd.to_datetime(df["simulation_date"], errors="coerce").dt.date)
              .groupby("simulation_date", dropna=False)
              .size()
              .reset_index(name="generated_rows")
        )
    else:
        grouped = pd.DataFrame([{"simulation_date": "NO_COLUMN", "generated_rows": len(df)}])

    for _, row in grouped.iterrows():
        generated_rows = int(row["generated_rows"])
        diff = None
        pct = None
        status = "NO_EXPECTED"

        if expected_rows is not None:
            diff = generated_rows - int(expected_rows)
            pct = generated_rows / int(expected_rows) * 100 if int(expected_rows) else None
            status = "OK" if generated_rows >= int(expected_rows) else "LOW_ROWS"

        rows.append({
            "scenario": scenario_name,
            "simulation_date": row["simulation_date"],
            "expected_rows_from_config": expected_rows,
            "generated_rows": generated_rows,
            "difference": diff,
            "generated_pct_vs_expected": pct,
            "validation_status": status,
        })

    return pd.DataFrame(rows)


def build_parameter_observed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    checks = [
        ("file_type", "Distribucion observada por tipo de archivo"),
        ("storage_tier", "Distribucion observada por tier"),
        ("time_slot", "Distribucion observada por franja horaria"),
        ("case_type", "Distribucion observada por tipo de caso"),
        ("case_group", "Distribucion observada por grupo de caso"),
        ("has_error", "Distribucion observada de errores"),
    ]

    for column, description in checks:
        if column not in df.columns:
            continue

        total = len(df)
        counts = df[column].fillna("NULL").astype(str).value_counts()

        for value, count in counts.items():
            rows.append({
                "metric": description,
                "column": column,
                "value": value,
                "count": int(count),
                "percentage": count / total * 100 if total else None,
            })

    return pd.DataFrame(rows)


def build_automatic_scenario_conclusions(
    scenario_name: str,
    df: pd.DataFrame,
    config_summary: pd.DataFrame,
    validation_table: pd.DataFrame,
) -> list[str]:
    conclusions = []

    conclusions.append(
        f"El escenario {scenario_name} fue analizado de forma independiente para validar comportamiento, configuracion y datos generados."
    )

    if not validation_table.empty:
        statuses = validation_table["validation_status"].astype(str).value_counts().to_dict()
        conclusions.append(
            "Validacion de volumen por fecha: "
            + ", ".join(f"{k}={v}" for k, v in statuses.items())
            + "."
        )

    if "has_error" in df.columns:
        error_rate = pd.to_numeric(df["has_error"], errors="coerce").fillna(0).mean() * 100
        conclusions.append(f"Tasa observada de error: {error_rate:.2f}%.")

    if "storage_cost" in df.columns:
        total_cost = pd.to_numeric(df["storage_cost"], errors="coerce").sum()
        conclusions.append(f"Costo total simulado observado: {total_cost:.6f}.")

    if "size_mb" in df.columns:
        size = pd.to_numeric(df["size_mb"], errors="coerce")
        conclusions.append(
            f"Tamaño promedio observado: {size.mean():.4f} MB; p95={size.quantile(0.95):.4f} MB."
        )

    if "queue_pressure" in df.columns:
        qp = pd.to_numeric(df["queue_pressure"], errors="coerce")
        conclusions.append(
            f"Presion de cola promedio observada: {qp.mean():.4f}; maxima={qp.max():.4f}."
        )

    if not config_summary.empty:
        row = config_summary.iloc[0]
        if "weekly_base_load_sum" in row and pd.notna(row["weekly_base_load_sum"]):
            weekly_sum = float(row["weekly_base_load_sum"])
            if abs(weekly_sum - 1.0) <= 0.001:
                conclusions.append("La distribucion semanal base_load suma correctamente 1.0.")
            else:
                conclusions.append(f"ALERTA: base_load semanal suma {weekly_sum:.4f}, deberia ser 1.0.")

    return conclusions


# ==========================================================
# EXPORTES POR ESCENARIO
# ==========================================================

def export_tables(
    scenario_output_dir: Path,
    tables: dict[str, Any],
    subfolder: str = "tables",
) -> dict[str, str]:
    table_dir = scenario_output_dir / subfolder
    table_dir.mkdir(parents=True, exist_ok=True)

    exported = {}

    for name, table in tables.items():
        if isinstance(table, list):
            table = pd.DataFrame({"conclusion": table})

        if isinstance(table, pd.DataFrame):
            path = table_dir / f"{name}.csv"
            table.to_csv(path, index=False, encoding="utf-8-sig")
            exported[name] = str(path)

    return exported


def build_scenario_report(
    scenario_name: str,
    df: pd.DataFrame,
    descriptive_results: dict,
    data_dictionary_df: pd.DataFrame,
    plots: list[tuple[str, str]],
    output_path: Path,
    relative_plot_dir: str,
    config_summary: pd.DataFrame,
    runtime_config_index: pd.DataFrame,
    file_inventory: pd.DataFrame,
    validation_table: pd.DataFrame,
    observed_parameters: pd.DataFrame,
    scenario_conclusions: list[str],
) -> Path:
    body = ""

    body += to_html_table(
        config_summary,
        "A. Configuracion del escenario",
        "Parametros principales usados para generar este caso.",
        css_class="summary-card",
    )

    body += to_html_table(
        runtime_config_index,
        "B. Configuraciones runtime por fecha",
        "Copias runtime que documentan la fecha ejecutada, semilla y salida usada por cada dia.",
    )

    body += to_html_table(
        file_inventory,
        "C. Inventario de archivos CSV del escenario",
        "Archivos encontrados dentro de output/dataset/<escenario>.",
    )

    body += to_html_table(
        validation_table,
        "D. Validacion de ejecucion esperada vs generada",
        "Compara filas generadas por fecha contra max_valid_files_per_day del JSON.",
        css_class="summary-card",
    )

    body += to_html_table(
        observed_parameters,
        "E. Parametros observados en el dataset",
        "Distribuciones empiricas generadas por tipo de archivo, tier, franja horaria, caso y errores.",
    )

    body += to_html_table(
        scenario_conclusions,
        "F. Conclusiones automaticas del escenario",
        "Lectura ejecutiva del comportamiento observado.",
        css_class="summary-card",
    )

    temp_report = output_path.parent / "_temp_base_report.html"
    build_descriptive_report(
        df=df,
        results=descriptive_results,
        data_dictionary_df=data_dictionary_df,
        key_tables=descriptive_results,
        plots=plots,
        output_path=temp_report,
    )

    temp_html = temp_report.read_text(encoding="utf-8")
    marker_start = temp_html.find("<section")
    marker_end = temp_html.rfind("</body>")
    if marker_start != -1 and marker_end != -1:
        body += temp_html[marker_start:marker_end]

    temp_report.unlink(missing_ok=True)

    html = build_html_page(
        title=f"Reporte exhaustivo por escenario - {scenario_name}",
        body=body,
    )

    html = html.replace("../plots/descriptive", relative_plot_dir)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    return output_path


def fix_advanced_report_plot_paths(report_path: Path) -> None:
    if not report_path or not report_path.exists():
        return

    html = report_path.read_text(encoding="utf-8")
    html = html.replace("../plots/advanced", "../plots/advanced")
    report_path.write_text(html, encoding="utf-8")


def run_advanced_stage_for_scenario(
    df: pd.DataFrame,
    scenario_output_dir: Path,
    html_dir: Path,
    scenario_name: str,
    enabled: bool,
) -> tuple[Path | None, Path | None, dict[str, str]]:
    if not enabled:
        return None, None, {}

    advanced_plot_dir = scenario_output_dir / "plots" / "advanced"
    advanced_table_dir = scenario_output_dir / "tables" / "advanced"
    advanced_plot_dir.mkdir(parents=True, exist_ok=True)
    advanced_table_dir.mkdir(parents=True, exist_ok=True)

    advanced_plot_files = run_advanced_plots(
        df=df,
        output_dir=advanced_plot_dir,
        enabled=True,
    )

    advanced_tables = build_advanced_tables(df)
    exported_advanced_tables = export_tables(
        scenario_output_dir=scenario_output_dir,
        tables=advanced_tables,
        subfolder="tables/advanced",
    )

    advanced_report = build_advanced_report(
        df=df,
        plot_files=advanced_plot_files,
        output_path=html_dir / f"advanced_report_{scenario_name}.html",
        enabled=True,
    )

    if advanced_report:
        fix_advanced_report_plot_paths(advanced_report)

    return advanced_report, advanced_plot_dir, exported_advanced_tables


def run_single_scenario(
    scenario_dir: Path,
    output_root: Path,
    overwrite: bool,
    run_advanced: bool,
) -> dict[str, Any]:
    scenario_name = scenario_dir.name
    scenario_output_dir = output_root / scenario_name

    if scenario_output_dir.exists() and overwrite:
        shutil.rmtree(scenario_output_dir)

    html_dir = scenario_output_dir / "html"
    plot_dir = scenario_output_dir / "plots" / "descriptive"
    table_dir = scenario_output_dir / "tables"

    html_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    df = load_scenario_dataset(scenario_dir)
    df = prepare_dataframe(df)

    data_dictionary_df = build_data_dictionary_table(df)
    descriptive_results = run_descriptive_analysis(df)

    plots = generate_descriptive_plots(
        df=df,
        output_dir=plot_dir,
    )

    config_summary = build_config_summary(scenario_name)
    runtime_config_index = build_runtime_config_index(scenario_name)
    file_inventory = build_file_inventory_table(scenario_dir)
    validation_table = build_execution_validation_table(df, scenario_dir)
    observed_parameters = build_parameter_observed_summary(df)
    scenario_conclusions = build_automatic_scenario_conclusions(
        scenario_name=scenario_name,
        df=df,
        config_summary=config_summary,
        validation_table=validation_table,
    )

    export_tables(
        scenario_output_dir=scenario_output_dir,
        tables={
            "data_dictionary": data_dictionary_df,
            "config_summary": config_summary,
            "runtime_config_index": runtime_config_index,
            "file_inventory": file_inventory,
            "execution_validation": validation_table,
            "observed_parameters": observed_parameters,
            "numeric_summary": descriptive_results.get("numeric_summary"),
            "categorical_summary": descriptive_results.get("categorical_summary"),
            "outlier_summary": descriptive_results.get("outlier_summary"),
            "data_quality": descriptive_results.get("data_quality"),
            "variable_classification": descriptive_results.get("variable_classification"),
            "automatic_conclusions": descriptive_results.get("automatic_conclusions"),
            "scenario_conclusions": scenario_conclusions,
        },
    )

    report_path = html_dir / f"report_{scenario_name}.html"

    build_scenario_report(
        scenario_name=scenario_name,
        df=df,
        descriptive_results=descriptive_results,
        data_dictionary_df=data_dictionary_df,
        plots=plots,
        output_path=report_path,
        relative_plot_dir="../plots/descriptive",
        config_summary=config_summary,
        runtime_config_index=runtime_config_index,
        file_inventory=file_inventory,
        validation_table=validation_table,
        observed_parameters=observed_parameters,
        scenario_conclusions=scenario_conclusions,
    )

    advanced_report, advanced_plot_dir, advanced_tables = run_advanced_stage_for_scenario(
        df=df,
        scenario_output_dir=scenario_output_dir,
        html_dir=html_dir,
        scenario_name=scenario_name,
        enabled=run_advanced,
    )

    return {
        "scenario": scenario_name,
        "rows": len(df),
        "columns": df.shape[1],
        "csv_files": len(list(scenario_dir.glob(DATASET_FILE_PATTERN))),
        "descriptive_report": str(report_path),
        "advanced_report": str(advanced_report) if advanced_report else None,
        "descriptive_plots_dir": str(plot_dir),
        "advanced_plots_dir": str(advanced_plot_dir) if advanced_plot_dir else None,
        "tables_dir": str(table_dir),
        "advanced_tables_count": len(advanced_tables),
        "status": "OK",
    }


def build_index_report(results: list[dict[str, Any]], output_root: Path) -> Path:
    df = pd.DataFrame(results)

    links = []
    for item in results:
        descriptive_report_path = Path(item["descriptive_report"])
        descriptive_relative = descriptive_report_path.relative_to(output_root)

        advanced_link = "No generado"
        if item.get("advanced_report"):
            advanced_report_path = Path(item["advanced_report"])
            advanced_relative = advanced_report_path.relative_to(output_root)
            advanced_link = f"<a href='{advanced_relative.as_posix()}'>Abrir avanzado</a>"

        links.append({
            "scenario": item["scenario"],
            "rows": item["rows"],
            "csv_files": item["csv_files"],
            "descriptive_report": f"<a href='{descriptive_relative.as_posix()}'>Abrir descriptivo</a>",
            "advanced_report": advanced_link,
            "descriptive_plots_dir": item["descriptive_plots_dir"],
            "advanced_plots_dir": item["advanced_plots_dir"] or "",
            "tables_dir": item["tables_dir"],
            "advanced_tables_count": item["advanced_tables_count"],
            "status": item["status"],
        })

    index_df = pd.DataFrame(links)

    body = to_html_table(
        index_df,
        "Indice de reportes por escenario",
        "Cada fila abre el reporte HTML descriptivo y avanzado de un escenario.",
        css_class="summary-card",
    )

    body += to_html_table(
        df,
        "Resumen tecnico de ejecucion",
        "Trazabilidad de filas, columnas, archivos, reportes, graficos y tablas generadas.",
    )

    html = build_html_page(
        title="Indice general - Analisis por escenario",
        body=body,
    )

    html = (
        html.replace("&lt;a href=", "<a href=")
            .replace("&lt;/a&gt;", "</a>")
            .replace("&gt;Abrir descriptivo", ">Abrir descriptivo")
            .replace("&gt;Abrir avanzado", ">Abrir avanzado")
    )

    output_path = output_root / "index_reportes_escenarios.html"
    output_path.write_text(html, encoding="utf-8")
    return output_path


def run_global_report(dataset_root: Path, output_root: Path, run_advanced: bool) -> dict[str, str | None]:
    scenario_dirs = discover_scenario_dataset_folders(dataset_root)
    frames = [load_scenario_dataset(folder) for folder in scenario_dirs]
    df = pd.concat(frames, ignore_index=True)
    df = prepare_dataframe(df)

    global_dir = output_root / "_global"
    html_dir = global_dir / "html"
    descriptive_plot_dir = global_dir / "plots" / "descriptive"
    advanced_plot_dir = global_dir / "plots" / "advanced"

    html_dir.mkdir(parents=True, exist_ok=True)
    descriptive_plot_dir.mkdir(parents=True, exist_ok=True)
    advanced_plot_dir.mkdir(parents=True, exist_ok=True)

    data_dictionary_df = build_data_dictionary_table(df)
    descriptive_results = run_descriptive_analysis(df)
    plots = generate_descriptive_plots(df=df, output_dir=descriptive_plot_dir)

    descriptive_report_path = html_dir / "report_global_all_scenarios.html"
    build_descriptive_report(
        df=df,
        results=descriptive_results,
        data_dictionary_df=data_dictionary_df,
        key_tables=descriptive_results,
        plots=plots,
        output_path=descriptive_report_path,
    )

    advanced_report_path = None

    if run_advanced:
        advanced_plots = run_advanced_plots(
            df=df,
            output_dir=advanced_plot_dir,
            enabled=True,
        )

        advanced_report_path = build_advanced_report(
            df=df,
            plot_files=advanced_plots,
            output_path=html_dir / "advanced_report_global_all_scenarios.html",
            enabled=True,
        )

    return {
        "descriptive_report": str(descriptive_report_path),
        "advanced_report": str(advanced_report_path) if advanced_report_path else None,
    }


# ==========================================================
# MAIN
# ==========================================================

def main() -> None:
    args = parse_args()
    run_advanced = should_run_advanced(args)

    dataset_root = Path(args.dataset_dir)
    output_root = Path(args.output_dir)

    output_root.mkdir(parents=True, exist_ok=True)

    scenario_dirs = discover_scenario_dataset_folders(dataset_root)

    if args.scenario_filter:
        scenario_dirs = [
            folder
            for folder in scenario_dirs
            if args.scenario_filter.lower() in folder.name.lower()
        ]

    if args.max_scenarios is not None:
        scenario_dirs = scenario_dirs[:args.max_scenarios]

    if not scenario_dirs:
        raise RuntimeError("No hay escenarios para analizar con los filtros indicados.")

    print("\n=== ESCENARIOS DETECTADOS PARA ANALISIS ===")
    print(f"Analitica avanzada: {'SI' if run_advanced else 'NO'}")
    for folder in scenario_dirs:
        print(f" - {folder.name}")

    results = []

    for scenario_dir in scenario_dirs:
        print(f"\n[ANALYSIS] Escenario: {scenario_dir.name}")
        result = run_single_scenario(
            scenario_dir=scenario_dir,
            output_root=output_root,
            overwrite=not args.no_overwrite,
            run_advanced=run_advanced,
        )
        results.append(result)

        print(f"[OK] Reporte descriptivo: {result['descriptive_report']}")
        print(f"[OK] Imagenes descriptivas: {result['descriptive_plots_dir']}")
        print(f"[OK] Tablas: {result['tables_dir']}")

        if result.get("advanced_report"):
            print(f"[OK] Reporte avanzado: {result['advanced_report']}")
            print(f"[OK] Imagenes avanzadas: {result['advanced_plots_dir']}")

    index_path = build_index_report(results, output_root)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "advanced_enabled": run_advanced,
        "total_scenarios": len(results),
        "results": results,
        "index_report": str(index_path),
    }

    manifest_path = output_root / "manifest_analysis_by_scenario.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    global_reports = None
    if args.include_global:
        print("\n[GLOBAL] Generando reporte global...")
        global_reports = run_global_report(
            dataset_root=dataset_root,
            output_root=output_root,
            run_advanced=run_advanced,
        )
        print(f"[OK] Reporte global descriptivo: {global_reports['descriptive_report']}")
        if global_reports.get("advanced_report"):
            print(f"[OK] Reporte global avanzado: {global_reports['advanced_report']}")

    print("\n✅ Analisis por escenario completado")
    print(f"✅ Indice general: {index_path}")
    print(f"✅ Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
