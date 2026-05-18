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

from analysis_pipeline.advanced_plots import run_advanced_plots
from analysis_pipeline.advanced_html_exports import build_advanced_report
from analysis_pipeline.advanced_tables import build_advanced_tables


# ==========================================================
# CONFIGURACION GENERAL
# ==========================================================

RUN_ADVANCED_DEFAULT = True

DATASET_FILE_PATTERN = "blob_inventory*.csv"

SCENARIO_CONFIG_ROOT = Path("config/escenarios")
GLOBAL_CONFIG_ROOT = Path("config/global")
RUNTIME_CONFIG_ROOT = Path("output/runtime_configs")

GLOBAL_CONFIG_FILES = {
    "time_distribution_config": GLOBAL_CONFIG_ROOT / "time_distribution_config.json",
    "lifecycle_config": GLOBAL_CONFIG_ROOT / "lifecycle_config.json",
    "cost_config": GLOBAL_CONFIG_ROOT / "cost_config.json",
    "error_config": GLOBAL_CONFIG_ROOT / "error_config.json",
    "noise_config": GLOBAL_CONFIG_ROOT / "noise_config.json",
    "realism_config": GLOBAL_CONFIG_ROOT / "realism_config.json",
    "correlation_config": GLOBAL_CONFIG_ROOT / "correlation_config.json",
}


# ==========================================================
# ARGUMENTOS
# ==========================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera reportes descriptivos, avanzados, validaciones y resumenes "
            "comparativos por escenario. Lee datasets desde output/dataset/<escenario>."
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
# UTILIDADES
# ==========================================================

def safe_load_json(path: Path | None) -> dict[str, Any] | None:
    if not path or not path.exists():
        return None

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def safe_numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series([default] * len(df), index=df.index)

    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return float(numerator) / float(denominator)


def count_pct(series: pd.Series, value: Any) -> float:
    if len(series) == 0:
        return 0.0

    return float((series.astype(str) == str(value)).mean() * 100)


def load_global_configs() -> dict[str, dict[str, Any]]:
    return {
        name: safe_load_json(path) or {}
        for name, path in GLOBAL_CONFIG_FILES.items()
    }


def get_nested(raw: MappingLike, path: list[str], default: Any = None) -> Any:
    current: Any = raw

    for key in path:
        if not isinstance(current, dict):
            return default
        current = current.get(key)

    if current is None:
        return default

    return current


MappingLike = dict[str, Any]


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
# CONFIGURACION VS DATASET
# ==========================================================

def find_scenario_config(scenario_name: str) -> Path | None:
    scenario_folder = SCENARIO_CONFIG_ROOT / scenario_name

    if not scenario_folder.exists():
        return None

    return next(iter(sorted(scenario_folder.glob("simulation_config*.json"))), None)


def build_runtime_config_index(scenario_name: str) -> pd.DataFrame:
    runtime_folder = RUNTIME_CONFIG_ROOT / scenario_name
    rows = []

    if not runtime_folder.exists():
        return pd.DataFrame(columns=[
            "runtime_date",
            "runtime_config",
            "simulation_date",
            "selected_day",
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
            "selected_day": simulation.get("selected_day"),
            "max_valid_files_per_day": simulation.get("max_valid_files_per_day"),
            "output_dir": simulation.get("output_dir"),
            "seed": simulation.get("seed"),
        })

    return pd.DataFrame(rows)


def build_config_summary(scenario_name: str) -> pd.DataFrame:
    scenario_config_path = find_scenario_config(scenario_name)
    scenario_config = safe_load_json(scenario_config_path) if scenario_config_path else None
    global_configs = load_global_configs()

    if not scenario_config:
        return pd.DataFrame([{
            "scenario": scenario_name,
            "config_status": "NO_CONFIG_FOUND",
        }])

    simulation = scenario_config.get("simulation", {})
    capacity = scenario_config.get("capacity", {})
    outliers = scenario_config.get("outliers", {})
    arrival = scenario_config.get("arrival_process", {})
    transfer = scenario_config.get("transfer", {}).get("speed_mbps", {})
    file_distribution = scenario_config.get("file_types", {}).get("distribution", {})
    daily_charge = scenario_config.get("daily_charge", {})
    base_load_profiles = scenario_config.get("base_load_profiles", {})

    error_cfg = global_configs["error_config"].get("errors", {})
    lifecycle_cfg = global_configs["lifecycle_config"].get("lifecycle", {})
    cost_cfg = global_configs["cost_config"]
    storage = cost_cfg.get("storage_tier", {})
    cost_model = cost_cfg.get("cost_model", {})
    time_cfg = global_configs["time_distribution_config"]

    weekly_distribution = time_cfg.get("weekly_time_distribution", [])
    weekly_percentage_sum = None
    slot_count = 0

    if weekly_distribution:
        weekly_percentage_sum = sum(
            sum(
                float(slot.get("percentage_load", 0))
                for slot in day.get("time_distribution", [])
            )
            for day in weekly_distribution
        )
        slot_count = sum(len(day.get("time_distribution", [])) for day in weekly_distribution)

    return pd.DataFrame([{
        "scenario": scenario_name,
        "config_status": "OK",
        "simulation_config": str(scenario_config_path),
        "time_distribution_config": str(GLOBAL_CONFIG_FILES["time_distribution_config"]),
        "lifecycle_config": str(GLOBAL_CONFIG_FILES["lifecycle_config"]),
        "cost_config": str(GLOBAL_CONFIG_FILES["cost_config"]),
        "error_config": str(GLOBAL_CONFIG_FILES["error_config"]),
        "noise_config": str(GLOBAL_CONFIG_FILES["noise_config"]),
        "realism_config": str(GLOBAL_CONFIG_FILES["realism_config"]),
        "correlation_config": str(GLOBAL_CONFIG_FILES["correlation_config"]),
        "max_valid_files_per_day": simulation.get("max_valid_files_per_day"),
        "base_simulation_date_in_config": simulation.get("simulation_date"),
        "seed": simulation.get("seed"),
        "arrival_strategy": arrival.get("strategy"),
        "hourly_noise": arrival.get("hourly_noise"),
        "capacity_enabled": capacity.get("enabled"),
        "files_per_hour": capacity.get("files_per_hour"),
        "duration_penalty_factor": capacity.get("duration_penalty_factor"),
        "error_penalty_factor": capacity.get("error_penalty_factor"),
        "min_capacity": capacity.get("min_capacity"),
        "max_capacity": capacity.get("max_capacity"),
        "outliers_enabled": outliers.get("enabled"),
        "outlier_probability": outliers.get("probability"),
        "outlier_multiplier_min": outliers.get("size_multiplier_min"),
        "outlier_multiplier_max": outliers.get("size_multiplier_max"),
        "outlier_max_size_mb": outliers.get("max_size_mb"),
        "transfer_mean_mbps": transfer.get("mean"),
        "transfer_sigma": transfer.get("sigma"),
        "transfer_min_mbps": transfer.get("min"),
        "transfer_max_mbps": transfer.get("max"),
        "base_error_probability": error_cfg.get("base_error_probability"),
        "movement_storage_probability": lifecycle_cfg.get("movement_storage", {}).get("default_probability"),
        "storage_strategy": storage.get("strategy"),
        "storage_cost_formula": cost_model.get("storage_cost_formula"),
        "time_slot_count": slot_count,
        "weekly_percentage_sum": weekly_percentage_sum,
        "daily_charge_json": json.dumps(daily_charge, ensure_ascii=False),
        "base_load_profiles_json": json.dumps(base_load_profiles, ensure_ascii=False),
        "file_type_distribution_json": json.dumps(file_distribution, ensure_ascii=False),
        "tier_cost_json": json.dumps(storage.get("cost_per_mb_per_month", {}), ensure_ascii=False),
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
    expected_max_rows = None

    if not config_summary.empty and "max_valid_files_per_day" in config_summary.columns:
        expected_max_rows = config_summary.iloc[0].get("max_valid_files_per_day")

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
        status = "NO_EXPECTED_MAX"

        if expected_max_rows is not None:
            expected_max_rows_int = int(expected_max_rows)
            diff = generated_rows - expected_max_rows_int
            pct = generated_rows / expected_max_rows_int * 100 if expected_max_rows_int else None

            if generated_rows <= 0:
                status = "NO_ROWS"
            elif generated_rows <= expected_max_rows_int:
                status = "OK"
            else:
                status = "ABOVE_MAX_VALID_FILES"

        rows.append({
            "scenario": scenario_name,
            "simulation_date": row["simulation_date"],
            "expected_max_rows_from_config": expected_max_rows,
            "generated_rows": generated_rows,
            "difference_vs_max": diff,
            "generated_pct_vs_max": pct,
            "validation_status": status,
        })

    return pd.DataFrame(rows)


def build_parameter_observed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    checks = [
        ("file_type", "Distribucion observada por tipo de archivo"),
        ("storage_tier", "Distribucion observada por tier"),
        ("storage_tier_final", "Distribucion observada por tier final"),
        ("time_slot", "Distribucion observada por franja horaria"),
        ("case_type", "Distribucion observada por tipo de caso"),
        ("case_group", "Distribucion observada por grupo de caso"),
        ("error_family", "Distribucion observada por familia de error"),
        ("error_type", "Distribucion observada por tipo de error"),
        ("has_error", "Distribucion observada de errores"),
        ("retry_success", "Distribucion observada de exito en retries"),
        ("storage_policy_noise_applied", "Distribucion observada de ruido de politica de storage"),
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


# ==========================================================
# NUEVAS VALIDACIONES Y RESUMENES
# ==========================================================

def build_dataset_integrity_checks(df: pd.DataFrame, scenario_name: str) -> pd.DataFrame:
    checks = []

    def add_check(name: str, passed_count: int, total_count: int, severity: str = "ERROR") -> None:
        failed_count = total_count - passed_count
        checks.append({
            "scenario": scenario_name,
            "check": name,
            "passed_count": int(passed_count),
            "failed_count": int(failed_count),
            "total_count": int(total_count),
            "pass_rate_pct": safe_ratio(passed_count, total_count) * 100,
            "status": "OK" if failed_count == 0 else "FAIL",
            "severity": severity,
        })

    total = len(df)

    if "queue_pressure" in df.columns:
        qp = safe_numeric(df, "queue_pressure")
        add_check("queue_pressure entre 0 y 1", int(((qp >= 0) & (qp <= 1)).sum()), total)

    if "queue_pressure_raw" in df.columns:
        qpr = safe_numeric(df, "queue_pressure_raw")
        add_check("queue_pressure_raw no negativa", int((qpr >= 0).sum()), total)

    if "error_probability" in df.columns:
        ep = safe_numeric(df, "error_probability")
        add_check("error_probability entre 0 y 0.95", int(((ep >= 0) & (ep <= 0.95)).sum()), total)

    if {"days_since_last_access", "days_stored"}.issubset(df.columns):
        access = safe_numeric(df, "days_since_last_access")
        stored = safe_numeric(df, "days_stored")
        add_check("days_since_last_access <= days_stored", int((access <= stored).sum()), total)

    if "storage_cost" in df.columns:
        storage_cost = safe_numeric(df, "storage_cost")
        add_check("storage_cost no negativo", int((storage_cost >= 0).sum()), total)

    if "total_operational_cost" in df.columns:
        total_cost = safe_numeric(df, "total_operational_cost")
        add_check("total_operational_cost no negativo", int((total_cost >= 0).sum()), total)

    if "transfer_duration_sec" in df.columns:
        duration = safe_numeric(df, "transfer_duration_sec")
        add_check("transfer_duration_sec positivo", int((duration > 0).sum()), total)

    if "transfer_speed_mbps" in df.columns:
        speed = safe_numeric(df, "transfer_speed_mbps")
        add_check("transfer_speed_mbps positivo", int((speed > 0).sum()), total)

    if "retry_count" in df.columns:
        retry_count = safe_numeric(df, "retry_count")
        add_check("retry_count no negativo", int((retry_count >= 0).sum()), total)

    if "size_mb" in df.columns:
        size = safe_numeric(df, "size_mb")
        add_check("size_mb positivo", int((size > 0).sum()), total)

    return pd.DataFrame(checks)


def build_config_vs_observed_table(
    df: pd.DataFrame,
    config_summary: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    rows = []

    if config_summary.empty:
        return pd.DataFrame(rows)

    cfg = config_summary.iloc[0].to_dict()

    configured_distribution = {}
    try:
        configured_distribution = json.loads(cfg.get("file_type_distribution_json") or "{}")
    except Exception:
        configured_distribution = {}

    if "file_type" in df.columns and configured_distribution:
        observed = df["file_type"].astype(str).value_counts(normalize=True).to_dict()

        for file_type, expected_pct in configured_distribution.items():
            observed_pct = observed.get(file_type, 0.0)
            rows.append({
                "scenario": scenario_name,
                "metric": "file_type_distribution",
                "key": file_type,
                "configured_value": float(expected_pct),
                "observed_value": float(observed_pct),
                "absolute_difference": abs(float(expected_pct) - float(observed_pct)),
            })

    if "has_error" in df.columns and cfg.get("base_error_probability") is not None:
        observed_error_rate = safe_numeric(df, "has_error").mean()
        rows.append({
            "scenario": scenario_name,
            "metric": "error_rate_vs_base_probability",
            "key": "has_error",
            "configured_value": float(cfg.get("base_error_probability")),
            "observed_value": float(observed_error_rate),
            "absolute_difference": abs(float(cfg.get("base_error_probability")) - float(observed_error_rate)),
        })

    if "queue_pressure" in df.columns and cfg.get("files_per_hour") is not None:
        rows.append({
            "scenario": scenario_name,
            "metric": "capacity_vs_queue_pressure",
            "key": "mean_queue_pressure",
            "configured_value": float(cfg.get("files_per_hour")),
            "observed_value": float(safe_numeric(df, "queue_pressure").mean()),
            "absolute_difference": None,
        })

    if "size_mb" in df.columns and cfg.get("outlier_probability") is not None:
        size = safe_numeric(df, "size_mb")
        p95 = size.quantile(0.95) if len(size) else 0.0
        observed_outlier_proxy = float((size > p95).mean()) if p95 else 0.0

        rows.append({
            "scenario": scenario_name,
            "metric": "outlier_probability_proxy",
            "key": "size_mb_gt_p95",
            "configured_value": float(cfg.get("outlier_probability")),
            "observed_value": observed_outlier_proxy,
            "absolute_difference": abs(float(cfg.get("outlier_probability")) - observed_outlier_proxy),
        })

    return pd.DataFrame(rows)


def build_scenario_master_summary(
    scenario_name: str,
    df: pd.DataFrame,
    validation_table: pd.DataFrame,
    integrity_checks: pd.DataFrame,
) -> pd.DataFrame:
    total_rows = len(df)

    error_rate = safe_numeric(df, "has_error").mean() if "has_error" in df.columns else 0.0
    retry_rate = (safe_numeric(df, "retry_count") > 0).mean() if "retry_count" in df.columns else 0.0

    size = safe_numeric(df, "size_mb")
    queue = safe_numeric(df, "queue_pressure")
    queue_raw = safe_numeric(df, "queue_pressure_raw")
    storage_cost = safe_numeric(df, "storage_cost")
    total_cost = safe_numeric(df, "total_operational_cost")
    transfer_duration = safe_numeric(df, "transfer_duration_sec")
    error_probability = safe_numeric(df, "error_probability")

    simulation_dates = (
        pd.to_datetime(df["simulation_date"], errors="coerce").dt.date.nunique()
        if "simulation_date" in df.columns
        else None
    )

    failed_integrity_checks = (
        int((integrity_checks["status"] != "OK").sum())
        if not integrity_checks.empty and "status" in integrity_checks.columns
        else 0
    )

    validation_failures = (
        int((validation_table["validation_status"] != "OK").sum())
        if not validation_table.empty and "validation_status" in validation_table.columns
        else 0
    )

    storage_tier = df["storage_tier_final"] if "storage_tier_final" in df.columns else df.get("storage_tier", pd.Series(dtype=str))

    row = {
        "scenario": scenario_name,
        "total_rows": total_rows,
        "simulation_dates": simulation_dates,
        "avg_rows_per_day": safe_ratio(total_rows, simulation_dates or 0),
        "error_rate": error_rate,
        "retry_rate": retry_rate,
        "avg_queue_pressure": queue.mean() if len(queue) else 0.0,
        "max_queue_pressure": queue.max() if len(queue) else 0.0,
        "avg_queue_pressure_raw": queue_raw.mean() if len(queue_raw) else 0.0,
        "max_queue_pressure_raw": queue_raw.max() if len(queue_raw) else 0.0,
        "avg_error_probability": error_probability.mean() if len(error_probability) else 0.0,
        "max_error_probability": error_probability.max() if len(error_probability) else 0.0,
        "avg_size_mb": size.mean() if len(size) else 0.0,
        "p95_size_mb": size.quantile(0.95) if len(size) else 0.0,
        "max_size_mb": size.max() if len(size) else 0.0,
        "avg_transfer_duration_sec": transfer_duration.mean() if len(transfer_duration) else 0.0,
        "p95_transfer_duration_sec": transfer_duration.quantile(0.95) if len(transfer_duration) else 0.0,
        "storage_cost_total": storage_cost.sum() if len(storage_cost) else 0.0,
        "total_operational_cost": total_cost.sum() if len(total_cost) else 0.0,
        "hot_pct": count_pct(storage_tier, "hot"),
        "cool_pct": count_pct(storage_tier, "cool"),
        "archive_pct": count_pct(storage_tier, "archive"),
        "validation_failures": validation_failures,
        "integrity_failures": failed_integrity_checks,
    }

    return pd.DataFrame([row])


def build_global_rankings(master_summary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if master_summary.empty:
        return {}

    rankings = {}

    ranking_specs = {
        "top_error_scenarios": "error_rate",
        "top_retry_scenarios": "retry_rate",
        "top_cost_scenarios": "total_operational_cost",
        "top_queue_pressure_scenarios": "max_queue_pressure",
        "top_large_file_scenarios": "p95_size_mb",
        "top_transfer_duration_scenarios": "p95_transfer_duration_sec",
    }

    for name, column in ranking_specs.items():
        if column in master_summary.columns:
            rankings[name] = (
                master_summary.sort_values(column, ascending=False)
                .head(20)
                .reset_index(drop=True)
            )

    return rankings


def build_automatic_scenario_conclusions(
    scenario_name: str,
    df: pd.DataFrame,
    config_summary: pd.DataFrame,
    validation_table: pd.DataFrame,
    integrity_checks: pd.DataFrame,
    scenario_summary: pd.DataFrame,
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

    if not integrity_checks.empty:
        failed_checks = int((integrity_checks["status"] != "OK").sum())
        conclusions.append(f"Validaciones de integridad fallidas: {failed_checks}.")

    if "has_error" in df.columns:
        error_rate = safe_numeric(df, "has_error").mean() * 100
        conclusions.append(f"Tasa observada de error: {error_rate:.2f}%.")

    if "retry_count" in df.columns:
        retry_rate = (safe_numeric(df, "retry_count") > 0).mean() * 100
        conclusions.append(f"Tasa observada de retries: {retry_rate:.2f}%.")

    if "total_operational_cost" in df.columns:
        total_cost = safe_numeric(df, "total_operational_cost").sum()
        conclusions.append(f"Costo operacional total simulado observado: {total_cost:.6f}.")

    elif "storage_cost" in df.columns:
        total_cost = safe_numeric(df, "storage_cost").sum()
        conclusions.append(f"Costo total de almacenamiento simulado observado: {total_cost:.6f}.")

    if "size_mb" in df.columns:
        size = safe_numeric(df, "size_mb")
        conclusions.append(
            f"Tamaño promedio observado: {size.mean():.4f} MB; p95={size.quantile(0.95):.4f} MB."
        )

    if "queue_pressure" in df.columns:
        qp = safe_numeric(df, "queue_pressure")
        conclusions.append(
            f"Presion de cola promedio observada: {qp.mean():.4f}; maxima={qp.max():.4f}."
        )

    if not scenario_summary.empty:
        row = scenario_summary.iloc[0]
        if row.get("integrity_failures", 0) == 0:
            conclusions.append("El dataset no presenta fallas de integridad en las reglas criticas evaluadas.")

    return conclusions


# ==========================================================
# EXPORTES
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
    integrity_checks: pd.DataFrame,
    config_vs_observed: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    scenario_conclusions: list[str],
) -> Path:
    body = ""

    body += to_html_table(
        scenario_summary,
        "A. Resumen maestro del escenario",
        "Metricas clave agregadas por escenario: errores, retries, costos, tamanos, colas y validaciones.",
        css_class="summary-card",
    )

    body += to_html_table(
        config_summary,
        "B. Configuracion del escenario y globales",
        "Parametros principales usados para generar este caso, incluyendo archivos globales.",
        css_class="summary-card",
    )

    body += to_html_table(
        runtime_config_index,
        "C. Configuraciones runtime por fecha",
        "Copias runtime que documentan fecha ejecutada, semilla y salida usada por cada dia.",
    )

    body += to_html_table(
        file_inventory,
        "D. Inventario de archivos CSV del escenario",
        "Archivos encontrados dentro de output/dataset/<escenario>.",
    )

    body += to_html_table(
        validation_table,
        "E. Validacion de ejecucion esperada vs generada",
        "Compara filas generadas por fecha contra max_valid_files_per_day como limite superior.",
        css_class="summary-card",
    )

    body += to_html_table(
        integrity_checks,
        "F. Validaciones de integridad del dataset",
        "Reglas criticas sobre rangos, costos, colas, errores, lifecycle y transferencia.",
        css_class="summary-card",
    )

    body += to_html_table(
        config_vs_observed,
        "G. Comparacion config vs observado",
        "Contrasta parametros configurados contra distribuciones empiricas observadas.",
    )

    body += to_html_table(
        observed_parameters,
        "H. Parametros observados en el dataset",
        "Distribuciones empiricas por tipo de archivo, tier, franja horaria, caso, errores y retries.",
    )

    body += to_html_table(
        scenario_conclusions,
        "I. Conclusiones automaticas del escenario",
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
    integrity_checks = build_dataset_integrity_checks(df, scenario_name)
    config_vs_observed = build_config_vs_observed_table(df, config_summary, scenario_name)
    scenario_summary = build_scenario_master_summary(
        scenario_name=scenario_name,
        df=df,
        validation_table=validation_table,
        integrity_checks=integrity_checks,
    )
    scenario_conclusions = build_automatic_scenario_conclusions(
        scenario_name=scenario_name,
        df=df,
        config_summary=config_summary,
        validation_table=validation_table,
        integrity_checks=integrity_checks,
        scenario_summary=scenario_summary,
    )

    export_tables(
        scenario_output_dir=scenario_output_dir,
        tables={
            "scenario_master_summary": scenario_summary,
            "data_dictionary": data_dictionary_df,
            "config_summary": config_summary,
            "runtime_config_index": runtime_config_index,
            "file_inventory": file_inventory,
            "execution_validation": validation_table,
            "dataset_integrity_checks": integrity_checks,
            "config_vs_observed": config_vs_observed,
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
        integrity_checks=integrity_checks,
        config_vs_observed=config_vs_observed,
        scenario_summary=scenario_summary,
        scenario_conclusions=scenario_conclusions,
    )

    advanced_report, advanced_plot_dir, advanced_tables = run_advanced_stage_for_scenario(
        df=df,
        scenario_output_dir=scenario_output_dir,
        html_dir=html_dir,
        scenario_name=scenario_name,
        enabled=run_advanced,
    )

    summary_row = scenario_summary.iloc[0].to_dict() if not scenario_summary.empty else {}

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
        **summary_row,
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
            "error_rate": item.get("error_rate"),
            "retry_rate": item.get("retry_rate"),
            "total_operational_cost": item.get("total_operational_cost"),
            "max_queue_pressure": item.get("max_queue_pressure"),
            "integrity_failures": item.get("integrity_failures"),
            "descriptive_report": f"<a href='{descriptive_relative.as_posix()}'>Abrir descriptivo</a>",
            "advanced_report": advanced_link,
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
        "Trazabilidad de filas, columnas, archivos, reportes, graficos, tablas y metricas generadas.",
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


def export_global_analysis_tables(results: list[dict[str, Any]], output_root: Path) -> dict[str, str]:
    global_table_dir = output_root / "_global" / "tables"
    global_table_dir.mkdir(parents=True, exist_ok=True)

    master_summary = pd.DataFrame(results)
    summary_cols = [
        "scenario",
        "total_rows",
        "simulation_dates",
        "avg_rows_per_day",
        "error_rate",
        "retry_rate",
        "avg_queue_pressure",
        "max_queue_pressure",
        "avg_queue_pressure_raw",
        "max_queue_pressure_raw",
        "avg_error_probability",
        "max_error_probability",
        "avg_size_mb",
        "p95_size_mb",
        "max_size_mb",
        "avg_transfer_duration_sec",
        "p95_transfer_duration_sec",
        "storage_cost_total",
        "total_operational_cost",
        "hot_pct",
        "cool_pct",
        "archive_pct",
        "validation_failures",
        "integrity_failures",
    ]

    available_cols = [col for col in summary_cols if col in master_summary.columns]
    scenario_summary = master_summary[available_cols].copy()

    exports = {}
    scenario_summary_path = global_table_dir / "scenario_summary.csv"
    scenario_summary.to_csv(scenario_summary_path, index=False, encoding="utf-8-sig")
    exports["scenario_summary"] = str(scenario_summary_path)

    rankings = build_global_rankings(scenario_summary)
    for name, table in rankings.items():
        path = global_table_dir / f"{name}.csv"
        table.to_csv(path, index=False, encoding="utf-8-sig")
        exports[name] = str(path)

    return exports


def build_global_comparison_report(
    results: list[dict[str, Any]],
    output_root: Path,
    global_exports: dict[str, str],
) -> Path:
    html_dir = output_root / "_global" / "html"
    html_dir.mkdir(parents=True, exist_ok=True)

    scenario_summary = pd.read_csv(global_exports["scenario_summary"])

    body = to_html_table(
        scenario_summary,
        "Resumen maestro comparativo de escenarios",
        "Comparacion consolidada de filas, errores, retries, costos, colas, tamanos y validaciones.",
        css_class="summary-card",
    )

    for key, path in global_exports.items():
        if key == "scenario_summary":
            continue

        table = pd.read_csv(path)
        body += to_html_table(
            table,
            key.replace("_", " ").title(),
            f"Ranking global generado desde {key}.",
        )

    html = build_html_page(
        title="Reporte comparativo global de escenarios",
        body=body,
    )

    output_path = html_dir / "scenario_comparison_report.html"
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
    global_exports = export_global_analysis_tables(results, output_root)
    comparison_report = build_global_comparison_report(
        results=results,
        output_root=output_root,
        global_exports=global_exports,
    )

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "advanced_enabled": run_advanced,
        "total_scenarios": len(results),
        "results": results,
        "index_report": str(index_path),
        "comparison_report": str(comparison_report),
        "global_tables": global_exports,
    }

    global_reports = None
    if args.include_global:
        print("\n[GLOBAL] Generando reporte global...")
        global_reports = run_global_report(
            dataset_root=dataset_root,
            output_root=output_root,
            run_advanced=run_advanced,
        )
        manifest["global_descriptive_report"] = global_reports["descriptive_report"]
        manifest["global_advanced_report"] = global_reports["advanced_report"]

        print(f"[OK] Reporte global descriptivo: {global_reports['descriptive_report']}")
        if global_reports.get("advanced_report"):
            print(f"[OK] Reporte global avanzado: {global_reports['advanced_report']}")

    manifest_path = output_root / "manifest_analysis_by_scenario.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nAnalisis por escenario completado")
    print(f"Indice general: {index_path}")
    print(f"Reporte comparativo: {comparison_report}")
    print(f"Resumen maestro: {global_exports.get('scenario_summary')}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
