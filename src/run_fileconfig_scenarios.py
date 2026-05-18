from __future__ import annotations

import argparse
import copy
import json
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_DIR = BASE_DIR / "config"
GLOBAL_CONFIG_DIR = CONFIG_DIR / "global"
SCENARIOS_DIR = CONFIG_DIR / "escenarios"

OUTPUT_BASE_DIR = BASE_DIR / "output"
DATASET_OUTPUT_DIR = OUTPUT_BASE_DIR / "dataset"
RUNTIME_CONFIG_DIR = OUTPUT_BASE_DIR / "runtime_configs"
LOG_DIR = OUTPUT_BASE_DIR / "logs"

GLOBAL_CONFIG_FILES = {
    "simulation_config": GLOBAL_CONFIG_DIR / "simulation_config.json",
    "time_distribution_config": GLOBAL_CONFIG_DIR / "time_distribution_config.json",
    "lifecycle_config": GLOBAL_CONFIG_DIR / "lifecycle_config.json",
    "cost_config": GLOBAL_CONFIG_DIR / "cost_config.json",
    "error_config": GLOBAL_CONFIG_DIR / "error_config.json",
    "noise_config": GLOBAL_CONFIG_DIR / "noise_config.json",
    "realism_config": GLOBAL_CONFIG_DIR / "realism_config.json",
    "correlation_config": GLOBAL_CONFIG_DIR / "correlation_config.json",
}

START_DATE = date(2026, 1, 1)
END_DATE = date(2026, 1, 16)


# ============================================================
# UTILIDADES BASE
# ============================================================

def iter_dates(start_date: date, end_date: date) -> Iterable[date]:
    if end_date < start_date:
        raise ValueError("END_DATE no puede ser menor que START_DATE.")

    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Fecha invalida: {value}. Usa formato YYYY-MM-DD.") from exc


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo JSON: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def find_first_json(folder: Path, prefix: str) -> Optional[Path]:
    candidates = sorted(folder.glob(f"{prefix}*.json"))
    return candidates[0] if candidates else None


def deep_set(target: dict[str, Any], dotted_path: str, value: Any) -> None:
    """
    Soporta paths tipo:
    - noise.active_profile
    - realism.incidents.probability_per_simulated_day
    - storage_tier.rules.2.min_days_since_last_access
    """
    parts = dotted_path.split(".")
    current: Any = target

    for part in parts[:-1]:
        if isinstance(current, list):
            index = int(part)
            current = current[index]
            continue

        if part not in current or not isinstance(current[part], (dict, list)):
            current[part] = {}

        current = current[part]

    last = parts[-1]

    if isinstance(current, list):
        current[int(last)] = value
    else:
        current[last] = value


def normalize_config_key(file_key: str) -> str:
    if file_key.endswith(".json"):
        file_key = file_key[:-5]

    return file_key


def split_override_path(override_path: str) -> tuple[str, str]:
    """
    Convierte:
      simulation_config.simulation.seed
    en:
      ("simulation_config", "simulation.seed")
    """
    parts = override_path.split(".", 1)

    if len(parts) != 2:
        raise ValueError(
            f"Override invalido: {override_path}. "
            "Debe tener forma '<config_file>.<ruta.interna>'."
        )

    return normalize_config_key(parts[0]), parts[1]


# ============================================================
# VALIDACION Y DESCUBRIMIENTO
# ============================================================

def validate_global_configs() -> None:
    missing = [path for path in GLOBAL_CONFIG_FILES.values() if not path.exists()]

    if missing:
        missing_text = "\n".join(f" - {path}" for path in missing)
        raise FileNotFoundError(
            "Faltan archivos globales de configuracion:\n"
            f"{missing_text}"
        )


def read_flat_scenario_file(path: Path) -> Optional[dict[str, Any]]:
    """
    Acepta escenarios planos con esta forma:

    {
      "scenario": {
        "scenario_id": "20_caos_controlado",
        "seed": 61,
        "overrides": {
          "simulation_config.simulation.seed": 61,
          "noise_config.noise.active_profile": "chaotic"
        }
      }
    }

    Tambien acepta forma corta:

    {
      "scenario_id": "20_caos_controlado",
      "seed": 61,
      "overrides": {}
    }
    """
    if path.name == "scenario_catalog.json":
        return None

    raw = load_json(path)

    scenario_raw = raw.get("scenario", raw)

    scenario_id = (
        scenario_raw.get("scenario_id")
        or scenario_raw.get("name")
        or path.stem.replace("_config", "")
    )

    if not scenario_id:
        return None

    return {
        "name": scenario_id,
        "mode": "flat_json",
        "source": path,
        "folder": path.parent,
        "scenario_config": path,
        "overrides": scenario_raw.get("overrides", {}),
        "description": scenario_raw.get("description"),
        "seed": scenario_raw.get("seed"),
    }


def read_folder_scenario(folder: Path) -> Optional[dict[str, Any]]:
    """
    Compatibilidad con estructura antigua:

    config/escenarios/<scenario>/
      simulation_config.json
    """
    simulation_config = find_first_json(folder, "simulation_config")

    if not simulation_config:
        print(
            f"[WARN] Escenario omitido: {folder.name}. "
            "Falta simulation_config*.json"
        )
        return None

    return {
        "name": folder.name,
        "mode": "folder",
        "source": simulation_config,
        "folder": folder,
        "simulation_config": simulation_config,
        "overrides": {},
        "description": None,
        "seed": None,
    }


def discover_scenarios(scenarios_dir: Path = SCENARIOS_DIR) -> List[dict[str, Any]]:
    if not scenarios_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de escenarios: {scenarios_dir}")

    scenarios: List[dict[str, Any]] = []

    # 1. Soporte moderno: JSON planos.
    for file in sorted(scenarios_dir.glob("*_config.json")):
        item = read_flat_scenario_file(file)
        if item:
            scenarios.append(item)

    # 2. Soporte legacy: carpetas por escenario.
    for folder in sorted(p for p in scenarios_dir.iterdir() if p.is_dir()):
        item = read_folder_scenario(folder)
        if item:
            # Evitar duplicado si existe plano y carpeta con el mismo nombre.
            if not any(str(s["name"]) == str(item["name"]) for s in scenarios):
                scenarios.append(item)

    if not scenarios:
        raise RuntimeError(
            f"No se encontraron escenarios validos dentro de {scenarios_dir}. "
            "Se aceptan archivos *_config.json o carpetas con simulation_config*.json."
        )

    return scenarios


# ============================================================
# RUNTIME CONFIG BUILDER
# ============================================================

def load_global_config_bundle() -> dict[str, dict[str, Any]]:
    return {
        key: load_json(path)
        for key, path in GLOBAL_CONFIG_FILES.items()
    }


def apply_scenario_overrides(
    configs: dict[str, dict[str, Any]],
    overrides: dict[str, Any],
) -> None:
    for override_path, value in overrides.items():
        config_key, internal_path = split_override_path(override_path)

        if config_key not in configs:
            raise KeyError(
                f"Override apunta a config desconocida: {config_key}. "
                f"Override: {override_path}"
            )

        deep_set(configs[config_key], internal_path, value)


def ensure_run_simulation_compatibility(
    simulation_config: dict[str, Any],
    scenario_name: str,
    scenario_description: Optional[str],
    simulation_day: date,
    scenario_output_dir: Path,
    start_date: date,
    end_date: date,
) -> None:
    """
    Mantiene compatibilidad con src.run_simulation.

    Ese modulo espera:
    - simulation
    - daily_charge
    - base_load_profiles
    - arrival_process
    - capacity
    - file_types
    - transfer
    """
    day_text = simulation_day.strftime("%Y-%m-%d")

    simulation_config.setdefault("simulation", {})
    simulation_config["simulation"]["output_dir"] = str(scenario_output_dir.as_posix())

    simulation_config.setdefault("scenario", {})
    simulation_config["scenario"]["name"] = scenario_name
    if scenario_description:
        simulation_config["scenario"]["description"] = scenario_description

    simulation_config.setdefault("scenario_execution", {})
    simulation_config["scenario_execution"].update(
        {
            "scenario_name": scenario_name,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "current_simulation_date": day_text,
            "expected_dataset_file": expected_dataset_name(scenario_name, simulation_day),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    required_sections = [
        "daily_charge",
        "base_load_profiles",
        "arrival_process",
        "capacity",
        "file_types",
        "transfer",
    ]

    missing = [
        section
        for section in required_sections
        if section not in simulation_config
    ]

    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(
            f"simulation_config del escenario '{scenario_name}' no contiene: {missing_text}. "
            "Agrega estos bloques en config/global/simulation_config.json o por override."
        )


def expected_dataset_name(scenario_name: str, simulation_day: date) -> str:
    return f"blob_inventory_{scenario_name}_{simulation_day.strftime('%Y-%m-%d')}.csv"


def prepare_runtime_configs(
    scenario: dict[str, Any],
    simulation_day: date,
    start_date: date,
    end_date: date,
) -> dict[str, Path]:
    scenario_name = str(scenario["name"])
    scenario_output_dir = DATASET_OUTPUT_DIR / scenario_name

    runtime_dir = RUNTIME_CONFIG_DIR / scenario_name / simulation_day.strftime("%Y-%m-%d")
    runtime_dir.mkdir(parents=True, exist_ok=True)

    if scenario["mode"] == "folder":
        # Legacy: toma simulation_config del folder y globales para el resto.
        configs = load_global_config_bundle()
        configs["simulation_config"] = load_json(Path(scenario["simulation_config"]))
    else:
        # Moderno: parte de global y aplica overrides del JSON plano.
        configs = load_global_config_bundle()
        apply_scenario_overrides(configs, scenario.get("overrides", {}))

    ensure_run_simulation_compatibility(
        simulation_config=configs["simulation_config"],
        scenario_name=scenario_name,
        scenario_description=scenario.get("description"),
        simulation_day=simulation_day,
        scenario_output_dir=scenario_output_dir,
        start_date=start_date,
        end_date=end_date,
    )

    runtime_paths: dict[str, Path] = {}

    for config_key, data in configs.items():
        file_name = f"{config_key}.json"
        output_path = runtime_dir / file_name
        write_json(output_path, data)
        runtime_paths[config_key] = output_path

    return runtime_paths


# ============================================================
# EJECUCION
# ============================================================

def run_command(command: List[str], log_file: Path) -> int:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    with log_file.open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command,
            cwd=BASE_DIR,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return process.returncode


def find_generated_csv_candidates(
    scenario_output_dir: Path,
    simulation_day: date,
    before_files: set[Path],
) -> List[Path]:
    day_text = simulation_day.strftime("%Y-%m-%d")

    search_roots = [
        scenario_output_dir,
        DATASET_OUTPUT_DIR,
        OUTPUT_BASE_DIR,
    ]

    candidates: List[Path] = []

    for root in search_roots:
        if not root.exists():
            continue

        for csv_file in root.rglob("*.csv"):
            if csv_file in before_files:
                continue

            name_lower = csv_file.name.lower()
            if day_text in csv_file.name and "blob_inventory" in name_lower:
                candidates.append(csv_file)

    unique_candidates = []
    seen = set()

    for item in candidates:
        resolved = item.resolve()
        if resolved not in seen:
            unique_candidates.append(item)
            seen.add(resolved)

    return unique_candidates


def normalize_generated_dataset_file(
    scenario_name: str,
    simulation_day: date,
    scenario_output_dir: Path,
    before_files: set[Path],
) -> Optional[Path]:
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    target_file = scenario_output_dir / expected_dataset_name(scenario_name, simulation_day)

    if target_file.exists():
        return target_file

    candidates = find_generated_csv_candidates(
        scenario_output_dir=scenario_output_dir,
        simulation_day=simulation_day,
        before_files=before_files,
    )

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda p: (
            0 if scenario_output_dir in p.parents or p.parent == scenario_output_dir else 1,
            len(str(p)),
            str(p),
        ),
    )

    source_file = candidates[0]

    if source_file.resolve() == target_file.resolve():
        return target_file

    if target_file.exists():
        target_file.unlink()

    shutil.move(str(source_file), str(target_file))
    return target_file


def execute_scenario_day(
    scenario: dict[str, Any],
    simulation_day: date,
    start_date: date,
    end_date: date,
    python_executable: str,
    overwrite_day: bool,
    dry_run: bool = False,
) -> dict[str, object]:
    scenario_name = str(scenario["name"])
    day_text = simulation_day.strftime("%Y-%m-%d")
    scenario_output_dir = DATASET_OUTPUT_DIR / scenario_name
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    runtime_paths = prepare_runtime_configs(
        scenario=scenario,
        simulation_day=simulation_day,
        start_date=start_date,
        end_date=end_date,
    )

    target_dataset_file = scenario_output_dir / expected_dataset_name(
        scenario_name,
        simulation_day,
    )

    if overwrite_day and target_dataset_file.exists():
        target_dataset_file.unlink()

    before_files = set(OUTPUT_BASE_DIR.rglob("*.csv")) if OUTPUT_BASE_DIR.exists() else set()

    command = [
        python_executable,
        "-m",
        "src.run_simulation",
        "--simulation-config",
        str(runtime_paths["simulation_config"]),
        "--time-distribution",
        str(runtime_paths["time_distribution_config"]),
        "--lifecycle-config",
        str(runtime_paths["lifecycle_config"]),
        "--cost-config",
        str(runtime_paths["cost_config"]),
        "--error-config",
        str(runtime_paths["error_config"]),
        "--noise-config",
        str(runtime_paths["noise_config"]),
        "--realism-config",
        str(runtime_paths["realism_config"]),
        "--correlation-config",
        str(runtime_paths["correlation_config"]),
        "--simulation-date",
        day_text,
    ]

    log_file = LOG_DIR / scenario_name / f"{day_text}.log"

    print("=" * 80)
    print(f"[RUN] Escenario: {scenario_name} | Fecha: {day_text}")
    print(f"[MODE] {scenario.get('mode')}")
    print(f"[SRC]  {scenario.get('source')}")
    print(f"[CFG] Runtime dir: {runtime_paths['simulation_config'].parent}")
    print(f"[OUT] Dataset    : {scenario_output_dir}")
    print(f"[CSV] Esperado   : {target_dataset_file}")
    print(f"[LOG] Log        : {log_file}")

    if dry_run:
        print("[DRY-RUN] Comando:")
        print(" ".join(str(part) for part in command))
        return {
            "scenario": scenario_name,
            "simulation_date": day_text,
            "status": "DRY_RUN",
            "return_code": None,
            "runtime_config_dir": str(runtime_paths["simulation_config"].parent),
            "dataset_file": None,
            "expected_dataset_file": str(target_dataset_file),
            "log_file": str(log_file),
        }

    return_code = run_command(command, log_file)

    normalized_file = None
    if return_code == 0:
        normalized_file = normalize_generated_dataset_file(
            scenario_name=scenario_name,
            simulation_day=simulation_day,
            scenario_output_dir=scenario_output_dir,
            before_files=before_files,
        )

    status = "OK" if return_code == 0 and normalized_file is not None else "ERROR"

    if return_code == 0 and normalized_file is None:
        print(
            f"[ERROR] {scenario_name} | date={day_text} | "
            "src.run_simulation termino OK, pero no se encontro CSV generado."
        )
    else:
        print(f"[{status}] {scenario_name} | date={day_text} | return_code={return_code}")

    return {
        "scenario": scenario_name,
        "simulation_date": day_text,
        "status": status,
        "return_code": return_code,
        "runtime_config_dir": str(runtime_paths["simulation_config"].parent),
        "simulation_config": str(runtime_paths["simulation_config"]),
        "time_distribution_config": str(runtime_paths["time_distribution_config"]),
        "lifecycle_config": str(runtime_paths["lifecycle_config"]),
        "cost_config": str(runtime_paths["cost_config"]),
        "error_config": str(runtime_paths["error_config"]),
        "noise_config": str(runtime_paths["noise_config"]),
        "realism_config": str(runtime_paths["realism_config"]),
        "correlation_config": str(runtime_paths["correlation_config"]),
        "dataset_output_dir": str(scenario_output_dir),
        "dataset_file": str(normalized_file) if normalized_file else None,
        "expected_dataset_file": str(target_dataset_file),
        "log_file": str(log_file),
    }


def write_execution_manifest(
    results: List[dict[str, object]],
    start_date: date,
    end_date: date,
) -> Path:
    scenarios = sorted({str(item["scenario"]) for item in results})
    dates = sorted({str(item["simulation_date"]) for item in results})

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(BASE_DIR),
        "scenarios_dir": str(SCENARIOS_DIR),
        "global_config_dir": str(GLOBAL_CONFIG_DIR),
        "runtime_config_dir": str(RUNTIME_CONFIG_DIR),
        "dataset_output_dir": str(DATASET_OUTPUT_DIR),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "total_scenarios": len(scenarios),
        "total_dates": len(dates),
        "total_runs": len(results),
        "successful": sum(1 for item in results if item["status"] == "OK"),
        "failed": sum(1 for item in results if item["status"] == "ERROR"),
        "dry_run": sum(1 for item in results if item["status"] == "DRY_RUN"),
        "scenarios": scenarios,
        "dates": dates,
        "global_configs": {
            key: str(path)
            for key, path in GLOBAL_CONFIG_FILES.items()
        },
        "results": results,
    }

    manifest_path = OUTPUT_BASE_DIR / "manifest_dataset_generation.json"
    write_json(manifest_path, manifest)
    return manifest_path


def run_all_scenarios(
    start_date: date = START_DATE,
    end_date: date = END_DATE,
    max_scenarios: Optional[int] = None,
    scenario_filter: Optional[str] = None,
    overwrite: bool = True,
    dry_run: bool = False,
) -> None:
    validate_global_configs()

    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    DATASET_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RUNTIME_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    scenarios = discover_scenarios()

    if scenario_filter:
        scenarios = [
            scenario
            for scenario in scenarios
            if scenario_filter.lower() in str(scenario["name"]).lower()
        ]

    if max_scenarios is not None:
        scenarios = scenarios[:max_scenarios]

    if not scenarios:
        raise RuntimeError("No hay escenarios para ejecutar con los filtros indicados.")

    print(f"Rango de fechas: {start_date} -> {end_date}")
    print(f"Escenarios detectados: {len(scenarios)}")
    print(f"Config global: {GLOBAL_CONFIG_DIR}")

    for scenario in scenarios:
        print(f" - {scenario['name']} ({scenario.get('mode')})")

    python_executable = sys.executable
    results: List[dict[str, object]] = []

    for scenario in scenarios:
        scenario_output_dir = DATASET_OUTPUT_DIR / str(scenario["name"])

        if scenario_output_dir.exists() and overwrite and not dry_run:
            shutil.rmtree(scenario_output_dir)

        scenario_output_dir.mkdir(parents=True, exist_ok=True)

        for simulation_day in iter_dates(start_date, end_date):
            result = execute_scenario_day(
                scenario=scenario,
                simulation_day=simulation_day,
                start_date=start_date,
                end_date=end_date,
                python_executable=python_executable,
                overwrite_day=overwrite,
                dry_run=dry_run,
            )
            results.append(result)

    manifest_path = write_execution_manifest(
        results=results,
        start_date=start_date,
        end_date=end_date,
    )

    print("=" * 80)
    print("Proceso finalizado.")
    print(f"Manifest: {manifest_path}")
    print(f"OK      : {sum(1 for item in results if item['status'] == 'OK')}")
    print(f"ERROR   : {sum(1 for item in results if item['status'] == 'ERROR')}")
    print(f"DRY-RUN : {sum(1 for item in results if item['status'] == 'DRY_RUN')}")
    print("CSV final: output/dataset/<escenario>/blob_inventory_<escenario>_<fecha>.csv")


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Runner unificado de escenarios. "
            "Soporta config/escenarios/*.json y carpetas legacy con simulation_config*.json."
        )
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Opcional. Reemplaza START_DATE. Formato YYYY-MM-DD.",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Opcional. Reemplaza END_DATE. Formato YYYY-MM-DD.",
    )

    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Limita cantidad de escenarios. Util para pruebas rapidas.",
    )

    parser.add_argument(
        "--scenario-filter",
        type=str,
        default=None,
        help="Ejecuta solo escenarios cuyo nombre contenga este texto.",
    )

    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="No elimina output/dataset/<escenario> antes de ejecutar.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo prepara runtime configs y muestra comandos. No ejecuta simulacion.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    selected_start_date = parse_iso_date(args.start_date) if args.start_date else START_DATE
    selected_end_date = parse_iso_date(args.end_date) if args.end_date else END_DATE

    run_all_scenarios(
        start_date=selected_start_date,
        end_date=selected_end_date,
        max_scenarios=args.max_scenarios,
        scenario_filter=args.scenario_filter,
        overwrite=not args.no_overwrite,
        dry_run=args.dry_run,
    )
