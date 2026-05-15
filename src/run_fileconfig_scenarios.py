from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_DIR = BASE_DIR / "config"
SCENARIOS_DIR = CONFIG_DIR / "escenarios"

OUTPUT_BASE_DIR = BASE_DIR / "output"
DATASET_OUTPUT_DIR = OUTPUT_BASE_DIR / "dataset"
RUNTIME_CONFIG_DIR = OUTPUT_BASE_DIR / "runtime_configs"
LOG_DIR = OUTPUT_BASE_DIR / "logs"


# ============================================================
# VARIABLES DEL RANGO DE SIMULACION
# ============================================================
START_DATE = date(2026, 4, 1)
END_DATE = date(2026, 4, 7)


# ============================================================
# UTILIDADES
# ============================================================
def iter_dates(start_date: date, end_date: date):
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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def find_first_json(folder: Path, prefix: str) -> Optional[Path]:
    candidates = sorted(folder.glob(f"{prefix}*.json"))
    return candidates[0] if candidates else None


def discover_scenarios(scenarios_dir: Path = SCENARIOS_DIR) -> List[Dict[str, Path]]:
    if not scenarios_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta de escenarios: {scenarios_dir}")

    scenarios: List[Dict[str, Path]] = []

    for folder in sorted(p for p in scenarios_dir.iterdir() if p.is_dir()):
        simulation_config = find_first_json(folder, "simulation_config")
        time_distribution_config = find_first_json(folder, "time_distribution_config")

        if not simulation_config or not time_distribution_config:
            print(
                f"[WARN] Escenario omitido: {folder.name}. "
                "Falta simulation_config*.json o time_distribution_config*.json"
            )
            continue

        scenarios.append(
            {
                "name": folder.name,
                "folder": folder,
                "simulation_config": simulation_config,
                "time_distribution_config": time_distribution_config,
            }
        )

    if not scenarios:
        raise RuntimeError(f"No se encontraron escenarios validos dentro de {scenarios_dir}")

    return scenarios


def expected_dataset_name(scenario_name: str, simulation_day: date) -> str:
    return f"blob_inventory_{scenario_name}_{simulation_day.strftime('%Y-%m-%d')}.csv"


def prepare_runtime_config(
    scenario_name: str,
    simulation_config_path: Path,
    scenario_output_dir: Path,
    simulation_day: date,
    start_date: date,
    end_date: date,
) -> Path:
    """
    Crea una copia runtime por escenario y dia.

    NO modifica:
      - el JSON original del escenario
      - src.run_simulation.py

    Si run_simulation respeta simulation.output_dir, escribira directamente
    en output/dataset/<escenario>. Luego este script renombra el CSV.
    """
    config = load_json(simulation_config_path)
    day_text = simulation_day.strftime("%Y-%m-%d")

    config.setdefault("simulation", {})
    config["simulation"]["simulation_date"] = day_text
    config["simulation"]["output_dir"] = str(scenario_output_dir.as_posix())

    config.setdefault("scenario_execution", {})
    config["scenario_execution"].update(
        {
            "scenario_name": scenario_name,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "current_simulation_date": day_text,
            "expected_dataset_file": expected_dataset_name(scenario_name, simulation_day),
            "source_config": str(simulation_config_path.as_posix()),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )

    runtime_path = RUNTIME_CONFIG_DIR / scenario_name / day_text / simulation_config_path.name
    write_json(runtime_path, config)
    return runtime_path


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
    """
    Busca CSV generados por run_simulation para el dia.

    Primero revisa output/dataset/<escenario>.
    Luego revisa output/dataset por si el simulador ignora output_dir.
    """
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

    # Quitar duplicados conservando orden
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
    """
    Garantiza que el CSV final quede en:
      output/dataset/<escenario>/blob_inventory_<escenario>_<YYYY-MM-DD>.csv

    Esto corrige dos casos:
      1. El simulador genero el CSV en otra carpeta.
      2. El simulador genero blob_inventory_YYYY-MM-DD.csv sin nombre de escenario.
    """
    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    target_file = scenario_output_dir / expected_dataset_name(scenario_name, simulation_day)

    # Si ya existe con el nombre correcto, no hacer nada.
    if target_file.exists():
        return target_file

    candidates = find_generated_csv_candidates(
        scenario_output_dir=scenario_output_dir,
        simulation_day=simulation_day,
        before_files=before_files,
    )

    if not candidates:
        return None

    # Preferir archivos dentro de la carpeta del escenario.
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
    scenario: Dict[str, Path],
    simulation_day: date,
    start_date: date,
    end_date: date,
    python_executable: str,
    overwrite_day: bool,
) -> Dict[str, object]:
    scenario_name = str(scenario["name"])
    day_text = simulation_day.strftime("%Y-%m-%d")
    scenario_output_dir = DATASET_OUTPUT_DIR / scenario_name

    scenario_output_dir.mkdir(parents=True, exist_ok=True)

    runtime_simulation_config = prepare_runtime_config(
        scenario_name=scenario_name,
        simulation_config_path=scenario["simulation_config"],
        scenario_output_dir=scenario_output_dir,
        simulation_day=simulation_day,
        start_date=start_date,
        end_date=end_date,
    )

    time_distribution_config = scenario["time_distribution_config"]
    target_dataset_file = scenario_output_dir / expected_dataset_name(scenario_name, simulation_day)

    if overwrite_day and target_dataset_file.exists():
        target_dataset_file.unlink()

    # Foto del estado antes de ejecutar para saber que CSV son nuevos.
    before_files = set(OUTPUT_BASE_DIR.rglob("*.csv")) if OUTPUT_BASE_DIR.exists() else set()

    command = [
        python_executable,
        "-m",
        "src.run_simulation",
        "--config",
        str(runtime_simulation_config),
        "--time-distribution",
        str(time_distribution_config),
    ]

    log_file = LOG_DIR / scenario_name / f"{day_text}.log"

    print("=" * 80)
    print(f"[RUN] Escenario: {scenario_name} | Fecha: {day_text}")
    print(f"[CFG] Simulation       : {runtime_simulation_config}")
    print(f"[CFG] Time distribution: {time_distribution_config}")
    print(f"[OUT] Dataset          : {scenario_output_dir}")
    print(f"[CSV] Esperado         : {target_dataset_file}")
    print(f"[LOG] Log              : {log_file}")

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
            "run_simulation termino OK, pero no se encontro CSV generado."
        )
    else:
        print(f"[{status}] {scenario_name} | date={day_text} | return_code={return_code}")

    return {
        "scenario": scenario_name,
        "simulation_date": day_text,
        "status": status,
        "return_code": return_code,
        "runtime_simulation_config": str(runtime_simulation_config),
        "time_distribution_config": str(time_distribution_config),
        "dataset_output_dir": str(scenario_output_dir),
        "dataset_file": str(normalized_file) if normalized_file else None,
        "expected_dataset_file": str(target_dataset_file),
        "log_file": str(log_file),
    }


def write_execution_manifest(
    results: List[Dict[str, object]],
    start_date: date,
    end_date: date,
) -> Path:
    scenarios = sorted({str(item["scenario"]) for item in results})
    dates = sorted({str(item["simulation_date"]) for item in results})

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "base_dir": str(BASE_DIR),
        "scenarios_dir": str(SCENARIOS_DIR),
        "dataset_output_dir": str(DATASET_OUTPUT_DIR),
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "total_scenarios": len(scenarios),
        "total_dates": len(dates),
        "total_runs": len(results),
        "successful": sum(1 for item in results if item["status"] == "OK"),
        "failed": sum(1 for item in results if item["status"] != "OK"),
        "scenarios": scenarios,
        "dates": dates,
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
) -> None:
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

    for scenario in scenarios:
        print(f" - {scenario['name']}")

    python_executable = sys.executable
    results = []

    for scenario in scenarios:
        scenario_output_dir = DATASET_OUTPUT_DIR / str(scenario["name"])

        if scenario_output_dir.exists() and overwrite:
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
    print(f"OK     : {sum(1 for item in results if item['status'] == 'OK')}")
    print(f"ERROR  : {sum(1 for item in results if item['status'] != 'OK')}")
    print("CSV final: output/dataset/<escenario>/blob_inventory_<escenario>_<fecha>.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ejecuta escenarios de config/escenarios por rango de fechas. "
            "Deja los CSV dentro de output/dataset/<escenario> con nombre de escenario."
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
    )
