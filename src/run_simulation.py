from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from simulator.config_models import DailyUniverseConfig, ErrorFamily, ErrorSubtype
from simulator.orchestrator import SimulationOrchestrator


def load_configuration(path: str) -> tuple[DailyUniverseConfig, list[ErrorFamily]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))

    config = DailyUniverseConfig(
        simulation_date=raw["simulation"]["simulation_date"],
        max_valid_files_per_day=raw["simulation"]["max_valid_files_per_day"],
        global_error_percentage=raw["simulation"]["global_error_percentage"],
        output_dir=raw["simulation"]["output_dir"],
        generate_pdf=raw["simulation"].get("generate_pdf", True),
        generate_json_sidecar=raw["simulation"].get("generate_json_sidecar", True),
        seed=raw["simulation"].get("seed"),
    )

    families = []
    for family_raw in raw["error_families"]:
        families.append(
            ErrorFamily(
                name=family_raw["name"],
                weight_within_global_error=family_raw["weight_within_global_error"],
                subtypes=[
                    ErrorSubtype(name=s["name"], weight_within_family=s["weight_within_family"])
                    for s in family_raw["subtypes"]
                ],
            )
        )

    return config, families


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the thesis MVP daily simulator.")
    parser.add_argument("--config", required=True, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    config, families = load_configuration(args.config)
    orchestrator = SimulationOrchestrator(config, families)
    summary = orchestrator.execute()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
