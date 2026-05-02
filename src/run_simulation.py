from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.simulator.config_models import (
    DailyUniverseConfig,
    ErrorFamily,
    ErrorSubtype,
    TimeRangeConfig,
)
from src.simulator.orchestrator import SimulationOrchestrator


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_time_distribution(path: str) -> list[TimeRangeConfig]:
    raw = load_json(path)

    return [
        TimeRangeConfig(
            name=r["name"],
            start=r["start"],
            end=r["end"],
            percentage_load=r["percentage_load"],
            error_rate=r["error_rate"],
        )
        for r in raw["time_distribution"]
    ]


def load_configuration(
    path: str,
    time_distribution: list[TimeRangeConfig],
) -> tuple[DailyUniverseConfig, list[ErrorFamily]]:
    raw = load_json(path)

    config = DailyUniverseConfig(
        simulation_date=raw["simulation"]["simulation_date"],
        max_valid_files_per_day=raw["simulation"]["max_valid_files_per_day"],
        global_error_percentage=raw["simulation"]["global_error_percentage"],
        output_dir=raw["simulation"]["output_dir"],
        time_distribution=time_distribution,
        generate_pdf=raw["simulation"].get("generate_pdf", True),
        generate_json_sidecar=False,
        seed=raw["simulation"].get("seed"),
    )

    config.validate()

    families = [
        ErrorFamily(
            name=family_raw["name"],
            weight_within_global_error=family_raw["weight_within_global_error"],
            subtypes=[
                ErrorSubtype(
                    name=s["name"],
                    weight_within_family=s["weight_within_family"],
                )
                for s in family_raw["subtypes"]
            ],
        )
        for family_raw in raw["error_families"]
    ]

    return config, families


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the thesis MVP daily simulator.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--time-distribution", required=True)

    args = parser.parse_args()

    time_distribution = load_time_distribution(args.time_distribution)
    config, families = load_configuration(args.config, time_distribution)

    orchestrator = SimulationOrchestrator(
        config=config,
        error_families=families,
        time_distribution=time_distribution,
    )

    summary = orchestrator.execute()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())