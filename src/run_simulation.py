from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.simulator.config_models import (
    DatasetSimulationConfig,
    SimulationConfig,
    FileTypeConfig,
    StorageTierConfig,
    LifecycleConfig,
    LifecycleDaysConfig,
    TransferConfig,
    TransferDurationConfig,
    HashConfig,
    WeekdayTimeDistribution,
    TimeSlotConfig,
    ErrorFamily,
    ErrorSubtype,
    SizeRangeMB,
)
from src.simulator.orchestrator import SimulationOrchestrator


def load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_weekly_time_distribution(path: str) -> list[WeekdayTimeDistribution]:
    raw = load_json(path)

    return [
        WeekdayTimeDistribution(
            day=day_raw["day"],
            base_load=day_raw["base_load"],
            time_distribution=[
                TimeSlotConfig(
                    name=slot["name"],
                    start=slot["start"],
                    end=slot["end"],
                    percentage_load=slot["percentage_load"],
                    error_rate=slot["error_rate"],
                )
                for slot in day_raw["time_distribution"]
            ],
        )
        for day_raw in raw["weekly_time_distribution"]
    ]

def load_configuration(
    simulation_config_path: str,
    time_distribution_path: str,
) -> DatasetSimulationConfig:
    raw = load_json(simulation_config_path)
    weekly_time_distribution = load_weekly_time_distribution(time_distribution_path)

    simulation = SimulationConfig(
        simulation_date=raw["simulation"]["simulation_date"],
        max_valid_files_per_day=raw["simulation"]["max_valid_files_per_day"],
        global_error_percentage=raw["simulation"]["global_error_percentage"],
        output_dir=raw["simulation"]["output_dir"],
        seed=raw["simulation"].get("seed"),
    )

    file_types = FileTypeConfig(
        distribution=raw["file_types"]["distribution"],
        size_distribution_mb={
            file_type: SizeRangeMB(**size_range)
            for file_type, size_range in raw["file_types"]["size_distribution_mb"].items()
        },
    )

    storage_tier = StorageTierConfig(
        distribution=raw["storage_tier"]["distribution"],
        cost_per_mb_per_month=raw["storage_tier"]["cost_per_mb_per_month"],
    )

    lifecycle = LifecycleConfig(
        days_stored=LifecycleDaysConfig(**raw["lifecycle"]["days_stored"]),
        read_level_distribution=raw["lifecycle"]["read_level_distribution"],
        modify_level_distribution=raw["lifecycle"]["modify_level_distribution"],
        movement_storage_probability=raw["lifecycle"]["movement_storage_probability"],
    )

    transfer = TransferConfig(
        duration_seconds=TransferDurationConfig(
            **raw["transfer"]["duration_seconds"]
        )
    )

    hash_config = HashConfig(**raw["hash"])

    error_families = [
        ErrorFamily(
            name=family_raw["name"],
            weight_within_global_error=family_raw["weight_within_global_error"],
            subtypes=[
                ErrorSubtype(
                    name=subtype_raw["name"],
                    weight_within_family=subtype_raw["weight_within_family"],
                    severity=subtype_raw.get("severity", 1.0),
                )
                for subtype_raw in family_raw["subtypes"]
            ],
        )
        for family_raw in raw["error_families"]
    ]

    config = DatasetSimulationConfig(
        simulation=simulation,
        file_types=file_types,
        storage_tier=storage_tier,
        lifecycle=lifecycle,
        transfer=transfer,
        hash=hash_config,
        weekly_time_distribution=weekly_time_distribution,
        error_families=error_families,
    )

    config.validate()
    return config


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run dataset-first thesis simulator."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--time-distribution", required=True)

    args = parser.parse_args()

    config = load_configuration(
        simulation_config_path=args.config,
        time_distribution_path=args.time_distribution,
    )

    orchestrator = SimulationOrchestrator(config)
    summary = orchestrator.execute()

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())