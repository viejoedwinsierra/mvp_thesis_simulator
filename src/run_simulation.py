from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.simulator.config_models import (
    ArrivalProcessConfig,
    BetaProfileConfig,
    CapacityConfig,
    CostModelConfig,
    DatasetSimulationConfig,
    DaysSinceLastAccessConfig,
    ErrorConditionMultiplier,
    ErrorFamily,
    ErrorsConfig,
    ErrorSubtype,
    FileTypeConfig,
    HashConfig,
    LifecycleConfig,
    LifecycleDaysConfig,
    LogNormalDistributionConfig,
    OutliersConfig,
    SimulationConfig,
    StorageTierConfig,
    StorageTierRule,
    TimeSlotConfig,
    TransferConfig,
    WeekdayTimeDistribution,
)
from src.simulator.orchestrator import SimulationOrchestrator


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_lognormal_config(raw: dict[str, Any]) -> LogNormalDistributionConfig:
    return LogNormalDistributionConfig(
        distribution=raw["distribution"],
        mean=raw["mean"],
        sigma=raw["sigma"],
        min=raw["min"],
        max=raw["max"],
    )


def build_error_multiplier(raw: Any) -> ErrorConditionMultiplier | float:
    if isinstance(raw, (int, float)):
        return float(raw)

    return ErrorConditionMultiplier(
        multiplier=raw["multiplier"],
        threshold_mb=raw.get("threshold_mb"),
        threshold_mbps=raw.get("threshold_mbps"),
        threshold_sec=raw.get("threshold_sec"),
    )


def load_weekly_time_distribution(
    path: str | Path,
) -> list[WeekdayTimeDistribution]:
    raw = load_json(path)

    error_rate_mode = raw.get("error_rate_mode")

    if not error_rate_mode or str(error_rate_mode).strip().lower() != "multiplier":
        raise ValueError(
            "time_distribution.error_rate_mode must be 'multiplier'. "
            f"Current value: {error_rate_mode}"
        )

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
                    error_multiplier=slot["error_multiplier"],
                )
                for slot in day_raw["time_distribution"]
            ],
        )
        for day_raw in raw["weekly_time_distribution"]
    ]


def build_arrival_process_config(raw: dict[str, Any]) -> ArrivalProcessConfig | None:
    arrival_raw = raw.get("arrival_process")

    if not arrival_raw:
        return None

    return ArrivalProcessConfig(
        strategy=arrival_raw.get("strategy", "poisson"),
        hourly_noise=arrival_raw.get("hourly_noise", 0.0),
    )


def build_capacity_config(raw: dict[str, Any]) -> CapacityConfig | None:
    capacity_raw = raw.get("capacity")

    if not capacity_raw:
        return None

    return CapacityConfig(
        enabled=capacity_raw.get("enabled", False),
        files_per_hour=capacity_raw.get("files_per_hour", 0),
        duration_penalty_factor=capacity_raw.get(
            "duration_penalty_factor",
            0.0,
        ),
        error_penalty_factor=capacity_raw.get(
            "error_penalty_factor",
            0.0,
        ),
    )


def build_outliers_config(raw: dict[str, Any]) -> OutliersConfig | None:
    outliers_raw = raw.get("outliers")

    if not outliers_raw:
        return None

    return OutliersConfig(
        enabled=outliers_raw.get("enabled", False),
        probability=outliers_raw.get("probability", 0.0),
        size_multiplier_min=outliers_raw.get("size_multiplier_min", 1.0),
        size_multiplier_max=outliers_raw.get("size_multiplier_max", 1.0),
    )


def load_configuration(
    simulation_config_path: str | Path,
    time_distribution_path: str | Path,
) -> DatasetSimulationConfig:
    raw = load_json(simulation_config_path)
    weekly_time_distribution = load_weekly_time_distribution(time_distribution_path)

    simulation = SimulationConfig(
        simulation_date=raw["simulation"]["simulation_date"],
        max_valid_files_per_day=raw["simulation"]["max_valid_files_per_day"],
        output_dir=raw["simulation"]["output_dir"],
        seed=raw["simulation"].get("seed"),
    )

    arrival_process = build_arrival_process_config(raw)
    capacity = build_capacity_config(raw)
    outliers = build_outliers_config(raw)

    file_types = FileTypeConfig(
        distribution=raw["file_types"]["distribution"],
        size_distribution_mb={
            file_type: build_lognormal_config(size_raw)
            for file_type, size_raw in raw["file_types"][
                "size_distribution_mb"
            ].items()
        },
    )

    storage_tier = StorageTierConfig(
        strategy=raw["storage_tier"]["strategy"],
        rules=[
            StorageTierRule(
                tier=rule["tier"],
                min_days_since_last_access=rule.get(
                    "min_days_since_last_access"
                ),
                max_days_since_last_access=rule.get(
                    "max_days_since_last_access"
                ),
            )
            for rule in raw["storage_tier"]["rules"]
        ],
        cost_per_mb_per_month=raw["storage_tier"]["cost_per_mb_per_month"],
    )

    lifecycle = LifecycleConfig(
        days_stored=LifecycleDaysConfig(
            distribution=raw["lifecycle"]["days_stored"]["distribution"],
            min=raw["lifecycle"]["days_stored"]["min"],
            max=raw["lifecycle"]["days_stored"]["max"],
        ),
        days_since_last_access=DaysSinceLastAccessConfig(
            strategy=raw["lifecycle"]["days_since_last_access"]["strategy"],
            profiles={
                name: BetaProfileConfig(
                    alpha=profile_raw["alpha"],
                    beta=profile_raw["beta"],
                )
                for name, profile_raw in raw["lifecycle"][
                    "days_since_last_access"
                ]["profiles"].items()
            },
            distribution_weights=raw["lifecycle"][
                "days_since_last_access"
            ]["distribution_weights"],
        ),
        movement_storage_probability=raw["lifecycle"][
            "movement_storage_probability"
        ],
    )

    transfer = TransferConfig(
        speed_mbps=build_lognormal_config(raw["transfer"]["speed_mbps"]),
        duration_strategy=raw["transfer"]["duration_strategy"],
    )

    cost_model = CostModelConfig(
        storage_cost_formula=raw["cost_model"]["storage_cost_formula"],
        non_linear_adjustments=raw["cost_model"].get(
            "non_linear_adjustments",
            {},
        ),
    )

    errors = ErrorsConfig(
        base_error_probability=raw["errors"]["base_error_probability"],
        multipliers={
            name: build_error_multiplier(value)
            for name, value in raw["errors"]["multipliers"].items()
        },
    )

    hash_config = HashConfig(**raw["hash"])

    error_families = [
        ErrorFamily(
            name=family_raw["name"],
            weight_within_global_error=family_raw[
                "weight_within_global_error"
            ],
            description=family_raw.get("description"),
            subtypes=[
                ErrorSubtype(
                    name=subtype_raw["name"],
                    weight_within_family=subtype_raw[
                        "weight_within_family"
                    ],
                    severity=subtype_raw.get("severity", 1.0),
                    description=subtype_raw.get("description"),
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
        cost_model=cost_model,
        errors=errors,
        hash=hash_config,
        weekly_time_distribution=weekly_time_distribution,
        error_families=error_families,
        arrival_process=arrival_process,
        capacity=capacity,
        outliers=outliers,
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

    raw_config = load_json(args.config)
    scenario = raw_config.get("scenario")

    if scenario:
        summary["scenario"] = scenario

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())