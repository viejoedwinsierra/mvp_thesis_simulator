from __future__ import annotations

import argparse
import inspect
import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

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
    ScenarioConfig,
    SimulationConfig,
    StorageTierConfig,
    StorageTierRule,
    TimeSlotConfig,
    TransferConfig,
    WeekdayTimeDistribution,
)
from src.simulator.orchestrator import SimulationOrchestrator


DEFAULT_HASH_CONFIG = {
    "use_full_hash": True,
    "hash_head_length": 12,
    "hash_tail_length": 12,
}

REQUIRED_SIMULATION_SECTIONS = [
    "simulation",
    "daily_charge",
    "base_load_profiles",
    "file_types",
    "transfer",
]


@dataclass(frozen=True)
class ConfigPaths:
    simulation_config: Path
    time_distribution: Path
    lifecycle_config: Path
    cost_config: Path
    error_config: Path
    noise_config: Path
    realism_config: Path
    correlation_config: Path


@dataclass(frozen=True)
class RawConfigBundle:
    simulation: dict[str, Any]
    time_distribution: dict[str, Any]
    lifecycle: dict[str, Any]
    cost: dict[str, Any]
    error: dict[str, Any]
    noise: dict[str, Any]
    realism: dict[str, Any]
    correlation: dict[str, Any]


# ============================================================
# JSON / VALIDATION HELPERS
# ============================================================

def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo de configuracion: {path}")

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON invalido en archivo: {path}. Error: {exc}") from exc


def load_raw_configs(paths: ConfigPaths) -> RawConfigBundle:
    return RawConfigBundle(
        simulation=load_json(paths.simulation_config),
        time_distribution=load_json(paths.time_distribution),
        lifecycle=load_json(paths.lifecycle_config),
        cost=load_json(paths.cost_config),
        error=load_json(paths.error_config),
        noise=load_json(paths.noise_config),
        realism=load_json(paths.realism_config),
        correlation=load_json(paths.correlation_config),
    )


def require_section(raw: Mapping[str, Any], section: str, source: str) -> Any:
    value = raw.get(section)

    if value is None:
        raise ValueError(f"{source} debe contener la seccion obligatoria '{section}'.")

    return value


def validate_required_simulation_sections(simulation_raw: Mapping[str, Any]) -> None:
    missing = [
        section
        for section in REQUIRED_SIMULATION_SECTIONS
        if section not in simulation_raw
    ]

    if missing:
        raise ValueError(
            "simulation_config.json incompleto. Faltan secciones obligatorias: "
            + ", ".join(missing)
        )


def supports_parameter(cls: type, name: str) -> bool:
    try:
        return name in inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return False


def instantiate_supported(cls: type, **kwargs: Any) -> Any:
    accepted = {
        key: value
        for key, value in kwargs.items()
        if supports_parameter(cls, key)
    }
    return cls(**accepted)


# ============================================================
# DATE / RANDOM HELPERS
# ============================================================

def resolve_simulation_date_and_day(
    simulation_date: str | None,
    day: str | None = None,
) -> tuple[str, str]:
    if simulation_date:
        try:
            parsed_date = datetime.strptime(simulation_date, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(
                f"simulation_date invalida: {simulation_date}. Usa formato YYYY-MM-DD."
            ) from exc
    else:
        parsed_date = datetime.now().date()

    calculated_day = parsed_date.strftime("%A").lower()

    if day and day.lower() != calculated_day:
        raise ValueError(
            f"El dia recibido por CLI ({day}) no coincide con la fecha "
            f"{parsed_date.isoformat()}. Dia calculado: {calculated_day}"
        )

    return parsed_date.isoformat(), calculated_day


def sample_uniform_factor(
    cfg: Mapping[str, Any],
    rng: random.Random,
    default: float = 1.0,
) -> float:
    if not cfg or cfg.get("distribution") != "uniform":
        return default

    return rng.uniform(float(cfg["min"]), float(cfg["max"]))


def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


# ============================================================
# BUILDERS
# ============================================================

def build_lognormal_config(raw: Mapping[str, Any]) -> LogNormalDistributionConfig:
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


def resolve_base_load_for_day(
    simulation_raw: Mapping[str, Any],
    selected_day: str,
    rng: random.Random,
) -> float:
    daily_charge = require_section(
        simulation_raw,
        "daily_charge",
        "simulation_config.json",
    )
    base_load_profiles = require_section(
        simulation_raw,
        "base_load_profiles",
        "simulation_config.json",
    )

    if selected_day not in daily_charge:
        available_days = ", ".join(sorted(daily_charge.keys()))
        raise ValueError(
            f"El dia '{selected_day}' no existe en simulation_config.daily_charge. "
            f"Dias disponibles: {available_days}"
        )

    profile_name = daily_charge[selected_day]

    if profile_name not in base_load_profiles:
        available_profiles = ", ".join(sorted(base_load_profiles.keys()))
        raise ValueError(
            f"El perfil '{profile_name}' no existe en simulation_config.base_load_profiles. "
            f"Perfiles disponibles: {available_profiles}"
        )

    profile = base_load_profiles[profile_name]

    if "min" not in profile or "max" not in profile:
        raise ValueError(
            f"El perfil base_load_profiles['{profile_name}'] debe contener min y max."
        )

    min_value = float(profile["min"])
    max_value = float(profile["max"])

    if min_value > max_value:
        raise ValueError(
            f"Perfil '{profile_name}' invalido: min ({min_value}) > max ({max_value})."
        )

    return clamp(rng.uniform(min_value, max_value), 0.0, 1.0)


def load_weekly_time_distribution(
    time_distribution_raw: Mapping[str, Any],
    simulation_raw: Mapping[str, Any],
    rng: random.Random,
) -> list[WeekdayTimeDistribution]:
    error_rate_mode = time_distribution_raw.get("error_rate_mode")

    if not error_rate_mode or str(error_rate_mode).strip().lower() != "multiplier":
        raise ValueError(
            "time_distribution_config.json: error_rate_mode debe ser 'multiplier'. "
            f"Valor actual: {error_rate_mode}"
        )

    weekly_raw = require_section(
        time_distribution_raw,
        "weekly_time_distribution",
        "time_distribution_config.json",
    )

    execution_variability = time_distribution_raw.get("execution_variability", {})
    weekly_base_load_factor = 1.0
    weekly_error_factor = 1.0

    if execution_variability.get("enabled"):
        weekly_base_load_factor = sample_uniform_factor(
            execution_variability.get("weekly_base_load_factor", {}),
            rng=rng,
        )
        weekly_error_factor = sample_uniform_factor(
            execution_variability.get("weekly_error_factor", {}),
            rng=rng,
        )

    result: list[WeekdayTimeDistribution] = []

    for day_raw in weekly_raw:
        day = day_raw["day"]

        base_load = resolve_base_load_for_day(
            simulation_raw=simulation_raw,
            selected_day=day,
            rng=rng,
        )
        base_load = clamp(base_load * weekly_base_load_factor, 0.0, 1.0)

        slots = [
            TimeSlotConfig(
                name=slot["name"],
                start=slot["start"],
                end=slot["end"],
                percentage_load=slot["percentage_load"],
                error_multiplier=slot["error_multiplier"] * weekly_error_factor,
            )
            for slot in day_raw["time_distribution"]
        ]

        result.append(
            WeekdayTimeDistribution(
                day=day,
                base_load=base_load,
                time_distribution=slots,
            )
        )

    return result


def build_arrival_process_config(raw: Mapping[str, Any]) -> ArrivalProcessConfig | None:
    arrival_raw = raw.get("arrival_process")

    if not arrival_raw:
        return None

    return instantiate_supported(
        ArrivalProcessConfig,
        strategy=arrival_raw.get("strategy", "poisson"),
        hourly_noise=arrival_raw.get("hourly_noise", 0.0),
    )


def build_capacity_config(raw: Mapping[str, Any]) -> CapacityConfig | None:
    capacity_raw = raw.get("capacity")

    if not capacity_raw:
        return None

    return instantiate_supported(
        CapacityConfig,
        enabled=capacity_raw.get("enabled", False),
        files_per_hour=capacity_raw.get("files_per_hour", 0),
        min_capacity=capacity_raw.get("min_capacity", 300),
        max_capacity=capacity_raw.get("max_capacity", 1500),
        duration_penalty_factor=capacity_raw.get("duration_penalty_factor", 0.0),
        error_penalty_factor=capacity_raw.get("error_penalty_factor", 0.0),
    )


def build_outliers_config(raw: Mapping[str, Any]) -> OutliersConfig | None:
    outliers_raw = raw.get("outliers")

    if not outliers_raw:
        return None

    return instantiate_supported(
        OutliersConfig,
        enabled=outliers_raw.get("enabled", False),
        probability=outliers_raw.get("probability", 0.0),
        size_multiplier_min=outliers_raw.get("size_multiplier_min", 1.0),
        size_multiplier_max=outliers_raw.get("size_multiplier_max", 1.0),
        max_size_mb=outliers_raw.get("max_size_mb", 1500),
    )


def build_storage_tier_config(cost_raw: Mapping[str, Any]) -> StorageTierConfig:
    raw = require_section(cost_raw, "storage_tier", "cost_config.json")

    return StorageTierConfig(
        strategy=raw["strategy"],
        rules=[
            StorageTierRule(
                tier=rule["tier"],
                min_days_since_last_access=rule.get("min_days_since_last_access"),
                max_days_since_last_access=rule.get("max_days_since_last_access"),
            )
            for rule in raw["rules"]
        ],
        cost_per_mb_per_month=raw["cost_per_mb_per_month"],
    )


def build_lifecycle_config(lifecycle_raw: Mapping[str, Any]) -> LifecycleConfig:
    raw = require_section(lifecycle_raw, "lifecycle", "lifecycle_config.json")

    return LifecycleConfig(
        days_stored=LifecycleDaysConfig(
            distribution=raw["days_stored"]["distribution"],
            min=raw["days_stored"]["min"],
            max=raw["days_stored"]["max"],
        ),
        days_since_last_access=DaysSinceLastAccessConfig(
            strategy=raw["days_since_last_access"]["strategy"],
            profiles={
                name: BetaProfileConfig(
                    alpha=profile_raw["alpha"],
                    beta=profile_raw["beta"],
                )
                for name, profile_raw in raw["days_since_last_access"]["profiles"].items()
            },
            distribution_weights=raw["days_since_last_access"]["distribution_weights"],
        ),
        movement_storage_probability=raw["movement_storage"]["default_probability"],
    )


def build_cost_model_config(cost_raw: Mapping[str, Any]) -> CostModelConfig:
    raw = require_section(cost_raw, "cost_model", "cost_config.json")

    return CostModelConfig(
        storage_cost_formula=raw["storage_cost_formula"],
        non_linear_adjustments=raw.get("non_linear_adjustments", {}),
    )


def build_errors_config(error_raw: Mapping[str, Any]) -> ErrorsConfig:
    raw = require_section(error_raw, "errors", "error_config.json")

    multipliers = {
        name: build_error_multiplier(value)
        for name, value in raw["multipliers"].items()
        if not isinstance(value, dict) or value.get("enabled", True)
    }

    return ErrorsConfig(
        base_error_probability=raw["base_error_probability"],
        multipliers=multipliers,
    )


def build_error_families(error_raw: Mapping[str, Any]) -> list[ErrorFamily]:
    raw = require_section(error_raw, "errors", "error_config.json")

    return [
        ErrorFamily(
            name=family_raw["name"],
            weight_within_global_error=family_raw["weight_within_global_error"],
            description=family_raw.get("description"),
            subtypes=[
                ErrorSubtype(
                    name=subtype_raw["name"],
                    weight_within_family=subtype_raw["weight_within_family"],
                    severity=subtype_raw.get("severity", 1.0),
                    description=subtype_raw.get("description"),
                )
                for subtype_raw in family_raw["subtypes"]
            ],
        )
        for family_raw in raw["error_families"]
        if family_raw.get("enabled", True)
    ]


def build_file_type_config(simulation_raw: Mapping[str, Any]) -> FileTypeConfig:
    raw = require_section(simulation_raw, "file_types", "simulation_config.json")

    return FileTypeConfig(
        distribution=raw["distribution"],
        size_distribution_mb={
            file_type: build_lognormal_config(size_raw)
            for file_type, size_raw in raw["size_distribution_mb"].items()
        },
    )


def build_transfer_config(simulation_raw: Mapping[str, Any]) -> TransferConfig:
    raw = require_section(simulation_raw, "transfer", "simulation_config.json")

    return TransferConfig(
        speed_mbps=build_lognormal_config(raw["speed_mbps"]),
        duration_strategy=raw["duration_strategy"],
    )


def build_scenario_config(simulation_raw: Mapping[str, Any]) -> ScenarioConfig | None:
    scenario_name = None
    scenario_description = None

    raw_scenario = simulation_raw.get("scenario")
    if isinstance(raw_scenario, str):
        scenario_name = raw_scenario
    elif isinstance(raw_scenario, dict):
        scenario_name = raw_scenario.get("name")
        scenario_description = raw_scenario.get("description")

    execution = simulation_raw.get("scenario_execution", {})
    scenario_name = scenario_name or execution.get("scenario_name")

    if not scenario_name:
        return None

    return ScenarioConfig(name=scenario_name, description=scenario_description)


# ============================================================
# CONFIGURATION LOADER
# ============================================================

def build_dataset_config(
    raw: RawConfigBundle,
    simulation_date: str | None = None,
    day: str | None = None,
) -> DatasetSimulationConfig:
    validate_required_simulation_sections(raw.simulation)

    simulation_section = require_section(
        raw.simulation,
        "simulation",
        "simulation_config.json",
    )

    resolved_date, selected_day = resolve_simulation_date_and_day(
        simulation_date=simulation_date,
        day=day,
    )

    seed = simulation_section.get("seed")
    rng = random.Random(seed)

    weekly_time_distribution = load_weekly_time_distribution(
        time_distribution_raw=raw.time_distribution,
        simulation_raw=raw.simulation,
        rng=rng,
    )

    simulation = SimulationConfig(
        simulation_date=resolved_date,
        selected_day=selected_day,
        max_valid_files_per_day=simulation_section["max_valid_files_per_day"],
        output_dir=simulation_section["output_dir"],
        seed=seed,
    )

    hash_raw = raw.simulation.get("hash", DEFAULT_HASH_CONFIG)
    hash_config = HashConfig(**hash_raw) if hash_raw else None

    config = instantiate_supported(
        DatasetSimulationConfig,
        simulation=simulation,
        file_types=build_file_type_config(raw.simulation),
        storage_tier=build_storage_tier_config(raw.cost),
        lifecycle=build_lifecycle_config(raw.lifecycle),
        transfer=build_transfer_config(raw.simulation),
        cost_model=build_cost_model_config(raw.cost),
        errors=build_errors_config(raw.error),
        hash=hash_config,
        weekly_time_distribution=weekly_time_distribution,
        error_families=build_error_families(raw.error),
        arrival_process=build_arrival_process_config(raw.simulation),
        capacity=build_capacity_config(raw.simulation),
        outliers=build_outliers_config(raw.simulation),
        scenario=build_scenario_config(raw.simulation),
        noise=raw.noise.get("noise"),
        realism=raw.realism.get("realism"),
        correlations=raw.correlation.get("correlations"),
        raw_noise_config=raw.noise,
        raw_realism_config=raw.realism,
        raw_correlation_config=raw.correlation,
        raw_cost_config=raw.cost,
        raw_lifecycle_config=raw.lifecycle,
        raw_error_config=raw.error,
    )

    config.validate()
    return config


def load_configuration(
    simulation_config_path: str | Path,
    time_distribution_path: str | Path,
    lifecycle_config_path: str | Path,
    cost_config_path: str | Path,
    error_config_path: str | Path,
    noise_config_path: str | Path,
    realism_config_path: str | Path,
    correlation_config_path: str | Path,
    simulation_date: str | None = None,
    day: str | None = None,
) -> DatasetSimulationConfig:
    paths = ConfigPaths(
        simulation_config=Path(simulation_config_path),
        time_distribution=Path(time_distribution_path),
        lifecycle_config=Path(lifecycle_config_path),
        cost_config=Path(cost_config_path),
        error_config=Path(error_config_path),
        noise_config=Path(noise_config_path),
        realism_config=Path(realism_config_path),
        correlation_config=Path(correlation_config_path),
    )

    raw = load_raw_configs(paths)

    return build_dataset_config(
        raw=raw,
        simulation_date=simulation_date,
        day=day,
    )


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dataset-first thesis simulator with modular configuration files."
    )

    parser.add_argument("--simulation-config", required=True)
    parser.add_argument("--time-distribution", required=True)
    parser.add_argument("--lifecycle-config", required=True)
    parser.add_argument("--cost-config", required=True)
    parser.add_argument("--error-config", required=True)
    parser.add_argument("--noise-config", required=True)
    parser.add_argument("--realism-config", required=True)
    parser.add_argument("--correlation-config", required=True)
    parser.add_argument("--simulation-date", required=False)
    parser.add_argument("--day", required=False)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = load_configuration(
        simulation_config_path=args.simulation_config,
        time_distribution_path=args.time_distribution,
        lifecycle_config_path=args.lifecycle_config,
        cost_config_path=args.cost_config,
        error_config_path=args.error_config,
        noise_config_path=args.noise_config,
        realism_config_path=args.realism_config,
        correlation_config_path=args.correlation_config,
        simulation_date=args.simulation_date,
        day=args.day,
    )

    orchestrator = SimulationOrchestrator(config)
    summary = orchestrator.execute()

    if config.scenario is not None:
        summary["scenario"] = config.scenario.name

    summary["simulation_date"] = config.simulation.simulation_date
    summary["selected_day"] = config.simulation.selected_day

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
