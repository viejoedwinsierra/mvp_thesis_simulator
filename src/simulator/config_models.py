from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence


EPSILON = 1e-6

VALID_SCENARIOS = {
    "normal",
    "high_load",
    "low_capacity",
    "network_degraded",
    "large_files",
    "high_error",
}


@dataclass(frozen=True, slots=True)
class ErrorSubtype:
    name: str
    weight_within_family: float
    severity: float = 1.0
    description: str | None = None


@dataclass(frozen=True, slots=True)
class ErrorFamily:
    name: str
    weight_within_global_error: float
    subtypes: Sequence[ErrorSubtype]
    description: str | None = None


@dataclass(frozen=True, slots=True)
class LogNormalDistributionConfig:
    distribution: str
    mean: float
    sigma: float
    min: float
    max: float

    def validate(self, label: str) -> None:
        if self.distribution != "lognormal":
            raise ValueError(f"{label}.distribution must be 'lognormal'.")

        for field_name in ("mean", "sigma", "min", "max"):
            value = getattr(self, field_name)
            if not math.isfinite(value):
                raise ValueError(f"{label}.{field_name} must be finite.")

        if self.sigma <= 0:
            raise ValueError(f"{label}.sigma must be greater than 0.")

        if self.min <= 0:
            raise ValueError(f"{label}.min must be greater than 0.")

        if self.max <= self.min:
            raise ValueError(f"{label}.max must be greater than min.")

        if self.mean <= 0:
            raise ValueError(f"{label}.mean must be greater than 0.")


@dataclass(frozen=True, slots=True)
class FileTypeConfig:
    distribution: Mapping[str, float]
    size_distribution_mb: Mapping[str, LogNormalDistributionConfig]


@dataclass(frozen=True, slots=True)
class StorageTierRule:
    tier: str
    min_days_since_last_access: int | None = None
    max_days_since_last_access: int | None = None


@dataclass(frozen=True, slots=True)
class StorageTierConfig:
    strategy: str
    rules: Sequence[StorageTierRule]
    cost_per_mb_per_month: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class LifecycleDaysConfig:
    distribution: str
    min: int
    max: int


@dataclass(frozen=True, slots=True)
class BetaProfileConfig:
    alpha: float
    beta: float


@dataclass(frozen=True, slots=True)
class DaysSinceLastAccessConfig:
    strategy: str
    profiles: Mapping[str, BetaProfileConfig]
    distribution_weights: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class LifecycleConfig:
    days_stored: LifecycleDaysConfig
    days_since_last_access: DaysSinceLastAccessConfig
    movement_storage_probability: float


@dataclass(frozen=True, slots=True)
class TransferConfig:
    speed_mbps: LogNormalDistributionConfig
    duration_strategy: str


@dataclass(frozen=True, slots=True)
class CostModelConfig:
    storage_cost_formula: str
    non_linear_adjustments: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class ErrorConditionMultiplier:
    multiplier: float
    threshold_mb: float | None = None
    threshold_mbps: float | None = None
    threshold_sec: float | None = None


@dataclass(frozen=True, slots=True)
class ErrorsConfig:
    base_error_probability: float
    multipliers: Mapping[str, ErrorConditionMultiplier | float]


@dataclass(frozen=True, slots=True)
class HashConfig:
    use_full_hash: bool
    hash_head_length: int
    hash_tail_length: int


@dataclass(frozen=True, slots=True)
class TimeSlotConfig:
    name: str
    start: str
    end: str
    percentage_load: float
    error_multiplier: float


@dataclass(frozen=True, slots=True)
class WeekdayTimeDistribution:
    day: str
    base_load: float
    time_distribution: Sequence[TimeSlotConfig]


@dataclass(frozen=True, slots=True)
class TimeDistributionConfig:
    error_rate_mode: str
    weekly_time_distribution: Sequence[WeekdayTimeDistribution]


@dataclass(frozen=True, slots=True)
class SimulationConfig:
    simulation_date: str
    max_valid_files_per_day: int
    output_dir: str
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class ArrivalProcessConfig:
    strategy: str = "poisson"
    hourly_noise: float = 0.0


@dataclass(frozen=True, slots=True)
class CapacityConfig:
    enabled: bool = False
    files_per_hour: int = 0
    duration_penalty_factor: float = 0.0
    error_penalty_factor: float = 0.0


@dataclass(frozen=True, slots=True)
class OutliersConfig:
    enabled: bool = False
    probability: float = 0.0
    size_multiplier_min: float = 1.0
    size_multiplier_max: float = 1.0


@dataclass(frozen=True, slots=True)
class ScenarioConfig:
    name: str
    description: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetSimulationConfig:
    simulation: SimulationConfig
    file_types: FileTypeConfig
    storage_tier: StorageTierConfig
    lifecycle: LifecycleConfig
    transfer: TransferConfig
    cost_model: CostModelConfig
    errors: ErrorsConfig
    hash: HashConfig
    weekly_time_distribution: Sequence[WeekdayTimeDistribution]
    error_families: Sequence[ErrorFamily]
    arrival_process: ArrivalProcessConfig | None = None
    capacity: CapacityConfig | None = None
    outliers: OutliersConfig | None = None
    scenario: ScenarioConfig | None = None

    def validate(self) -> None:
        self._validate_simulation()
        self._validate_file_types()
        self._validate_storage_tier()
        self._validate_lifecycle()
        self._validate_transfer()
        self._validate_cost_model()
        self._validate_errors()
        self._validate_hash()
        self._validate_weekly_time_distribution()
        self._validate_error_families()
        self._validate_optional_monte_carlo_configs()
        self._validate_scenario()
        self._validate_system_consistency()

    def _validate_simulation(self) -> None:
        if self.simulation.max_valid_files_per_day <= 0:
            raise ValueError("max_valid_files_per_day must be greater than 0.")

        try:
            from datetime import date
            date.fromisoformat(self.simulation.simulation_date)
        except ValueError as exc:
            raise ValueError(
                "simulation.simulation_date must use YYYY-MM-DD format."
            ) from exc

        if not self.simulation.output_dir:
            raise ValueError("simulation.output_dir cannot be empty.")

    def _validate_file_types(self) -> None:
        self._validate_weights_sum_to_one(
            self.file_types.distribution,
            "file_types.distribution",
        )

        missing = set(self.file_types.distribution) - set(
            self.file_types.size_distribution_mb
        )

        if missing:
            raise ValueError(f"Missing size distributions for file types: {missing}")

        for file_type, distribution in self.file_types.size_distribution_mb.items():
            distribution.validate(f"file_types.size_distribution_mb.{file_type}")

    def _validate_storage_tier(self) -> None:
        if self.storage_tier.strategy != "by_days_since_last_access":
            raise ValueError(
                "storage_tier.strategy must be 'by_days_since_last_access'."
            )

        if not self.storage_tier.rules:
            raise ValueError("storage_tier.rules cannot be empty.")

        tiers_from_rules = {rule.tier for rule in self.storage_tier.rules}
        tiers_from_costs = set(self.storage_tier.cost_per_mb_per_month)

        missing_costs = tiers_from_rules - tiers_from_costs
        if missing_costs:
            raise ValueError(f"Missing cost configuration for tiers: {missing_costs}")

        for tier, cost in self.storage_tier.cost_per_mb_per_month.items():
            if cost < 0:
                raise ValueError(f"Storage cost cannot be negative: {tier}={cost}")

        for rule in self.storage_tier.rules:
            if (
                rule.min_days_since_last_access is not None
                and rule.min_days_since_last_access < 0
            ):
                raise ValueError(
                    f"{rule.tier}.min_days_since_last_access cannot be negative."
                )

            if (
                rule.max_days_since_last_access is not None
                and rule.max_days_since_last_access < 0
            ):
                raise ValueError(
                    f"{rule.tier}.max_days_since_last_access cannot be negative."
                )

            if (
                rule.min_days_since_last_access is not None
                and rule.max_days_since_last_access is not None
                and rule.max_days_since_last_access < rule.min_days_since_last_access
            ):
                raise ValueError(
                    f"{rule.tier}.max_days_since_last_access must be >= min."
                )

    def _validate_lifecycle(self) -> None:
        if self.lifecycle.days_stored.distribution != "uniform":
            raise ValueError("lifecycle.days_stored.distribution must be 'uniform'.")

        if self.lifecycle.days_stored.min <= 0:
            raise ValueError("days_stored.min must be greater than 0.")

        if self.lifecycle.days_stored.max < self.lifecycle.days_stored.min:
            raise ValueError("days_stored.max must be >= days_stored.min.")

        access_cfg = self.lifecycle.days_since_last_access

        if access_cfg.strategy != "beta_scaled_to_days_stored":
            raise ValueError(
                "days_since_last_access.strategy must be 'beta_scaled_to_days_stored'."
            )

        self._validate_weights_sum_to_one(
            access_cfg.distribution_weights,
            "lifecycle.days_since_last_access.distribution_weights",
        )

        missing_profiles = set(access_cfg.distribution_weights) - set(
            access_cfg.profiles
        )

        if missing_profiles:
            raise ValueError(f"Missing beta profiles: {missing_profiles}")

        for name, profile in access_cfg.profiles.items():
            if profile.alpha <= 0:
                raise ValueError(f"{name}.alpha must be greater than 0.")
            if profile.beta <= 0:
                raise ValueError(f"{name}.beta must be greater than 0.")

        if not 0 <= self.lifecycle.movement_storage_probability <= 1:
            raise ValueError("movement_storage_probability must be between 0 and 1.")

    def _validate_transfer(self) -> None:
        self.transfer.speed_mbps.validate("transfer.speed_mbps")

        if self.transfer.duration_strategy != "size_divided_by_speed":
            raise ValueError(
                "transfer.duration_strategy must be 'size_divided_by_speed'."
            )

    def _validate_cost_model(self) -> None:
        if not self.cost_model.storage_cost_formula:
            raise ValueError("cost_model.storage_cost_formula cannot be empty.")

        for key, value in self.cost_model.non_linear_adjustments.items():
            if not math.isfinite(value):
                raise ValueError(
                    f"cost_model.non_linear_adjustments.{key} must be finite."
                )

            if key == "discount_factor" and not 0 < value <= 1:
                raise ValueError("cost_model.discount_factor must be in (0, 1].")

            if key.endswith("_threshold_mb") and value <= 0:
                raise ValueError(
                    f"cost_model.non_linear_adjustments.{key} must be greater than 0."
                )

    def _validate_errors(self) -> None:
        if not 0 <= self.errors.base_error_probability <= 1:
            raise ValueError("base_error_probability must be between 0 and 1.")

        if not self.errors.multipliers:
            raise ValueError("errors.multipliers cannot be empty.")

        for name, multiplier_cfg in self.errors.multipliers.items():
            if isinstance(multiplier_cfg, (int, float)):
                if multiplier_cfg <= 0:
                    raise ValueError(
                        f"errors.multipliers.{name} must be greater than 0."
                    )

                if multiplier_cfg > 5:
                    raise ValueError(
                        f"errors.multipliers.{name} is too high (>5): "
                        f"{multiplier_cfg}"
                    )

                continue

            if multiplier_cfg.multiplier <= 0:
                raise ValueError(
                    f"errors.multipliers.{name}.multiplier must be greater than 0."
                )

            if multiplier_cfg.multiplier > 5:
                raise ValueError(
                    f"errors.multipliers.{name}.multiplier is too high (>5): "
                    f"{multiplier_cfg.multiplier}"
                )

            if (
                multiplier_cfg.threshold_mb is not None
                and multiplier_cfg.threshold_mb <= 0
            ):
                raise ValueError(
                    f"errors.multipliers.{name}.threshold_mb must be greater than 0."
                )

            if (
                multiplier_cfg.threshold_mbps is not None
                and multiplier_cfg.threshold_mbps <= 0
            ):
                raise ValueError(
                    f"errors.multipliers.{name}.threshold_mbps must be greater than 0."
                )

            if (
                multiplier_cfg.threshold_sec is not None
                and multiplier_cfg.threshold_sec <= 0
            ):
                raise ValueError(
                    f"errors.multipliers.{name}.threshold_sec must be greater than 0."
                )

    def _validate_hash(self) -> None:
        if self.hash.hash_head_length <= 0:
            raise ValueError("hash_head_length must be greater than 0.")

        if self.hash.hash_tail_length <= 0:
            raise ValueError("hash_tail_length must be greater than 0.")

        if self.hash.hash_head_length + self.hash.hash_tail_length > 64:
            raise ValueError(
                "hash_head_length + hash_tail_length cannot exceed SHA-256 length."
            )

    def _validate_weekly_time_distribution(self) -> None:
        if not self.weekly_time_distribution:
            raise ValueError("weekly_time_distribution cannot be empty.")

        expected_days = {
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        }

        actual_days = {item.day.lower() for item in self.weekly_time_distribution}

        if actual_days != expected_days:
            raise ValueError(
                "weekly_time_distribution must contain exactly the 7 weekdays. "
                f"Current days: {sorted(actual_days)}"
            )

        weekday_weights = {
            item.day: item.base_load
            for item in self.weekly_time_distribution
        }

        self._validate_weights_sum_to_one(
            weekday_weights,
            "weekly_time_distribution.base_load",
        )

        for weekday in self.weekly_time_distribution:
            if not weekday.time_distribution:
                raise ValueError(f"{weekday.day}.time_distribution cannot be empty.")

            slot_weights = {
                slot.name: slot.percentage_load
                for slot in weekday.time_distribution
            }

            self._validate_weights_sum_to_one(
                slot_weights,
                f"{weekday.day}.time_distribution.percentage_load",
            )

            for slot in weekday.time_distribution:
                self._validate_time_format(slot.start, f"{weekday.day}/{slot.name}.start")
                self._validate_time_format(slot.end, f"{weekday.day}/{slot.name}.end")

                if slot.error_multiplier <= 0:
                    raise ValueError(
                        f"Invalid error_multiplier in {weekday.day}/{slot.name}: "
                        f"{slot.error_multiplier}"
                    )

                if slot.error_multiplier > 5:
                    raise ValueError(
                        f"Suspiciously high error_multiplier in "
                        f"{weekday.day}/{slot.name}: {slot.error_multiplier}"
                    )

    def _validate_error_families(self) -> None:
        if not self.error_families:
            raise ValueError("error_families cannot be empty.")

        family_weights = {
            family.name: family.weight_within_global_error
            for family in self.error_families
        }

        self._validate_weights_sum_to_one(
            family_weights,
            "error_families.weight_within_global_error",
        )

        for family in self.error_families:
            if not family.subtypes:
                raise ValueError(f"Error family has no subtypes: {family.name}")

            subtype_weights = {
                subtype.name: subtype.weight_within_family
                for subtype in family.subtypes
            }

            self._validate_weights_sum_to_one(
                subtype_weights,
                f"{family.name}.subtypes.weight_within_family",
            )

            for subtype in family.subtypes:
                if not 0 <= subtype.severity <= 1:
                    raise ValueError(
                        f"Severity must be between 0 and 1: {subtype.name}"
                    )

    def _validate_optional_monte_carlo_configs(self) -> None:
        if self.arrival_process is not None:
            if self.arrival_process.strategy != "poisson":
                raise ValueError("arrival_process.strategy must be 'poisson'.")

            if not 0 <= self.arrival_process.hourly_noise <= 0.5:
                raise ValueError(
                    "arrival_process.hourly_noise must be between 0 and 0.5."
                )

        if self.capacity is not None:
            if self.capacity.enabled:
                if self.capacity.files_per_hour <= 0:
                    raise ValueError(
                        "capacity.files_per_hour must be greater than 0 when enabled."
                    )

            if self.capacity.duration_penalty_factor < 0:
                raise ValueError(
                    "capacity.duration_penalty_factor cannot be negative."
                )

            if self.capacity.duration_penalty_factor > 2:
                raise ValueError(
                    "capacity.duration_penalty_factor is too high (>2)."
                )

            if self.capacity.error_penalty_factor < 0:
                raise ValueError(
                    "capacity.error_penalty_factor cannot be negative."
                )

            if self.capacity.error_penalty_factor > 3:
                raise ValueError(
                    "capacity.error_penalty_factor is too high (>3)."
                )

        if self.outliers is not None:
            if not 0 <= self.outliers.probability <= 1:
                raise ValueError("outliers.probability must be between 0 and 1.")

            if self.outliers.probability > 0.05:
                raise ValueError(
                    "outliers.probability is too high (>0.05) for rare events."
                )

            if self.outliers.size_multiplier_min <= 0:
                raise ValueError("outliers.size_multiplier_min must be greater than 0.")

            if self.outliers.size_multiplier_max <= self.outliers.size_multiplier_min:
                raise ValueError(
                    "outliers.size_multiplier_max must be greater than "
                    "outliers.size_multiplier_min."
                )

            if self.outliers.size_multiplier_max > 15:
                raise ValueError(
                    "outliers.size_multiplier_max is too high (>15)."
                )

    def _validate_scenario(self) -> None:
        if self.scenario is None:
            return

        if not self.scenario.name:
            raise ValueError("scenario.name cannot be empty.")

        if self.scenario.name not in VALID_SCENARIOS:
            raise ValueError(
                f"Invalid scenario.name={self.scenario.name}. "
                f"Allowed values: {sorted(VALID_SCENARIOS)}"
            )

    def _validate_system_consistency(self) -> None:
        if self.capacity is not None and self.capacity.enabled:
            estimated_peak = self.simulation.max_valid_files_per_day * 0.50 / 6

            if self.capacity.files_per_hour > estimated_peak * 2.5:
                raise ValueError(
                    "capacity.files_per_hour is too high relative to estimated "
                    f"peak load. capacity={self.capacity.files_per_hour}, "
                    f"estimated_peak={estimated_peak:.2f}"
                )

            if self.capacity.files_per_hour < estimated_peak * 0.35:
                raise ValueError(
                    "capacity.files_per_hour is too low relative to estimated "
                    f"peak load. capacity={self.capacity.files_per_hour}, "
                    f"estimated_peak={estimated_peak:.2f}"
                )

        if self.outliers is not None and self.outliers.enabled:
            for file_type, dist in self.file_types.size_distribution_mb.items():
                effective_max = dist.max * self.outliers.size_multiplier_max

                if effective_max > 10000:
                    raise ValueError(
                        f"Outlier configuration can generate extremely large "
                        f"{file_type} files. effective_max={effective_max}"
                    )

    @staticmethod
    def _validate_weights_sum_to_one(
        weights: Mapping[str, float],
        label: str,
    ) -> None:
        if not weights:
            raise ValueError(f"{label} cannot be empty.")

        for key, value in weights.items():
            if not math.isfinite(value):
                raise ValueError(f"{label} has non-finite weight: {key}={value}")

            if value < 0:
                raise ValueError(f"{label} has negative weight: {key}={value}")

        total = sum(weights.values())

        if abs(total - 1.0) > EPSILON:
            raise ValueError(f"{label} must sum to 1.0. Current sum: {total}")

    @staticmethod
    def _validate_time_format(value: str, label: str) -> None:
        parts = value.split(":")

        if len(parts) != 2:
            raise ValueError(f"{label} must use HH:MM format.")

        hour, minute = parts

        if not hour.isdigit() or not minute.isdigit():
            raise ValueError(f"{label} must use numeric HH:MM format.")

        hour_int = int(hour)
        minute_int = int(minute)

        if not 0 <= hour_int <= 23:
            raise ValueError(f"{label} hour must be between 0 and 23.")

        if not 0 <= minute_int <= 59:
            raise ValueError(f"{label} minute must be between 0 and 59.")