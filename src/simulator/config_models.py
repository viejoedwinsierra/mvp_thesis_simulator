from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


EPSILON = 1e-6


@dataclass(frozen=True)
class ErrorSubtype:
    """Specific error case inside an error family."""

    name: str
    weight_within_family: float
    severity: float = 1.0


@dataclass(frozen=True)
class ErrorFamily:
    """Family of related error subtypes."""

    name: str
    weight_within_global_error: float
    subtypes: List[ErrorSubtype]
    description: str | None = None


@dataclass(frozen=True)
class SizeRangeMB:
    """File size range expressed in MB."""

    min: float
    max: float


@dataclass(frozen=True)
class FileTypeConfig:
    """Distribution and size ranges by file type."""

    distribution: Dict[str, float]
    size_distribution_mb: Dict[str, SizeRangeMB]


@dataclass(frozen=True)
class StorageTierConfig:
    """Storage tier probabilities and cost per MB."""

    distribution: Dict[str, float]
    cost_per_mb_per_month: Dict[str, float]


@dataclass(frozen=True)
class LifecycleDaysConfig:
    """Range for days stored."""

    min: int
    max: int


@dataclass(frozen=True)
class LifecycleConfig:
    """Lifecycle behavior configuration."""

    days_stored: LifecycleDaysConfig
    read_level_distribution: Dict[str, float]
    modify_level_distribution: Dict[str, float]
    movement_storage_probability: float


@dataclass(frozen=True)
class TransferDurationConfig:
    """Transfer duration range in seconds."""

    min: float
    max: float


@dataclass(frozen=True)
class TransferConfig:
    """Transfer simulation configuration."""

    duration_seconds: TransferDurationConfig


@dataclass(frozen=True)
class HashConfig:
    """Partial hash configuration."""

    use_full_hash: bool
    hash_head_length: int
    hash_tail_length: int


@dataclass(frozen=True)
class TimeSlotConfig:
    """Time slot inside a weekday."""

    name: str
    start: str
    end: str
    percentage_load: float
    error_rate: float


@dataclass(frozen=True)
class WeekdayTimeDistribution:
    """Traffic distribution for one weekday."""

    day: str
    base_load: float
    time_distribution: List[TimeSlotConfig]


@dataclass(frozen=True)
class SimulationConfig:
    """General simulation parameters."""

    simulation_date: str
    max_valid_files_per_day: int
    global_error_percentage: float
    output_dir: str
    seed: int | None = None


@dataclass(frozen=True)
class DatasetSimulationConfig:
    """Full configuration used to generate the tabular dataset."""

    simulation: SimulationConfig
    file_types: FileTypeConfig
    storage_tier: StorageTierConfig
    lifecycle: LifecycleConfig
    transfer: TransferConfig
    hash: HashConfig
    weekly_time_distribution: List[WeekdayTimeDistribution]
    error_families: List[ErrorFamily]

    def validate(self) -> None:
        self._validate_simulation()
        self._validate_file_types()
        self._validate_storage_tier()
        self._validate_lifecycle()
        self._validate_transfer()
        self._validate_hash()
        self._validate_weekly_time_distribution()
        self._validate_error_families()

    def _validate_simulation(self) -> None:
        if self.simulation.max_valid_files_per_day <= 0:
            raise ValueError("max_valid_files_per_day must be greater than 0.")

        if not 0 <= self.simulation.global_error_percentage <= 1:
            raise ValueError(
                "global_error_percentage must be expressed as a proportion between 0 and 1."
            )

    def _validate_file_types(self) -> None:
        self._validate_weights_sum_to_one(
            self.file_types.distribution,
            "file_types.distribution",
        )

        for file_type, size_range in self.file_types.size_distribution_mb.items():
            if size_range.min <= 0:
                raise ValueError(f"{file_type}.min size must be greater than 0.")
            if size_range.max <= size_range.min:
                raise ValueError(f"{file_type}.max size must be greater than min.")

        missing = set(self.file_types.distribution) - set(
            self.file_types.size_distribution_mb
        )
        if missing:
            raise ValueError(f"Missing size ranges for file types: {missing}")

    def _validate_storage_tier(self) -> None:
        self._validate_weights_sum_to_one(
            self.storage_tier.distribution,
            "storage_tier.distribution",
        )

        missing = set(self.storage_tier.distribution) - set(
            self.storage_tier.cost_per_mb_per_month
        )
        if missing:
            raise ValueError(f"Missing cost configuration for tiers: {missing}")

        for tier, cost in self.storage_tier.cost_per_mb_per_month.items():
            if cost < 0:
                raise ValueError(f"Storage cost cannot be negative: {tier}={cost}")

    def _validate_lifecycle(self) -> None:
        if self.lifecycle.days_stored.min <= 0:
            raise ValueError("days_stored.min must be greater than 0.")

        if self.lifecycle.days_stored.max < self.lifecycle.days_stored.min:
            raise ValueError("days_stored.max must be >= days_stored.min.")

        self._validate_weights_sum_to_one(
            self.lifecycle.read_level_distribution,
            "lifecycle.read_level_distribution",
        )

        self._validate_weights_sum_to_one(
            self.lifecycle.modify_level_distribution,
            "lifecycle.modify_level_distribution",
        )

        if not 0 <= self.lifecycle.movement_storage_probability <= 1:
            raise ValueError("movement_storage_probability must be between 0 and 1.")

    def _validate_transfer(self) -> None:
        if self.transfer.duration_seconds.min <= 0:
            raise ValueError("transfer.duration_seconds.min must be greater than 0.")

        if self.transfer.duration_seconds.max < self.transfer.duration_seconds.min:
            raise ValueError(
                "transfer.duration_seconds.max must be >= transfer.duration_seconds.min."
            )

    def _validate_hash(self) -> None:
        if self.hash.hash_head_length <= 0:
            raise ValueError("hash_head_length must be greater than 0.")

        if self.hash.hash_tail_length <= 0:
            raise ValueError("hash_tail_length must be greater than 0.")

    def _validate_weekly_time_distribution(self) -> None:
        if not self.weekly_time_distribution:
            raise ValueError("weekly_time_distribution cannot be empty.")

        weekday_weights = {
            item.day: item.base_load for item in self.weekly_time_distribution
        }

        self._validate_weights_sum_to_one(
            weekday_weights,
            "weekly_time_distribution.base_load",
        )

        for weekday in self.weekly_time_distribution:
            slot_weights = {
                slot.name: slot.percentage_load
                for slot in weekday.time_distribution
            }

            self._validate_weights_sum_to_one(
                slot_weights,
                f"{weekday.day}.time_distribution.percentage_load",
            )

            for slot in weekday.time_distribution:
                if not 0 <= slot.error_rate <= 1:
                    raise ValueError(
                        f"Invalid error_rate in {weekday.day}/{slot.name}: {slot.error_rate}"
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

    @staticmethod
    def _validate_weights_sum_to_one(weights: Dict[str, float], label: str) -> None:
        if not weights:
            raise ValueError(f"{label} cannot be empty.")

        for key, value in weights.items():
            if value < 0:
                raise ValueError(f"{label} has negative weight: {key}={value}")

        total = sum(weights.values())

        if abs(total - 1.0) > EPSILON:
            raise ValueError(f"{label} must sum to 1.0. Current sum: {total}")