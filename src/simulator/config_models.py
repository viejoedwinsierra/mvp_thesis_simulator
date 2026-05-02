from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ErrorSubtype:
    """Specific error case inside a family."""

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
class TimeRangeConfig:
    """Time range used to distribute the daily load."""

    name: str
    start: str
    end: str
    percentage_load: float
    error_rate: float


@dataclass(frozen=True)
class DailyUniverseConfig:
    """High-level configuration for one daily simulation run."""

    simulation_date: str
    max_valid_files_per_day: int
    global_error_percentage: float
    output_dir: str
    time_distribution: List[TimeRangeConfig]

    generate_pdf: bool = True
    generate_json_sidecar: bool = True
    seed: int | None = None

    def validate(self) -> None:
        self._validate_global_error_percentage()
        self._validate_max_valid_files_per_day()
        self._validate_time_distribution()

    def _validate_global_error_percentage(self) -> None:
        if not 0 <= self.global_error_percentage <= 100:
            raise ValueError(
                "global_error_percentage debe estar entre 0 y 100"
            )

    def _validate_max_valid_files_per_day(self) -> None:
        if self.max_valid_files_per_day <= 0:
            raise ValueError(
                "max_valid_files_per_day debe ser mayor que 0"
            )

    def _validate_time_distribution(self) -> None:
        if not self.time_distribution:
            raise ValueError("time_distribution no puede estar vacío")

        total_load = sum(r.percentage_load for r in self.time_distribution)

        if abs(total_load - 1.0) > 1e-6:
            raise ValueError(
                f"percentage_load debe sumar 1.0. Actualmente suma {total_load}"
            )

        for time_range in self.time_distribution:
            if not 0 <= time_range.percentage_load <= 1:
                raise ValueError(
                    f"percentage_load inválido en {time_range.name}: "
                    f"{time_range.percentage_load}"
                )

            if not 0 <= time_range.error_rate <= 1:
                raise ValueError(
                    f"error_rate inválido en {time_range.name}: "
                    f"{time_range.error_rate}"
                )