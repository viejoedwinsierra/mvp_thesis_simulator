from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ErrorSubtype:
    """Specific error case inside a family.

    Attributes:
        name: Stable identifier for the error subtype.
        weight_within_family: Relative probability inside the family. The
            complete family is normalized later, so absolute scale is not
            required, only proportional weights.
    """

    name: str
    weight_within_family: float


@dataclass(frozen=True)
class ErrorFamily:
    """Family of related error subtypes.

    This supports a hierarchical probability model:

        total_day -> global_error -> family -> subtype
    """

    name: str
    weight_within_global_error: float
    subtypes: List[ErrorSubtype]


@dataclass(frozen=True)
class DailyUniverseConfig:
    """High-level configuration for one daily simulation run.

    Attributes:
        simulation_date: Logical date used by the simulator.
        max_valid_files_per_day: Maximum base population for the day.
        global_error_percentage: Percentage of the base population assigned to
            the error universe.
        output_dir: Root output directory.
        generate_pdf: Whether the valid and applicable error cases should
            create a PDF artifact.
        generate_json_sidecar: Whether sidecar metadata should be created.
        seed: Optional seed for reproducible runs.
    """

    simulation_date: str
    max_valid_files_per_day: int
    global_error_percentage: float
    output_dir: str
    generate_pdf: bool = True
    generate_json_sidecar: bool = True
    seed: int | None = None
