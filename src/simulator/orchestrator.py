from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import random

from .allocator import largest_remainder_allocation
from .case_definitions import CASE_CATALOG
from .config_models import DailyUniverseConfig, ErrorFamily, TimeRangeConfig
from .content_factory import build_payload, compute_content_hash
from .file_generator import build_file_stem, write_pdf_placeholder
from .summary_writer import write_summary_json


class SimulationOrchestrator:

    def __init__(
        self,
        config: DailyUniverseConfig,
        error_families: List[ErrorFamily],
        time_distribution: List[TimeRangeConfig],
    ):
        self.config = config
        self.error_families = error_families
        self.time_distribution = time_distribution

        if config.seed is not None:
            random.seed(config.seed)

    def build_universe_distribution(self) -> Dict[str, Any]:
        total_files = self.config.max_valid_files_per_day
        global_error_count = round(
            total_files * (self.config.global_error_percentage / 100.0)
        )
        unique_valid_count = total_files - global_error_count

        family_allocations = largest_remainder_allocation(
            global_error_count,
            {
                family.name: family.weight_within_global_error
                for family in self.error_families
            },
        )

        subtype_allocations: Dict[str, int] = {}

        for family in self.error_families:
            family_total = family_allocations.get(family.name, 0)

            family_subtypes = largest_remainder_allocation(
                family_total,
                {
                    subtype.name: subtype.weight_within_family
                    for subtype in family.subtypes
                },
            )

            subtype_allocations.update(family_subtypes)

        return {
            "total_files": total_files,
            "global_error_count": global_error_count,
            "unique_valid_count": unique_valid_count,
            "subtype_allocations": subtype_allocations,
        }

    def build_time_distribution(self) -> Dict[str, int]:
        return largest_remainder_allocation(
            self.config.max_valid_files_per_day,
            {
                time_range.name: time_range.percentage_load
                for time_range in self.time_distribution
            },
        )

    def execute(self) -> Dict[str, Any]:
        universe_distribution = self.build_universe_distribution()
        time_allocations = self.build_time_distribution()

        root = Path(self.config.output_dir) / f"simulation_date={self.config.simulation_date}"
        files_dir = root / "files"
        summary_dir = root / "summary"

        records: List[Dict[str, Any]] = []
        sequence = 1

        for time_range in self.time_distribution:
            range_name = time_range.name
            range_count = time_allocations.get(range_name, 0)

            for _ in range(range_count):
                case_type = self._pick_case_type(time_range, universe_distribution)

                record = self._materialize_case(
                    case_type=case_type,
                    sequence=sequence,
                    files_dir=files_dir,
                    time_range=time_range,
                )

                records.append(record)
                sequence += 1

        summary = {
            "simulation_date": self.config.simulation_date,
            "max_valid_files_per_day": self.config.max_valid_files_per_day,
            "global_error_percentage": self.config.global_error_percentage,
            "configured_error_count": universe_distribution["global_error_count"],
            "configured_unique_valid_count": universe_distribution["unique_valid_count"],
            "records_created": len(records),
            "time_distribution": time_allocations,
            "case_breakdown": self._case_breakdown(records),
            "time_range_breakdown": self._time_range_breakdown(records),
            "pdf_created_count": sum(1 for record in records if record["pdf_created"]),
        }

        write_summary_json(summary_dir / "daily_summary.json", summary)
        return summary

    def _pick_case_type(
        self,
        time_range: TimeRangeConfig,
        universe_distribution: Dict[str, Any],
    ) -> str:
        error_rate = time_range.error_rate

        if random.random() > error_rate:
            return "UNIQUE_VALID"

        subtype_allocations = universe_distribution["subtype_allocations"]

        population = list(subtype_allocations.keys())
        weights = list(subtype_allocations.values())

        if not population or sum(weights) == 0:
            return "UNIQUE_VALID"

        return random.choices(population, weights=weights, k=1)[0]

    def _materialize_case(
        self,
        case_type: str,
        sequence: int,
        files_dir: Path,
        time_range: TimeRangeConfig,
    ) -> Dict[str, Any]:

        case_def = CASE_CATALOG[case_type]

        payload = build_payload(
            self.config.simulation_date,
            sequence,
            time_range,
        )

        content_hash = compute_content_hash(payload)

        file_stem = build_file_stem(
            payload,
            sequence,
            time_range,
        )

        pdf_name = f"{file_stem}.pdf"
        pdf_path = files_dir / pdf_name

        file_hash = None

        if case_def.pdf_should_exist and self.config.generate_pdf:
            file_hash = write_pdf_placeholder(
                pdf_path,
                payload,
                time_range,
            )

        return {
            "sequence": sequence,
            "case_type": case_type,
            "case_group": case_def.case_group,
            "time_range": time_range.name,
            "time_start": time_range.start,
            "time_end": time_range.end,
            "error_rate_range": time_range.error_rate,
            "pdf_created": pdf_path.exists(),
            "pdf_name": pdf_name,
            "content_hash": content_hash,
            "file_hash": file_hash,
            "logical_creation_datetime": payload["logical_creation_datetime"],
        }

    @staticmethod
    def _case_breakdown(records: List[Dict[str, Any]]) -> Dict[str, int]:
        breakdown: Dict[str, int] = {}

        for record in records:
            case_type = record["case_type"]
            breakdown[case_type] = breakdown.get(case_type, 0) + 1

        return dict(sorted(breakdown.items()))

    @staticmethod
    def _time_range_breakdown(records: List[Dict[str, Any]]) -> Dict[str, int]:
        breakdown: Dict[str, int] = {}

        for record in records:
            time_range = record["time_range"]
            breakdown[time_range] = breakdown.get(time_range, 0) + 1

        return dict(sorted(breakdown.items()))