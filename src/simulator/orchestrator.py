from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any, List
import random

from .allocator import largest_remainder_allocation
from .case_definitions import CASE_CATALOG
from .config_models import DailyUniverseConfig, ErrorFamily
from .content_factory import build_payload, compute_content_hash
from .file_generator import build_file_stem, write_pdf_placeholder
from .metadata_writer import write_json, write_null_json


class SimulationOrchestrator:
    """Main coordinator for the MVP.

    The design intentionally favors orchestration over code centralization.
    Each module owns a specific responsibility, and this orchestrator only
    composes those responsibilities into a daily execution workflow.
    """

    def __init__(self, config: DailyUniverseConfig, error_families: List[ErrorFamily]):
        self.config = config
        self.error_families = error_families
        if config.seed is not None:
            random.seed(config.seed)

    def build_universe_distribution(self) -> Dict[str, Any]:
        max_valid = self.config.max_valid_files_per_day
        global_error_count = round(max_valid * (self.config.global_error_percentage / 100.0))
        unique_valid_count = max_valid - global_error_count

        family_allocations = largest_remainder_allocation(
            global_error_count,
            {family.name: family.weight_within_global_error for family in self.error_families},
        )

        subtype_allocations: Dict[str, int] = {}
        for family in self.error_families:
            family_total = family_allocations.get(family.name, 0)
            family_subs = largest_remainder_allocation(
                family_total,
                {sub.name: sub.weight_within_family for sub in family.subtypes},
            )
            subtype_allocations.update(family_subs)

        return {
            "max_valid_files_per_day": max_valid,
            "global_error_count": global_error_count,
            "unique_valid_count": unique_valid_count,
            "subtype_allocations": subtype_allocations,
        }

    def execute(self) -> Dict[str, Any]:
        distribution = self.build_universe_distribution()
        root = Path(self.config.output_dir) / f"simulation_date={self.config.simulation_date}"
        files_dir = root / "files"
        metadata_dir = root / "metadata"
        summary_dir = root / "summary"

        records: List[Dict[str, Any]] = []
        sequence = 1

        for _ in range(distribution["unique_valid_count"]):
            records.append(self._materialize_case("UNIQUE_VALID", sequence, files_dir, metadata_dir))
            sequence += 1

        for subtype_name, count in distribution["subtype_allocations"].items():
            for _ in range(count):
                records.append(self._materialize_case(subtype_name, sequence, files_dir, metadata_dir))
                sequence += 1

        summary = {
            "simulation_date": self.config.simulation_date,
            "max_valid_files_per_day": self.config.max_valid_files_per_day,
            "global_error_percentage": self.config.global_error_percentage,
            "global_error_count": distribution["global_error_count"],
            "unique_valid_count": distribution["unique_valid_count"],
            "records_created": len(records),
            "case_breakdown": self._case_breakdown(records),
        }
        write_json(summary_dir / "daily_summary.json", summary)
        return summary

    def _materialize_case(self, case_type: str, sequence: int, files_dir: Path, metadata_dir: Path) -> Dict[str, Any]:
        case_def = CASE_CATALOG[case_type]
        payload = build_payload(self.config.simulation_date, sequence)
        content_hash = compute_content_hash(payload)
        file_stem = build_file_stem(payload, sequence)
        pdf_name = f"{file_stem}.pdf"
        json_name = f"{file_stem}.json"

        pdf_path = files_dir / pdf_name
        json_path = metadata_dir / json_name
        file_hash = None

        if case_def.pdf_should_exist and self.config.generate_pdf:
            file_hash = write_pdf_placeholder(pdf_path, payload)

        metadata = {
            "simulation_date": self.config.simulation_date,
            "sequence": sequence,
            "case_group": case_def.case_group,
            "case_type": case_def.case_type,
            "description": case_def.description,
            "pdf_should_exist": case_def.pdf_should_exist,
            "json_should_exist": case_def.json_should_exist,
            "json_should_be_valid": case_def.json_should_be_valid,
            "pdf_name": pdf_name,
            "json_name": json_name,
            "content_hash": content_hash,
            "file_hash": file_hash,
            "payload": payload,
        }

        if case_def.json_should_exist and self.config.generate_json_sidecar:
            if case_def.json_should_be_valid:
                write_json(json_path, metadata)
            else:
                write_null_json(json_path)

        if case_type == "DUP_SAME_CONTENT_DIFFERENT_NAME" and pdf_path.exists():
            alt_name = files_dir / f"dupname_{pdf_name}"
            alt_name.write_text(pdf_path.read_text(encoding="utf-8"), encoding="utf-8")
        elif case_type == "DUP_SAME_CONTENT_DIFFERENT_ROUTE" and pdf_path.exists():
            alt_route = files_dir / "alternate_route" / pdf_name
            alt_route.parent.mkdir(parents=True, exist_ok=True)
            alt_route.write_text(pdf_path.read_text(encoding="utf-8"), encoding="utf-8")
        elif case_type == "DUP_SAME_CONTENT_DIFFERENT_NAME_AND_ROUTE" and pdf_path.exists():
            alt_route = files_dir / "alternate_route_2" / f"dup_{pdf_name}"
            alt_route.parent.mkdir(parents=True, exist_ok=True)
            alt_route.write_text(pdf_path.read_text(encoding="utf-8"), encoding="utf-8")

        return {
            "case_type": case_type,
            "pdf_created": pdf_path.exists(),
            "json_created": json_path.exists(),
            "pdf_name": pdf_name,
            "json_name": json_name,
        }

    @staticmethod
    def _case_breakdown(records: List[Dict[str, Any]]) -> Dict[str, int]:
        breakdown: Dict[str, int] = {}
        for record in records:
            breakdown[record["case_type"]] = breakdown.get(record["case_type"], 0) + 1
        return dict(sorted(breakdown.items()))
