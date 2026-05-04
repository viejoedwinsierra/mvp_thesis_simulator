from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import csv
import random
import uuid

from .allocator import largest_remainder_allocation
from .case_definitions import CASE_CATALOG
from .config_models import DatasetSimulationConfig, TimeSlotConfig
from .content_factory import build_logical_payload, build_content_signature


class SimulationOrchestrator:
    """Dataset-first simulation orchestrator.

    This orchestrator does not create physical files.
    It only generates the tabular dataset used as the source of truth.
    """

    def __init__(self, config: DatasetSimulationConfig):
        self.config = config
        self.config.validate()

        if self.config.simulation.seed is not None:
            random.seed(self.config.simulation.seed)

    def execute(self) -> Dict[str, Any]:
        records = self.generate_records()

        output_dir = Path(self.config.simulation.output_dir) / "dataset"
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "blob_inventory.csv"
        self.write_csv(csv_path, records)

        return {
            "simulation_date": self.config.simulation.simulation_date,
            "records_created": len(records),
            "dataset_path": str(csv_path),
            "case_breakdown": self.case_breakdown(records),
            "file_type_breakdown": self.file_type_breakdown(records),
            "storage_tier_breakdown": self.storage_tier_breakdown(records),
        }

    def generate_records(self) -> List[Dict[str, Any]]:
        total_files = self.config.simulation.max_valid_files_per_day

        weekday_allocations = largest_remainder_allocation(
            total_files,
            {
                item.day: item.base_load
                for item in self.config.weekly_time_distribution
            },
        )

        records: List[Dict[str, Any]] = []
        sequence = 1

        for weekday_config in self.config.weekly_time_distribution:
            day_count = weekday_allocations.get(weekday_config.day, 0)

            slot_allocations = largest_remainder_allocation(
                day_count,
                {
                    slot.name: slot.percentage_load
                    for slot in weekday_config.time_distribution
                },
            )

            for slot in weekday_config.time_distribution:
                slot_count = slot_allocations.get(slot.name, 0)

                for _ in range(slot_count):
                    record = self.build_record(
                        sequence=sequence,
                        day_of_week=weekday_config.day,
                        time_slot=slot,
                    )
                    records.append(record)
                    sequence += 1

        return records

    def build_record(
        self,
        sequence: int,
        day_of_week: str,
        time_slot: TimeSlotConfig,
    ) -> Dict[str, Any]:
        file_type = self.pick_weighted(self.config.file_types.distribution)
        size_mb = self.generate_size_mb(file_type)
        storage_tier = self.pick_weighted(self.config.storage_tier.distribution)

        days_stored = random.randint(
            self.config.lifecycle.days_stored.min,
            self.config.lifecycle.days_stored.max,
        )

        days_since_last_access = random.randint(0, days_stored)

        read_level = self.pick_weighted(
            self.config.lifecycle.read_level_distribution
        )
        modify_level = self.pick_weighted(
            self.config.lifecycle.modify_level_distribution
        )

        movement_storage = self.bernoulli(
            self.config.lifecycle.movement_storage_probability
        )

        transfer_duration_sec = random.uniform(
            self.config.transfer.duration_seconds.min,
            self.config.transfer.duration_seconds.max,
        )

        transfer_speed_mbps = (size_mb * 8) / transfer_duration_sec

        case_type = self.pick_case_type(time_slot)
        case_def = CASE_CATALOG[case_type]

        payload = build_logical_payload(
            simulation_date=self.config.simulation.simulation_date,
            sequence=sequence,
            file_type=file_type,
            time_slot=time_slot,
        )

        signature = build_content_signature(
            payload=payload,
            head_length=self.config.hash.hash_head_length,
            tail_length=self.config.hash.hash_tail_length,
        )

        rate = self.config.storage_tier.cost_per_mb_per_month[storage_tier]
        storage_cost = size_mb * rate * (days_stored / 30)

        return {
            "file_id": str(uuid.uuid4()),
            "sequence": sequence,
            "case_type": case_def.case_type,
            "case_group": case_def.case_group,
            "file_type": file_type,
            "size_mb": round(size_mb, 6),
            "storage_tier": storage_tier,
            "days_stored": days_stored,
            "days_since_last_access": days_since_last_access,
            "read_level": read_level,
            "modify_level": modify_level,
            "movement_storage": movement_storage,
            "transfer_duration_sec": round(transfer_duration_sec, 6),
            "transfer_speed_mbps": round(transfer_speed_mbps, 6),
            "day_of_week": day_of_week,
            "time_slot": time_slot.name,
            "created_at": payload["logical_creation_datetime"],
            "hash_head": signature["hash_head"],
            "hash_tail": signature["hash_tail"],
            "error_duplicado": case_def.error_duplicado,
            "error_orphan": case_def.error_orphan,
            "error_null": case_def.error_null,
            "error_blob_timeout": case_def.error_blob_timeout,
            "has_error": case_def.has_error,
            "is_duplicate": case_def.is_duplicate,
            "severity": case_def.severity,
            "storage_cost": round(storage_cost, 10),
        }

    def pick_case_type(self, time_slot: TimeSlotConfig) -> str:
        effective_error_rate = min(
            1.0,
            self.config.simulation.global_error_percentage * time_slot.error_rate,
        )

        if random.random() > effective_error_rate:
            return "UNIQUE_VALID"

        family = random.choices(
            self.config.error_families,
            weights=[
                item.weight_within_global_error
                for item in self.config.error_families
            ],
            k=1,
        )[0]

        subtype = random.choices(
            family.subtypes,
            weights=[item.weight_within_family for item in family.subtypes],
            k=1,
        )[0]

        return subtype.name

    def generate_size_mb(self, file_type: str) -> float:
        size_range = self.config.file_types.size_distribution_mb[file_type]
        return random.uniform(size_range.min, size_range.max)

    @staticmethod
    def pick_weighted(weights: Dict[str, float]) -> str:
        population = list(weights.keys())
        values = list(weights.values())
        return random.choices(population, weights=values, k=1)[0]

    @staticmethod
    def bernoulli(probability: float) -> int:
        return int(random.random() < probability)

    @staticmethod
    def write_csv(path: Path, records: List[Dict[str, Any]]) -> None:
        if not records:
            raise ValueError("No records generated. CSV cannot be written.")

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    @staticmethod
    def case_breakdown(records: List[Dict[str, Any]]) -> Dict[str, int]:
        return SimulationOrchestrator.breakdown(records, "case_type")

    @staticmethod
    def file_type_breakdown(records: List[Dict[str, Any]]) -> Dict[str, int]:
        return SimulationOrchestrator.breakdown(records, "file_type")

    @staticmethod
    def storage_tier_breakdown(records: List[Dict[str, Any]]) -> Dict[str, int]:
        return SimulationOrchestrator.breakdown(records, "storage_tier")

    @staticmethod
    def breakdown(records: List[Dict[str, Any]], field: str) -> Dict[str, int]:
        result: Dict[str, int] = {}

        for record in records:
            value = record[field]
            result[value] = result.get(value, 0) + 1

        return dict(sorted(result.items()))