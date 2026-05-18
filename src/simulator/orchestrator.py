from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
import math
import csv
import random
import shutil
import uuid

from .case_definitions import build_case_catalog
from .config_models import (
    DatasetSimulationConfig,
    ErrorConditionMultiplier,
    TimeSlotConfig,
    WeekdayTimeDistribution,
)
from .content_factory import build_logical_payload, build_content_signature


class SimulationOrchestrator:
    """Dataset-first simulation orchestrator.

    This orchestrator does not create physical files.
    It only generates the tabular dataset used as the source of truth.
    """

    def __init__(self, config: DatasetSimulationConfig):
        self.config = config
        self.config.validate()

        self.rng = random.Random(self.config.simulation.seed)

        self.noise_config = self.config.noise or {}
        self.realism_config = self.config.realism or {}
        self.correlation_config = self.config.correlations or {}

        self.case_catalog = build_case_catalog(self.config.error_families)
        self.run_id = self.build_run_id()

        self.daily_runtime = self.build_daily_runtime_context()

    def build_run_id(self) -> str:
        simulation_date = self.config.simulation.simulation_date
        seed = self.config.simulation.seed

        if seed is None:
            return f"run_{simulation_date}"

        return f"run_{simulation_date}_seed_{seed}"

    def execute(self) -> dict[str, Any]:
        records = self.generate_records()

        output_dir = Path(self.config.simulation.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        simulation_date = self.config.simulation.simulation_date

        dated_csv_path = output_dir / f"blob_inventory_{simulation_date}.csv"
        latest_csv_path = output_dir / "blob_inventory.csv"

        self.write_csv(dated_csv_path, records)
        shutil.copyfile(dated_csv_path, latest_csv_path)

        total_files = len(records)
        total_errors = sum(1 for r in records if r.get("has_error") == 1)
        total_retries = sum(1 for r in records if r.get("retry_count", 0) > 0)

        return {
            "simulation_date": simulation_date,
            "selected_day": getattr(self.config.simulation, "selected_day", None),
            "scenario": self.get_scenario_name(),
            "run_id": self.run_id,
            "records_created": total_files,
            "total_files": total_files,
            "error_rate": total_errors / total_files if total_files else 0,
            "retry_rate": total_retries / total_files if total_files else 0,
            "dataset_path": str(dated_csv_path),
            "latest_dataset_path": str(latest_csv_path),
            "case_breakdown": self.case_breakdown(records),
            "file_type_breakdown": self.file_type_breakdown(records),
            "storage_tier_breakdown": self.storage_tier_breakdown(records),
            "error_breakdown": self.error_breakdown(records),
            "queue_pressure_stats": self.queue_pressure_stats(records, "queue_pressure"),
            "queue_pressure_raw_stats": self.queue_pressure_stats(records, "queue_pressure_raw"),
            "tier_distribution": self.storage_tier_breakdown(records),
            "operational_costs": self.operational_cost_summary(records),
        }

    def generate_records(self) -> list[dict[str, Any]]:
        weekday_config = self.resolve_weekday_config()
        total_files = self.compute_daily_record_count(weekday_config)

        records: list[dict[str, Any]] = []
        sequence = 1

        for slot in weekday_config.time_distribution:
            slot_hour_weights = self.expand_slot_hours(slot)

            if not slot_hour_weights:
                continue

            expected_slot_count = total_files * slot.percentage_load

            for created_hour, hour_weight in slot_hour_weights:
                expected_hourly_count = expected_slot_count * hour_weight

                lambda_hour = self.apply_hourly_noise(expected_hourly_count)
                hourly_arrival_count = self.poisson(lambda_hour)

                hourly_capacity = self.get_hourly_capacity(hour=created_hour)

                queue_pressure_raw = self.compute_queue_pressure_raw(
                    hourly_arrival_count=hourly_arrival_count,
                    hourly_capacity=hourly_capacity,
                )

                queue_pressure = max(0.0, min(queue_pressure_raw, 1.0))
                congestion_factor = self.compute_congestion_factor(queue_pressure_raw)

                for _ in range(hourly_arrival_count):
                    record = self.build_record(
                        sequence=sequence,
                        day_of_week=weekday_config.day,
                        weekday_base_load=weekday_config.base_load,
                        time_slot=slot,
                        created_hour=created_hour,
                        hourly_arrival_count=hourly_arrival_count,
                        hourly_capacity=hourly_capacity,
                        congestion_factor=congestion_factor,
                        queue_pressure=queue_pressure,
                        queue_pressure_raw=queue_pressure_raw,
                    )

                    records.append(record)
                    sequence += 1

        return records

    def resolve_weekday_config(self) -> WeekdayTimeDistribution:
        simulation_date = datetime.fromisoformat(
            self.config.simulation.simulation_date
        )

        weekday_name = simulation_date.strftime("%A").lower()

        for weekday_config in self.config.weekly_time_distribution:
            if weekday_config.day.lower() == weekday_name:
                return weekday_config

        available_days = [item.day for item in self.config.weekly_time_distribution]

        raise ValueError(
            f"No time distribution configured for weekday={weekday_name}. "
            f"Available days: {available_days}"
        )

    def compute_daily_record_count(
        self,
        weekday_config: WeekdayTimeDistribution,
    ) -> int:
        total = round(
            self.config.simulation.max_valid_files_per_day * weekday_config.base_load
        )

        return max(total, 1)

    def build_record(
        self,
        sequence: int,
        day_of_week: str,
        weekday_base_load: float,
        time_slot: TimeSlotConfig,
        created_hour: int,
        hourly_arrival_count: int,
        hourly_capacity: int,
        congestion_factor: float,
        queue_pressure: float,
        queue_pressure_raw: float,
    ) -> dict[str, Any]:
        file_type = self.pick_weighted(self.config.file_types.distribution)

        size_mb = self.generate_lognormal_value(
            self.config.file_types.size_distribution_mb[file_type]
        )

        size_mb = self.apply_scenario_to_size(size_mb)
        size_mb = self.apply_outlier_size(size_mb)

        days_stored = self.generate_days_stored()
        days_since_last_access = self.generate_days_since_last_access(days_stored)

        storage_tier_original = self.resolve_storage_tier(days_since_last_access)

        transfer_speed_mbps = self.generate_lognormal_value(
            self.config.transfer.speed_mbps
        )

        transfer_speed_mbps = self.apply_scenario_to_transfer_speed(
            transfer_speed_mbps=transfer_speed_mbps,
            queue_pressure=queue_pressure_raw,
            congestion_factor=congestion_factor,
        )

        transfer_duration_sec = self.compute_transfer_duration_sec(
            size_mb=size_mb,
            transfer_speed_mbps=transfer_speed_mbps,
            congestion_factor=congestion_factor,
        )

        transfer_duration_sec = self.apply_congestion_to_duration(
            transfer_duration_sec=transfer_duration_sec,
            congestion_factor=congestion_factor,
        )

        movement_storage = self.bernoulli(
            self.config.lifecycle.movement_storage_probability
        )

        created_at_dt = self.generate_created_at_datetime_for_hour(created_hour)

        payload = build_logical_payload(
            simulation_date=self.config.simulation.simulation_date,
            sequence=sequence,
            file_type=file_type,
            time_slot=time_slot,
            created_at=created_at_dt,
            rng=self.rng,
        )

        signature = self.build_signature(payload)

        record = {
            "file_id": str(uuid.uuid4()),
            "run_id": self.run_id,
            "simulation_date": self.config.simulation.simulation_date,
            "weekday_base_load": weekday_base_load,
            "sequence": sequence,
            "file_type": file_type,
            "size_mb": size_mb,
            "storage_tier_original": storage_tier_original,
            "storage_tier_final": storage_tier_original,
            "storage_tier": storage_tier_original,
            "days_stored": days_stored,
            "days_since_last_access": days_since_last_access,
            "movement_storage": movement_storage,
            "transfer_duration_sec": transfer_duration_sec,
            "transfer_speed_mbps": transfer_speed_mbps,
            "day_of_week": day_of_week,
            "time_slot": time_slot.name,
            "created_at": payload["logical_creation_datetime"],
            "created_hour": created_hour,
            "hourly_arrival_count": hourly_arrival_count,
            "hourly_capacity": hourly_capacity,
            "queue_pressure_raw": queue_pressure_raw,
            "queue_pressure": queue_pressure,
            "congestion_factor": congestion_factor,
            "content_hash": signature["content_hash"],
            "hash_head": signature["hash_head"],
            "hash_tail": signature["hash_tail"],
            "error_probability": 0.0,
            "error_probability_multiplier": 1.0,
            "case_type": "UNIQUE_VALID",
            "case_group": "CORRECT",
            "error_family": None,
            "error_type": None,
            "error_duplicado": 0,
            "error_orphan": 0,
            "error_null": 0,
            "error_blob_timeout": 0,
            "has_error": 0,
            "is_duplicate": 0,
            "severity": 0.0,
            "retry_count": 0,
            "retry_success": 0,
            "retry_delay_sec": 0.0,
            "retry_cost": 0.0,
            "error_penalty_cost": 0.0,
            "storage_cost": 0.0,
            "total_operational_cost": 0.0,
            "has_incident": 0,
            "incident_type": None,
            "metadata_quality_score": 1.0,
            "missing_metadata": 0,
            "wrong_extension": 0,
            "empty_file": 0,
            "invalid_timestamp": 0,
            "malformed_json": 0,
            "unknown_content_type": 0,
            "storage_policy_noise_applied": 0,
        }

        record = self.apply_noise_to_record(record)
        record = self.apply_realism_to_record(record)
        record = self.apply_correlations_to_record(record)

        record["error_probability"] = self.compute_error_probability_from_record(
            record=record,
            time_slot=time_slot,
        )

        record = self.assign_error_from_probability(record)
        record = self.apply_retries(record)
        record = self.apply_storage_policy_noise(record)
        record = self.recalculate_costs(record)
        record = self.round_record(record)

        return record

    def expand_slot_hours(self, time_slot: TimeSlotConfig) -> list[tuple[int, float]]:
        start_hour = int(time_slot.start.split(":")[0])
        end_hour = int(time_slot.end.split(":")[0])

        if end_hour >= start_hour:
            hours = list(range(start_hour, end_hour + 1))
        else:
            hours = list(range(start_hour, 24)) + list(range(0, end_hour + 1))

        n = len(hours)

        # 🔥 generar forma tipo campana (pico en el centro del slot)
        weights = []
        for i in range(n):
            center = (n - 1) / 2
            distance = abs(i - center)

            # menos peso en extremos, más en centro
            weight = 1 - (distance / center) if center > 0 else 1
            weight *= self.rng.uniform(0.9, 1.1)  # ruido leve

            weights.append(weight)

        # normalizar
        total = sum(weights)
        weights = [w / total for w in weights]

        return list(zip(hours, weights))

    def apply_hourly_noise(self, expected_hourly_count: float) -> float:
        arrival_process = getattr(self.config, "arrival_process", None)

        hourly_noise = 0.0
        if arrival_process is not None:
            hourly_noise = getattr(arrival_process, "hourly_noise", 0.10)

        # 🔥 1. Ruido base (normal, no uniforme)
        noise = self.rng.normalvariate(0, hourly_noise / 2)

        # 🔥 2. Eventos ocasionales (picos o caídas)
        event_prob = 0.08  # 8% de horas con evento

        if self.rng.random() < event_prob:
            if self.rng.random() < 0.7:
                # 📈 pico (más común)
                noise += self.rng.uniform(0.15, 0.35)
            else:
                # 📉 caída
                noise -= self.rng.uniform(0.10, 0.25)

        # 🔥 3. Escalar ruido con volumen (más carga = más variabilidad)
        scale_factor = min(expected_hourly_count / 1000, 2.0)
        noise *= (1 + 0.5 * scale_factor)

        # 🔒 4. Clamp para evitar extremos absurdos
        noise = max(min(noise, 0.5), -0.4)

        result = expected_hourly_count * (1 + noise)

        return max(result, 0.0)

    def poisson(self, lambda_value: float) -> int:
        if lambda_value <= 0:
            return 0

        if lambda_value < 50:
            limit = math.exp(-lambda_value)
            k = 0
            product = 1.0

            while product > limit:
                k += 1
                product *= self.rng.random()

            return max(k - 1, 0)

        value = round(self.rng.gauss(lambda_value, math.sqrt(lambda_value)))
        return max(value, 0)

    def compute_congestion_factor(self, queue_pressure: float) -> float:
        if queue_pressure <= 1:
            return 1.0

        overload = queue_pressure - 1

        # 🔥 1. Base no lineal más agresiva
        base = 1 + (overload ** 1.3)

        # 🔥 2. Amplificación progresiva
        if queue_pressure > 1.5:
            base *= self.rng.uniform(1.2, 1.8)

        if queue_pressure > 2.0:
            base *= self.rng.uniform(1.5, 2.5)

        # 🔥 3. Variabilidad (evita dataset artificial)
        noise = self.rng.uniform(0.9, 1.2)

        congestion_factor = base * noise

        # 🔒 4. Clamp (control estadístico)
        return min(congestion_factor, 5.0)

    def apply_congestion_to_duration(
        self,
        transfer_duration_sec: float,
        congestion_factor: float,
    ) -> float:

        capacity = getattr(self.config, "capacity", None)

        if capacity is None or not getattr(capacity, "enabled", False):
            return transfer_duration_sec

        penalty_base = getattr(capacity, "duration_penalty_factor", 0.5)

        # Sin congestión real
        if congestion_factor <= 1:
            return transfer_duration_sec

        overload = congestion_factor - 1

        # 🔥 1. Penalización no lineal (clave)
        nonlinear_penalty = 1 + (overload ** 1.5) * penalty_base

        # 🔥 2. Variabilidad (muy importante)
        noise = self.rng.uniform(0.9, 1.3)

        # 🔥 3. Penalización adicional si hay congestión severa
        if congestion_factor > 1.5:
            spike = self.rng.uniform(1.2, 2.5)
        else:
            spike = 1.0

        adjusted_duration = transfer_duration_sec * nonlinear_penalty * noise * spike

        # 🔒 4. Clamp para evitar explosiones irreales
        return min(adjusted_duration, 3600)

    def generate_created_at_datetime_for_hour(self, created_hour: int) -> datetime:
        base_date = datetime.fromisoformat(self.config.simulation.simulation_date)

        minute = self.rng.randint(0, 59)
        second = self.rng.randint(0, 59)

        return base_date.replace(
            hour=created_hour,
            minute=minute,
            second=second,
            microsecond=0,
        )

    def apply_outlier_size(self, size_mb: float) -> float:
        outliers = getattr(self.config, "outliers", None)

        if outliers is None or not getattr(outliers, "enabled", False):
            return size_mb

        probability = getattr(outliers, "probability", 0.05)

        # 🔥 1. Menor probabilidad real de outlier
        if self.rng.random() >= probability:
            return size_mb

        min_multiplier = getattr(outliers, "size_multiplier_min", 2.0)
        max_multiplier = getattr(outliers, "size_multiplier_max", 10.0)

        # 🔥 2. Distribución sesgada (más realista)
        multiplier = min_multiplier + (
            (max_multiplier - min_multiplier)
            * (self.rng.random() ** 2.5)  # sesgo hacia valores bajos
        )

        new_size = size_mb * multiplier

        # 🔒 3. Clamp (CRÍTICO)
        max_size = getattr(outliers, "max_size_mb", 1500)

        return min(new_size, max_size)
    
    def get_scenario_name(self) -> str | None:
        scenario = getattr(self.config, "scenario", None)

        if scenario is None:
            return None

        return getattr(scenario, "name", None)

    def apply_scenario_to_size(self, size_mb: float) -> float:
        scenario_name = self.get_scenario_name()

        if scenario_name == "large_files":
            return size_mb * self.rng.uniform(1.10, 1.35)

        if scenario_name == "high_load":
            return size_mb * self.rng.uniform(0.95, 1.10)

        return size_mb

    def apply_scenario_to_transfer_speed(
        self,
        transfer_speed_mbps: float,
        queue_pressure: float = 1.0,
        congestion_factor: float = 1.0,
    ) -> float:
        scenario_name = self.get_scenario_name()

        speed = transfer_speed_mbps

        # Penalización por escenario
        if scenario_name == "network_degraded":
            speed *= self.rng.uniform(0.45, 0.75)

        elif scenario_name == "low_capacity":
            speed *= self.rng.uniform(0.70, 0.95)

        elif scenario_name == "large_files":
            speed *= self.rng.uniform(0.90, 1.00)

        elif scenario_name == "high_error":
            speed *= self.rng.uniform(0.85, 1.00)

        # Penalización por presión de cola
        if queue_pressure > 1:
            overload = queue_pressure - 1
            pressure_penalty = 1 / (1 + min(overload * 0.35, 0.75))
            speed *= pressure_penalty

        # Penalización por congestión real
        if congestion_factor > 1:
            congestion_penalty = 1 / (1 + min((congestion_factor - 1) * 0.25, 0.60))
            speed *= congestion_penalty

        # Ruido operativo leve
        speed *= self.rng.uniform(0.95, 1.05)

        return max(speed, 0.5)

    def generate_days_stored(self) -> int:
        return self.rng.randint(
            self.config.lifecycle.days_stored.min,
            self.config.lifecycle.days_stored.max,
        )

    def generate_days_since_last_access(self, days_stored: int) -> int:
        access_cfg = self.config.lifecycle.days_since_last_access

        profile_name = self.pick_weighted(access_cfg.distribution_weights)
        profile = access_cfg.profiles[profile_name]

        # 🔥 1. Ajuste dinámico (muy importante)
        alpha = profile.alpha
        beta = profile.beta

        if days_stored < 7:
            alpha *= 1.5  # más probabilidad de acceso reciente
        elif days_stored > 120:
            beta *= 1.3   # más abandono

        # 🔥 2. Beta distribution
        ratio = self.rng.betavariate(alpha, beta)

        value = days_stored * ratio

        # 🔥 3. Evitar sesgo hacia abajo
        value = int(round(value))

        # 🔥 4. Clamp inferior y superior
        min_days = 0
        max_days = days_stored

        return max(min(value, max_days), min_days)

    def resolve_storage_tier(self, days_since_last_access: int) -> str:
        for rule in self.config.storage_tier.rules:
            min_days = rule.min_days_since_last_access
            max_days = rule.max_days_since_last_access

            if min_days is not None and days_since_last_access < min_days:
                continue

            if max_days is not None and days_since_last_access > max_days:
                continue

            return rule.tier

        raise ValueError(
            "No storage tier rule matched "
            f"days_since_last_access={days_since_last_access}"
        )

    def compute_transfer_duration_sec(
        self,
        size_mb: float,
        transfer_speed_mbps: float,
        congestion_factor: float = 1.0,
    ) -> float:

        if transfer_speed_mbps <= 0:
            raise ValueError("transfer_speed_mbps must be greater than 0.")

        # 1. Tiempo base físico (segundos)
        base_time = (size_mb * 8) / transfer_speed_mbps

        # 2. Latencia fija (red / handshake)
        latency = self.rng.uniform(0.05, 0.3)

        # 3. Overhead proporcional (protocolos, chunks, retries leves)
        overhead_factor = self.rng.uniform(1.02, 1.10)

        # 4. Penalización por congestión (CLAVE)
        if congestion_factor > 1:
            congestion_penalty = 1 + (congestion_factor - 1) * self.rng.uniform(0.4, 0.9)
        else:
            congestion_penalty = 1.0

        # 5. Ruido lognormal (evita colas artificiales extremas)
        noise = self.rng.lognormvariate(0, 0.25)

        duration = base_time * overhead_factor * congestion_penalty * noise + latency

        # 6. Clamp para evitar explosiones irreales
        return min(duration, 3600)  # máximo 1 hora

    def compute_storage_cost(
        self,
        size_mb: float,
        storage_tier: str,
        days_stored: int,
    ) -> float:
        monthly_rate = self.config.storage_tier.cost_per_mb_per_month[storage_tier]
        daily_rate = monthly_rate / 30

        cost = size_mb * daily_rate * days_stored

        adjustments = self.config.cost_model.non_linear_adjustments

        threshold = adjustments.get("large_file_discount_threshold_mb")
        discount_factor = adjustments.get("discount_factor")

        if (
            threshold is not None
            and discount_factor is not None
            and size_mb >= threshold
        ):
            cost *= discount_factor

        return cost

    def apply_scenario_to_error_probability(    
        self,
        probability: float,
        transfer_duration_sec: float,
        queue_pressure: float,
    ) -> float:
        scenario_name = self.get_scenario_name()

        probability = max(0.0, probability)

        duration_factor = 1.0
        pressure_factor = 1.0

        # Penalización por duración por tramos
        if transfer_duration_sec > 600:
            duration_factor *= 1.80
        elif transfer_duration_sec > 300:
            duration_factor *= 1.50
        elif transfer_duration_sec > 120:
            duration_factor *= 1.25
        elif transfer_duration_sec > 60:
            duration_factor *= 1.10

        # Penalización por presión de cola
        if queue_pressure > 1:
            overload = queue_pressure - 1
            pressure_factor *= 1 + min(overload * 0.65, 1.25)

        if scenario_name == "high_error":
            probability *= 1.70

        elif scenario_name == "network_degraded":
            probability *= 1.35
            probability *= duration_factor

        elif scenario_name == "low_capacity":
            probability *= 1.25
            probability *= pressure_factor

        elif scenario_name == "large_files":
            probability *= duration_factor

        else:
            # Incluso en escenario normal, duración y congestión deben influir un poco
            probability *= 1 + min(max(queue_pressure - 1, 0) * 0.20, 0.30)

            if transfer_duration_sec > 300:
                probability *= 1.20

        return min(probability, 0.95)

    def generate_lognormal_value(self, config: Any) -> float:
        # 1. Generar valor base
        value = self.rng.lognormvariate(config.mean, config.sigma)

        # 2. Soft clamp superior (MUY importante)
        max_val = config.max
        if value > max_val:
            # en vez de cortar, comprime suavemente
            excess = value - max_val
            value = max_val + excess * 0.15  # solo deja pasar un 15%

        # 3. Clamp inferior normal
        value = max(value, config.min)

        # 4. Ruido leve adicional (evita patrones)
        value *= self.rng.uniform(0.98, 1.02)

        return value
    
    def pick_weighted(self, weights: Mapping[str, float]) -> str:
        population = list(weights.keys())
        values = list(weights.values())

        return self.rng.choices(
            population=population,
            weights=values,
            k=1,
        )[0]

    def bernoulli(self, probability: float) -> int:
        return int(self.rng.random() < probability)

    def get_hourly_capacity(self, hour: int | None = None) -> int:
        capacity = getattr(self.config, "capacity", None)

        if capacity is None or not getattr(capacity, "enabled", False):
            return 0

        base_capacity = getattr(capacity, "files_per_hour", 800)
        scenario_name = self.get_scenario_name()

        if scenario_name == "low_capacity":
            base_capacity *= self.rng.uniform(0.60, 0.85)
        elif scenario_name == "network_degraded":
            base_capacity *= self.rng.uniform(0.75, 0.95)
        elif scenario_name == "high_error":
            base_capacity *= self.rng.uniform(0.85, 1.00)

        if hour is not None:
            if 0 <= hour < 6:
                base_capacity *= self.rng.uniform(0.80, 0.95)
            elif 6 <= hour < 12:
                base_capacity *= self.rng.uniform(0.90, 1.05)
            elif 12 <= hour < 18:
                base_capacity *= self.rng.uniform(0.95, 1.10)
            else:
                base_capacity *= self.rng.uniform(0.85, 1.00)

        base_capacity *= self.rng.uniform(0.90, 1.10)

        min_cap = getattr(capacity, "min_capacity", 300)
        max_cap = getattr(capacity, "max_capacity", 1500)

        return int(max(min(base_capacity, max_cap), min_cap))

    @staticmethod
    def write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
        if not records:
            raise ValueError("No records generated. CSV cannot be written.")

        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)

    @staticmethod
    def case_breakdown(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return SimulationOrchestrator.breakdown(records, "case_type")

    @staticmethod
    def file_type_breakdown(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return SimulationOrchestrator.breakdown(records, "file_type")

    @staticmethod
    def storage_tier_breakdown(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return SimulationOrchestrator.breakdown(records, "storage_tier")

    @staticmethod
    def error_breakdown(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        total = len(records)

        with_error = sum(1 for r in records if r["has_error"] == 1)

        field_map = {
            "duplicity": "error_duplicado",
            "orphan": "error_orphan",
            "null": "error_null",
            "blob_timeout": "error_blob_timeout",
        }

        error_counts = {k: 0 for k in field_map}
        total_error_flags = 0

        for r in records:
            for key, field in field_map.items():
                if r.get(field, 0) == 1:
                    error_counts[key] += 1
                    total_error_flags += 1

        return {
            "total_records": total,
            "valid_records": total - with_error,
            "records_with_error": with_error,
            "total_error_flags": total_error_flags,
            **error_counts,
        }
    
    @staticmethod
    def breakdown(
        records: Sequence[Mapping[str, Any]],
        field: str,
    ) -> dict[str, int]:
        result: dict[str, int] = {}

        for record in records:
            value = str(record[field])
            result[value] = result.get(value, 0) + 1

        return dict(sorted(result.items()))
    
    def build_daily_runtime_context(self) -> dict[str, Any]:
        realism = self.realism_config or {}

        return {
            "speed_drift_factor": self.sample_daily_speed_drift(realism),
            "error_drift_factor": self.sample_daily_error_drift(realism),
            "incidents": self.sample_daily_incidents(realism),
        }
    
    def sample_daily_speed_drift(self, realism: Mapping[str, Any]) -> float:
        daily_drift = realism.get("daily_drift", {})

        std = daily_drift.get("speed_drift_std", 0.0)

        if std <= 0:
            return 1.0

        return max(0.50, min(self.rng.normalvariate(1.0, std), 1.50))

    def sample_daily_error_drift(self, realism: Mapping[str, Any]) -> float:
        daily_drift = realism.get("daily_drift", {})

        std = daily_drift.get("error_drift_std", 0.0)

        if std <= 0:
            return 1.0

        return max(0.50, min(self.rng.normalvariate(1.0, std), 2.00))

    def sample_daily_incidents(self, realism: Mapping[str, Any]) -> list[dict[str, Any]]:
        incidents_cfg = realism.get("incidents", {})

        probability = incidents_cfg.get("probability_per_simulated_day", 0.0)
        max_incidents = incidents_cfg.get("max_incidents_per_day", 0)
        incident_types = incidents_cfg.get("types", {})

        if probability <= 0 or max_incidents <= 0 or not incident_types:
            return []

        if self.rng.random() >= probability:
            return []

        count = self.rng.randint(1, max_incidents)

        names = list(incident_types.keys())
        weights = [
            incident_types[name].get("weight", 1.0)
            for name in names
        ]

        selected = []

        for _ in range(count):
            name = self.rng.choices(names, weights=weights, k=1)[0]
            cfg = incident_types[name]

            start_hour = self.rng.randint(0, 23)
            duration = self.rng.randint(
                cfg.get("duration_hours_min", 1),
                cfg.get("duration_hours_max", 1),
            )

            selected.append(
                {
                    "type": name,
                    "start_hour": start_hour,
                    "end_hour": min(start_hour + duration, 23),
                    "speed_multiplier": cfg.get("speed_multiplier", 1.0),
                    "capacity_multiplier": cfg.get("capacity_multiplier", 1.0),
                    "duration_multiplier": cfg.get("duration_multiplier", 1.0),
                    "error_multiplier": cfg.get("error_multiplier", 1.0),
                }
            )

        return selected
    
    def compute_queue_pressure_raw(
        self,
        hourly_arrival_count: int,
        hourly_capacity: int,
    ) -> float:
        if hourly_capacity <= 0:
            return 0.0

        return max(hourly_arrival_count / hourly_capacity, 0.0)
    
    def build_signature(self, payload: Mapping[str, Any]) -> dict[str, str | None]:
        hash_config = self.config.hash

        if hash_config is None:
            return {
                "content_hash": None,
                "hash_head": None,
                "hash_tail": None,
            }

        return build_content_signature(
            payload=payload,
            head_length=hash_config.hash_head_length,
            tail_length=hash_config.hash_tail_length,
            use_full_hash=hash_config.use_full_hash,
        )
    
    def apply_noise_to_record(self, record: dict[str, Any]) -> dict[str, Any]:
        noise = self.noise_config or {}

        if not noise.get("enabled", False):
            return record

        profile_name = noise.get("active_profile", "medium")
        profile_multiplier = (
            noise.get("profiles", {})
            .get(profile_name, {})
            .get("multiplier", 1.0)
        )

        variables = noise.get("variables", {})

        record["size_mb"] = self.apply_variable_noise(
            value=record["size_mb"],
            cfg=variables.get("size_mb", {}),
            profile_multiplier=profile_multiplier,
        )

        record["transfer_speed_mbps"] = self.apply_variable_noise(
            value=record["transfer_speed_mbps"],
            cfg=variables.get("transfer_speed_mbps", {}),
            profile_multiplier=profile_multiplier,
        )

        record["days_stored"] = int(
            round(
                self.apply_variable_noise(
                    value=record["days_stored"],
                    cfg=variables.get("days_stored", {}),
                    profile_multiplier=profile_multiplier,
                )
            )
        )

        record["days_since_last_access"] = int(
            round(
                self.apply_variable_noise(
                    value=record["days_since_last_access"],
                    cfg=variables.get("days_since_last_access", {}),
                    profile_multiplier=profile_multiplier,
                )
            )
        )

        record["hourly_capacity"] = int(
            round(
                self.apply_variable_noise(
                    value=record["hourly_capacity"],
                    cfg=variables.get("effective_capacity", {}),
                    profile_multiplier=profile_multiplier,
                )
            )
        )

        record["queue_pressure"] = self.apply_variable_noise(
            value=record["queue_pressure"],
            cfg=variables.get("queue_pressure", {}),
            profile_multiplier=profile_multiplier,
        )

        record["congestion_factor"] = self.apply_variable_noise(
            value=record["congestion_factor"],
            cfg=variables.get("congestion_factor", {}),
            profile_multiplier=profile_multiplier,
        )

        record = self.enforce_record_constraints(record)
        record = self.recalculate_transfer(record)

        return record

    def apply_variable_noise(
        self,
        value: float,
        cfg: Mapping[str, Any],
        profile_multiplier: float,
    ) -> float:
        if not cfg or not cfg.get("enabled", False):
            return value

        noise_type = cfg.get("noise_type")

        if noise_type == "relative_gaussian":
            std_factor = cfg.get("std_factor", 0.0) * profile_multiplier
            value *= self.rng.normalvariate(1.0, std_factor)

        elif noise_type == "residual_gaussian":
            std = cfg.get("std", cfg.get("std_factor", 0.0)) * profile_multiplier
            value += self.rng.normalvariate(0.0, std)

        elif noise_type == "discrete_jitter":
            max_delta = int(cfg.get("max_delta", 0) * profile_multiplier)
            if max_delta > 0:
                value += self.rng.randint(-max_delta, max_delta)

        min_value = cfg.get("min")
        max_value = cfg.get("max")

        if min_value is not None:
            value = max(value, min_value)

        if max_value is not None:
            value = min(value, max_value)

        return value
    
    def apply_realism_to_record(self, record: dict[str, Any]) -> dict[str, Any]:
        realism = self.realism_config or {}

        if not realism.get("enabled", False):
            return record

        record = self.apply_daily_drift(record)
        record = self.apply_incidents(record)
        record = self.apply_metadata_quality(record)
        record = self.apply_saturation(record)
        record = self.enforce_record_constraints(record)
        record = self.recalculate_transfer(record)

        return record
    
    def apply_daily_drift(self, record: dict[str, Any]) -> dict[str, Any]:
        record["transfer_speed_mbps"] *= self.daily_runtime.get(
            "speed_drift_factor",
            1.0,
        )
        record["error_probability_multiplier"] *= self.daily_runtime.get(
            "error_drift_factor",
            1.0,
        )

        return record
    
    def apply_incidents(self, record: dict[str, Any]) -> dict[str, Any]:
        incidents = self.daily_runtime.get("incidents", [])

        for incident in incidents:
            start_hour = incident.get("start_hour", 0)
            end_hour = incident.get("end_hour", 0)

            if start_hour <= record["created_hour"] <= end_hour:
                record["has_incident"] = 1
                record["incident_type"] = incident.get("type")

                record["transfer_speed_mbps"] *= incident.get(
                    "speed_multiplier",
                    1.0,
                )
                record["hourly_capacity"] = int(
                    record["hourly_capacity"]
                    * incident.get("capacity_multiplier", 1.0)
                )
                record["transfer_duration_sec"] *= incident.get(
                    "duration_multiplier",
                    1.0,
                )
                record["error_probability_multiplier"] *= incident.get(
                    "error_multiplier",
                    1.0,
                )

                break

        return record

    def enforce_record_constraints(self, record: dict[str, Any]) -> dict[str, Any]:
        record["size_mb"] = max(float(record.get("size_mb", 0.001)), 0.001)

        record["transfer_speed_mbps"] = max(
            float(record.get("transfer_speed_mbps", 1.0)),
            1.0,
        )

        record["days_stored"] = max(int(record.get("days_stored", 1)), 1)

        record["days_since_last_access"] = max(
            int(record.get("days_since_last_access", 0)),
            0,
        )

        if record["days_since_last_access"] > record["days_stored"]:
            record["days_since_last_access"] = record["days_stored"]

        record["hourly_capacity"] = max(
            int(record.get("hourly_capacity", 1)),
            1,
        )

        record["queue_pressure_raw"] = self.compute_queue_pressure_raw(
            hourly_arrival_count=int(record.get("hourly_arrival_count", 0)),
            hourly_capacity=int(record["hourly_capacity"]),
        )

        record["queue_pressure"] = max(
            0.0,
            min(float(record["queue_pressure_raw"]), 1.0),
        )

        record["congestion_factor"] = max(
            float(record.get("congestion_factor", 1.0)),
            1.0,
        )

        record["error_probability"] = max(
            0.0,
            min(float(record.get("error_probability", 0.0)), 0.95),
        )

        record["error_probability_multiplier"] = max(
            float(record.get("error_probability_multiplier", 1.0)),
            0.0,
        )

        return record

    def recalculate_transfer(self, record: dict[str, Any]) -> dict[str, Any]:
        record["transfer_duration_sec"] = self.compute_transfer_duration_sec(
            size_mb=float(record["size_mb"]),
            transfer_speed_mbps=float(record["transfer_speed_mbps"]),
            congestion_factor=float(record.get("congestion_factor", 1.0)),
        )

        record["transfer_duration_sec"] = self.apply_congestion_to_duration(
            transfer_duration_sec=float(record["transfer_duration_sec"]),
            congestion_factor=float(record.get("congestion_factor", 1.0)),
        )

        return record


    def apply_metadata_quality(self, record: dict[str, Any]) -> dict[str, Any]:
        metadata_cfg = self.realism_config.get("metadata_quality", {})

        if not metadata_cfg:
            return record

        checks = {
            "missing_metadata": metadata_cfg.get("missing_metadata_probability", 0.0),
            "wrong_extension": metadata_cfg.get("wrong_extension_probability", 0.0),
            "empty_file": metadata_cfg.get("empty_file_probability", 0.0),
            "invalid_timestamp": metadata_cfg.get("invalid_timestamp_probability", 0.0),
            "malformed_json": metadata_cfg.get("malformed_json_probability", 0.0),
            "unknown_content_type": metadata_cfg.get(
                "unknown_content_type_probability",
                0.0,
            ),
        }

        failures = 0

        for field, probability in checks.items():
            value = int(self.rng.random() < probability)
            record[field] = value
            failures += value

        record["metadata_quality_score"] = max(0.0, 1.0 - failures * 0.15)

        if failures > 0:
            record["error_probability_multiplier"] *= 1.0 + failures * 0.08

        return record


    def apply_saturation(self, record: dict[str, Any]) -> dict[str, Any]:
        saturation_cfg = self.realism_config.get("saturation", {})

        if not saturation_cfg:
            return record

        threshold = saturation_cfg.get("queue_pressure_threshold", 0.80)

        queue_pressure_raw = float(
            record.get("queue_pressure_raw", record.get("queue_pressure", 0.0))
        )

        if queue_pressure_raw < threshold:
            return record

        overload = max(queue_pressure_raw - threshold, 0.0)

        duration_multiplier_max = saturation_cfg.get("duration_multiplier_max", 4.0)
        error_multiplier_max = saturation_cfg.get("error_multiplier_max", 3.0)
        congestion_multiplier_max = saturation_cfg.get("congestion_multiplier_max", 2.8)

        duration_multiplier = min(
            1.0 + overload ** 1.5,
            duration_multiplier_max,
        )
        error_multiplier = min(
            1.0 + overload,
            error_multiplier_max,
        )
        congestion_multiplier = min(
            1.0 + overload,
            congestion_multiplier_max,
        )

        record["transfer_duration_sec"] *= duration_multiplier
        record["error_probability_multiplier"] *= error_multiplier
        record["congestion_factor"] *= congestion_multiplier

        return record


    def apply_correlations_to_record(self, record: dict[str, Any]) -> dict[str, Any]:
        correlations = self.correlation_config or {}

        if not correlations.get("enabled", False):
            return record

        rules = sorted(
            correlations.get("rules", []),
            key=lambda item: item.get("priority", 999),
        )

        for rule in rules:
            if not rule.get("enabled", True):
                continue

            if not self.evaluate_correlation_rule(record, rule):
                continue

            effects = rule.get("effects", {})

            for field, effect in effects.items():
                operation = effect.get("operation")
                value = effect.get("value")

                target_field = field

                if field == "error_probability":
                    target_field = "error_probability_multiplier"

                if target_field not in record:
                    if target_field == "error_probability_multiplier":
                        record[target_field] = 1.0
                    else:
                        continue

                if operation == "multiply":
                    record[target_field] *= value
                elif operation == "add":
                    record[target_field] += value
                elif operation == "set":
                    record[target_field] = value

        post_processing = correlations.get("post_processing", {})

        clamp_speed = post_processing.get("clamp_transfer_speed_mbps", {})
        if clamp_speed.get("enabled", False):
            record["transfer_speed_mbps"] = max(
                clamp_speed.get("min", 1),
                min(
                    record["transfer_speed_mbps"],
                    clamp_speed.get("max", 300),
                ),
            )

        clamp_queue = post_processing.get("clamp_queue_pressure", {})
        if clamp_queue.get("enabled", False):
            record["queue_pressure"] = max(
                clamp_queue.get("min", 0),
                min(
                    record["queue_pressure"],
                    clamp_queue.get("max", 1),
                ),
            )

        record = self.enforce_record_constraints(record)
        record = self.recalculate_transfer(record)

        return record


    def evaluate_correlation_rule(
        self,
        record: Mapping[str, Any],
        rule: Mapping[str, Any],
    ) -> bool:
        if "when" in rule:
            return self.evaluate_condition(record, rule["when"])

        if "when_all" in rule:
            return all(
                self.evaluate_condition(record, condition)
                for condition in rule["when_all"]
            )

        return False


    def evaluate_condition(
        self,
        record: Mapping[str, Any],
        condition: Mapping[str, Any],
    ) -> bool:
        variable = condition.get("variable")
        operator = condition.get("operator")
        expected = condition.get("value")

        actual = record.get(variable)

        if operator == "==":
            return actual == expected

        if operator == "!=":
            return actual != expected

        if actual is None:
            return False

        try:
            actual_float = float(actual)
            expected_float = float(expected)
        except (TypeError, ValueError):
            return False

        if operator == ">":
            return actual_float > expected_float
        if operator == ">=":
            return actual_float >= expected_float
        if operator == "<":
            return actual_float < expected_float
        if operator == "<=":
            return actual_float <= expected_float

        return False


    def compute_error_probability_from_record(
        self,
        record: Mapping[str, Any],
        time_slot: TimeSlotConfig,
    ) -> float:
        probability = self.config.errors.base_error_probability
        probability *= time_slot.error_multiplier

        multipliers = self.config.errors.multipliers

        large_file = multipliers.get("large_file")

        if isinstance(large_file, ErrorConditionMultiplier):
            if (
                large_file.threshold_mb is not None
                and record["size_mb"] >= large_file.threshold_mb
            ):
                probability *= large_file.multiplier

        low_speed = multipliers.get("low_speed")

        if isinstance(low_speed, ErrorConditionMultiplier):
            if (
                low_speed.threshold_mbps is not None
                and record["transfer_speed_mbps"] <= low_speed.threshold_mbps
            ):
                probability *= low_speed.multiplier

        long_transfer = multipliers.get("long_transfer")

        if isinstance(long_transfer, ErrorConditionMultiplier):
            if (
                long_transfer.threshold_sec is not None
                and record["transfer_duration_sec"] >= long_transfer.threshold_sec
            ):
                probability *= long_transfer.multiplier

        archive_tier = multipliers.get("archive_tier")

        if record["storage_tier"] == "archive":
            if isinstance(archive_tier, ErrorConditionMultiplier):
                probability *= archive_tier.multiplier

            elif isinstance(archive_tier, (int, float)):
                probability *= archive_tier

        movement_multiplier = multipliers.get("movement_storage")

        if record.get("movement_storage") == 1:
            if isinstance(movement_multiplier, ErrorConditionMultiplier):
                probability *= movement_multiplier.multiplier

            elif isinstance(movement_multiplier, (int, float)):
                probability *= movement_multiplier

        probability *= record.get("error_probability_multiplier", 1.0)

        return max(0.0, min(probability, 0.95))


    def assign_error_from_probability(
        self,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        error_probability = record.get("error_probability", 0.0)

        if self.rng.random() > error_probability:
            return self.apply_case_definition(record, "UNIQUE_VALID")

        eligible_families = self.get_eligible_error_families(record)

        if not eligible_families:
            return self.apply_case_definition(record, "UNIQUE_VALID")

        family = self.rng.choices(
            population=eligible_families,
            weights=[
                family.weight_within_global_error
                for family in eligible_families
            ],
            k=1,
        )[0]

        subtype = self.rng.choices(
            population=list(family.subtypes),
            weights=[
                subtype.weight_within_family
                for subtype in family.subtypes
            ],
            k=1,
        )[0]

        return self.apply_case_definition(record, subtype.name)


    def get_eligible_error_families(self, record: Mapping[str, Any]):
        eligible = []

        multipliers = self.config.errors.multipliers

        large_file = multipliers.get("large_file")
        low_speed = multipliers.get("low_speed")

        large_file_condition = True
        low_speed_condition = True

        if isinstance(large_file, ErrorConditionMultiplier):
            large_file_condition = (
                large_file.threshold_mb is not None
                and record["size_mb"] >= large_file.threshold_mb
            )

        if isinstance(low_speed, ErrorConditionMultiplier):
            low_speed_condition = (
                low_speed.threshold_mbps is not None
                and record["transfer_speed_mbps"] <= low_speed.threshold_mbps
            )

        for family in self.config.error_families:
            if family.name == "BLOB_STORAGE_ERRORS":
                if not (large_file_condition and low_speed_condition):
                    continue

            eligible.append(family)

        return eligible

    def apply_case_definition(
        self,
        record: dict[str, Any],
        case_type: str,
    ) -> dict[str, Any]:
        case_def = self.case_catalog[case_type]

        record["case_type"] = case_def.case_type
        record["case_group"] = case_def.case_group
        record["error_family"] = None if case_def.case_group == "CORRECT" else case_def.case_group
        record["error_type"] = None if case_def.case_group == "CORRECT" else case_def.case_type
        record["error_duplicado"] = case_def.error_duplicado
        record["error_orphan"] = case_def.error_orphan
        record["error_null"] = case_def.error_null
        record["error_blob_timeout"] = case_def.error_blob_timeout
        record["has_error"] = case_def.has_error
        record["is_duplicate"] = case_def.is_duplicate
        record["severity"] = case_def.severity

        return record

    def apply_retries(self, record: dict[str, Any]) -> dict[str, Any]:
        retries_cfg = self.realism_config.get("retries", {})

        if not retries_cfg or record.get("has_error", 0) != 1:
            return record

        max_retries = int(retries_cfg.get("max_retries", 0))
        retry_probability = retries_cfg.get("retry_probability_after_error", 0.0)
        retry_success_probability = retries_cfg.get("retry_success_probability", 0.0)

        if max_retries <= 0 or self.rng.random() >= retry_probability:
            return record

        retry_count = self.rng.randint(1, max_retries)
        retry_success = int(self.rng.random() < retry_success_probability)

        delay_cfg = retries_cfg.get("retry_delay_sec", {})
        retry_delay_sec = self.sample_retry_delay(delay_cfg, retry_count)

        record["retry_count"] = retry_count
        record["retry_success"] = retry_success
        record["retry_delay_sec"] = retry_delay_sec

        side_effects = retries_cfg.get("retry_side_effects", {})

        queue_increase = side_effects.get("increase_queue_pressure", 0.0) * retry_count
        duration_multiplier = (
            side_effects.get("increase_duration_multiplier", 1.0) ** retry_count
        )

        record["queue_pressure_raw"] = max(
            float(record.get("queue_pressure_raw", 0.0)),
            float(record.get("queue_pressure", 0.0)) + queue_increase,
        )

        record["queue_pressure"] = max(
            0.0,
            min(float(record["queue_pressure_raw"]), 1.0),
        )

        record["transfer_duration_sec"] *= duration_multiplier

        record["transfer_duration_sec"] += retry_delay_sec

        return record

    def sample_retry_delay(
        self,
        delay_cfg: Mapping[str, Any],
        retry_count: int,
    ) -> float:
        if delay_cfg.get("distribution") == "exponential":
            lambda_value = delay_cfg.get("lambda", 0.08)
            min_value = delay_cfg.get("min", 1)
            max_value = delay_cfg.get("max", 120)

            total_delay = 0.0

            for _ in range(retry_count):
                total_delay += self.rng.expovariate(lambda_value)

            return max(min(total_delay, max_value), min_value)

        return float(retry_count)

    def apply_storage_policy_noise(
        self,
        record: dict[str, Any],
    ) -> dict[str, Any]:
        policy_cfg = self.realism_config.get("storage_policy_noise", {})

        if not policy_cfg:
            return record

        available_tiers = set(
            self.config.storage_tier.cost_per_mb_per_month.keys()
        )

        if not available_tiers:
            return record

        current_tier = record.get("storage_tier_final", record.get("storage_tier"))

        if current_tier not in available_tiers:
            current_tier = record.get("storage_tier_original")

        if current_tier not in available_tiers:
            return record

        new_tier = current_tier

        if self.rng.random() < policy_cfg.get("wrong_tier_probability", 0.0):
            alternatives = [
                tier
                for tier in available_tiers
                if tier != current_tier
            ]

            if alternatives:
                new_tier = self.rng.choice(alternatives)

        if (
            "hot" in available_tiers
            and record["size_mb"] >= 100
            and self.rng.random() < policy_cfg.get(
                "force_hot_large_files_probability",
                0.0,
            )
        ):
            new_tier = "hot"

        if (
            "archive" in available_tiers
            and record["days_since_last_access"] <= 7
            and self.rng.random() < policy_cfg.get(
                "archive_recent_file_probability",
                0.0,
            )
        ):
            new_tier = "archive"

        if (
            "cool" in available_tiers
            and record["days_since_last_access"] <= 30
            and self.rng.random() < policy_cfg.get(
                "cool_recent_file_probability",
                0.0,
            )
        ):
            new_tier = "cool"

        if new_tier != current_tier:
            record["storage_policy_noise_applied"] = 1
            record["storage_tier_final"] = new_tier
            record["storage_tier"] = new_tier
        else:
            record["storage_policy_noise_applied"] = 0
            record["storage_tier_final"] = current_tier
            record["storage_tier"] = current_tier

        return record

    def recalculate_costs(self, record: dict[str, Any]) -> dict[str, Any]:
        storage_cost = self.compute_storage_cost(
            size_mb=record["size_mb"],
            storage_tier=record["storage_tier_final"],
            days_stored=record["days_stored"],
        )

        retry_cost = 0.0
        error_penalty_cost = 0.0

        operational_cfg = (
            self.config.raw_cost_config
            .get("cost_model", {})
            .get("operational_costs", {})
            if self.config.raw_cost_config
            else {}
        )

        if operational_cfg.get("enabled", False):
            retry_cost = (
                record.get("retry_count", 0)
                * operational_cfg.get("retry_cost_per_attempt", 0.0)
            )

            if record.get("error_type") == "BLOB_TIMEOUT":
                error_penalty_cost += operational_cfg.get(
                    "timeout_penalty_cost",
                    0.0,
                )

            if record.get("error_family") == "ORPHAN_ERRORS":
                error_penalty_cost += operational_cfg.get(
                    "orphan_penalty_cost",
                    0.0,
                )

            if record.get("error_family") == "DUPLICITY_ERRORS":
                error_penalty_cost += operational_cfg.get(
                    "duplicity_penalty_cost",
                    0.0,
                )

        record["storage_cost"] = storage_cost
        record["retry_cost"] = retry_cost
        record["error_penalty_cost"] = error_penalty_cost
        record["total_operational_cost"] = (
            storage_cost + retry_cost + error_penalty_cost
        )

        return record


    def round_record(self, record: dict[str, Any]) -> dict[str, Any]:
        float_fields = [
            "size_mb",
            "transfer_duration_sec",
            "transfer_speed_mbps",
            "queue_pressure",
            "queue_pressure_raw",
            "congestion_factor",
            "error_probability",
            "error_probability_multiplier",
            "retry_delay_sec",
            "retry_cost",
            "error_penalty_cost",
            "storage_cost",
            "total_operational_cost",
            "metadata_quality_score",
        ]

        for field in float_fields:
            if field in record and record[field] is not None:
                record[field] = round(float(record[field]), 6)

        if "storage_cost" in record:
            record["storage_cost"] = round(float(record["storage_cost"]), 10)

        if "total_operational_cost" in record:
            record["total_operational_cost"] = round(
                float(record["total_operational_cost"]),
                10,
            )

        return record


    @staticmethod
    def queue_pressure_stats(
        records: Sequence[Mapping[str, Any]],
        field: str = "queue_pressure",
    ) -> dict[str, float]:
        values = sorted(float(r.get(field, 0.0)) for r in records)

        if not values:
            return {
                "min": 0.0,
                "mean": 0.0,
                "max": 0.0,
                "p95": 0.0,
            }

        p95_index = min(int(len(values) * 0.95), len(values) - 1)

        return {
            "min": values[0],
            "mean": sum(values) / len(values),
            "max": values[-1],
            "p95": values[p95_index],
        }

    @staticmethod
    def operational_cost_summary(
        records: Sequence[Mapping[str, Any]],
    ) -> dict[str, float]:

        storage_cost_total = sum(
            float(r.get("storage_cost", 0.0))
            for r in records
        )

        retry_cost_total = sum(
            float(r.get("retry_cost", 0.0))
            for r in records
        )

        error_penalty_cost_total = sum(
            float(r.get("error_penalty_cost", 0.0))
            for r in records
        )

        total_operational_cost = sum(
            float(r.get("total_operational_cost", 0.0))
            for r in records
        )

        return {
            "storage_cost_total": round(storage_cost_total, 10),
            "retry_cost_total": round(retry_cost_total, 10),
            "error_penalty_cost_total": round(
                error_penalty_cost_total,
                10,
            ),
            "total_operational_cost": round(
                total_operational_cost,
                10,
            ),
        }