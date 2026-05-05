from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import math
import random
import shutil
import uuid
import numpy as np

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
        self.case_catalog = build_case_catalog(self.config.error_families)
        self.run_id = self.build_run_id()

    def build_run_id(self) -> str:
        simulation_date = self.config.simulation.simulation_date
        seed = self.config.simulation.seed

        if seed is None:
            return f"run_{simulation_date}"

        return f"run_{simulation_date}_seed_{seed}"

    def execute(self) -> dict[str, Any]:
        records = self.generate_records()

        output_dir = Path(self.config.simulation.output_dir) / "dataset"
        output_dir.mkdir(parents=True, exist_ok=True)

        simulation_date = self.config.simulation.simulation_date

        dated_csv_path = output_dir / f"blob_inventory_{simulation_date}.csv"
        latest_csv_path = output_dir / "blob_inventory.csv"

        self.write_csv(dated_csv_path, records)
        shutil.copyfile(dated_csv_path, latest_csv_path)

        return {
            "simulation_date": simulation_date,
            "run_id": self.run_id,
            "records_created": len(records),
            "dataset_path": str(dated_csv_path),
            "latest_dataset_path": str(latest_csv_path),
            "case_breakdown": self.case_breakdown(records),
            "file_type_breakdown": self.file_type_breakdown(records),
            "storage_tier_breakdown": self.storage_tier_breakdown(records),
            "error_breakdown": self.error_breakdown(records),
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

                queue_pressure = self.compute_queue_pressure(
                    hourly_arrival_count=hourly_arrival_count,
                    hourly_capacity=hourly_capacity,
                )

                congestion_factor = self.compute_congestion_factor(queue_pressure)

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
        average_daily_load = 1 / 7
        load_factor = weekday_config.base_load / average_daily_load

        total = round(
            self.config.simulation.max_valid_files_per_day * load_factor
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
    ) -> dict[str, Any]:
        file_type = self.pick_weighted(self.config.file_types.distribution)

        size_mb = self.generate_lognormal_value(
            self.config.file_types.size_distribution_mb[file_type]
        )

        size_mb = self.apply_scenario_to_size(size_mb)
        size_mb = self.apply_outlier_size(size_mb)

        days_stored = self.generate_days_stored()
        days_since_last_access = self.generate_days_since_last_access(days_stored)

        storage_tier = self.resolve_storage_tier(days_since_last_access)

        transfer_speed_mbps = self.generate_lognormal_value(
            self.config.transfer.speed_mbps
        )

        transfer_speed_mbps = self.apply_scenario_to_transfer_speed(
            transfer_speed_mbps
        )

        transfer_duration_sec = self.compute_transfer_duration_sec(
            size_mb=size_mb,
            transfer_speed_mbps=transfer_speed_mbps,
        )

        transfer_duration_sec = self.apply_congestion_to_duration(
            transfer_duration_sec=transfer_duration_sec,
            congestion_factor=congestion_factor,
        )

        movement_storage = self.bernoulli(
            self.config.lifecycle.movement_storage_probability
        )

        case_type = self.pick_case_type(
            size_mb=size_mb,
            transfer_speed_mbps=transfer_speed_mbps,
            transfer_duration_sec=transfer_duration_sec,
            storage_tier=storage_tier,
            time_slot=time_slot,
            movement_storage=movement_storage,
            congestion_factor=congestion_factor,
            queue_pressure=queue_pressure,
        )

        case_def = self.case_catalog[case_type]

        created_at_dt = self.generate_created_at_datetime_for_hour(created_hour)

        payload = build_logical_payload(
            simulation_date=self.config.simulation.simulation_date,
            sequence=sequence,
            file_type=file_type,
            time_slot=time_slot,
            created_at=created_at_dt,
            rng=self.rng,
        )

        signature = build_content_signature(
            payload=payload,
            head_length=self.config.hash.hash_head_length,
            tail_length=self.config.hash.hash_tail_length,
            use_full_hash=self.config.hash.use_full_hash,
        )

        storage_cost = self.compute_storage_cost(
            size_mb=size_mb,
            storage_tier=storage_tier,
            days_stored=days_stored,
        )

        return {
            "file_id": str(uuid.uuid4()),
            "run_id": self.run_id,
            "simulation_date": self.config.simulation.simulation_date,
            "weekday_base_load": weekday_base_load,
            "sequence": sequence,
            "case_type": case_def.case_type,
            "case_group": case_def.case_group,
            "file_type": file_type,
            "size_mb": round(size_mb, 6),
            "storage_tier": storage_tier,
            "days_stored": days_stored,
            "days_since_last_access": days_since_last_access,
            "movement_storage": movement_storage,
            "transfer_duration_sec": round(transfer_duration_sec, 6),
            "transfer_speed_mbps": round(transfer_speed_mbps, 6),
            "day_of_week": day_of_week,
            "time_slot": time_slot.name,
            "created_at": payload["logical_creation_datetime"],
            "created_hour": created_hour,
            "hourly_arrival_count": hourly_arrival_count,
            "hourly_capacity": hourly_capacity,
            "congestion_factor": round(congestion_factor, 6),
            "queue_pressure": round(queue_pressure, 6),
            "content_hash": signature["content_hash"],
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

    def compute_queue_pressure(
        self,
        hourly_arrival_count: int,
        hourly_capacity: int,
    ) -> float:
        if hourly_capacity <= 0:
            return 0.0

        base_ratio = hourly_arrival_count / hourly_capacity

        # 🔥 1. Suavizado (evita extremos lineales)
        if base_ratio > 1:
            # compresión leve para evitar crecimiento exagerado
            adjusted = 1 + (base_ratio - 1) ** 0.85
        else:
            adjusted = base_ratio

        # 🔥 2. Ruido operativo (clave para realismo)
        noise = self.rng.uniform(0.95, 1.10)
        adjusted *= noise

        # 🔥 3. Saturación superior (sistema no crece infinito)
        max_pressure = 3.0
        adjusted = min(adjusted, max_pressure)

        return adjusted

    def compute_congestion_factor(self, queue_pressure: float) -> float:
        if queue_pressure <= 1:
            return 1.0

        overload = queue_pressure - 1

        # 🔥 1. Base no lineal más agresiva
        base = 1 + (overload ** 1.3)

        # 🔥 2. Amplificación progresiva
        if queue_pressure > 1.5:
            base *= random.uniform(1.2, 1.8)

        if queue_pressure > 2.0:
            base *= random.uniform(1.5, 2.5)

        # 🔥 3. Variabilidad (evita dataset artificial)
        noise = random.uniform(0.9, 1.2)

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
        noise = random.uniform(0.9, 1.3)

        # 🔥 3. Penalización adicional si hay congestión severa
        if congestion_factor > 1.5:
            spike = random.uniform(1.2, 2.5)
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
        latency = random.uniform(0.05, 0.3)

        # 3. Overhead proporcional (protocolos, chunks, retries leves)
        overhead_factor = random.uniform(1.02, 1.10)

        # 4. Penalización por congestión (CLAVE)
        if congestion_factor > 1:
            congestion_penalty = 1 + (congestion_factor - 1) * random.uniform(0.4, 0.9)
        else:
            congestion_penalty = 1.0

        # 5. Ruido lognormal (evita colas artificiales extremas)
        noise = np.random.lognormal(mean=0, sigma=0.25)

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

    def pick_case_type(
        self,
        size_mb: float,
        transfer_speed_mbps: float,
        transfer_duration_sec: float,
        storage_tier: str,
        time_slot: TimeSlotConfig,
        movement_storage: int,
        congestion_factor: float,
        queue_pressure: float,
    ) -> str:
        error_probability = self.compute_error_probability(
            size_mb=size_mb,
            transfer_speed_mbps=transfer_speed_mbps,
            transfer_duration_sec=transfer_duration_sec,
            storage_tier=storage_tier,
            time_slot=time_slot,
            movement_storage=movement_storage,
            congestion_factor=congestion_factor,
            queue_pressure=queue_pressure,
        )

        if self.rng.random() > error_probability:
            return "UNIQUE_VALID"

        family = self.rng.choices(
            population=list(self.config.error_families),
            weights=[
                item.weight_within_global_error
                for item in self.config.error_families
            ],
            k=1,
        )[0]

        subtype = self.rng.choices(
            population=list(family.subtypes),
            weights=[
                item.weight_within_family
                for item in family.subtypes
            ],
            k=1,
        )[0]

        return subtype.name

    def compute_error_probability(
        self,
        size_mb: float,
        transfer_speed_mbps: float,
        transfer_duration_sec: float,
        storage_tier: str,
        time_slot: TimeSlotConfig,
        movement_storage: int,
        congestion_factor: float,
        queue_pressure: float,
    ) -> float:
        probability = self.config.errors.base_error_probability
        probability *= time_slot.error_multiplier

        multipliers = self.config.errors.multipliers

        large_file = multipliers.get("large_file")
        if isinstance(large_file, ErrorConditionMultiplier):
            if (
                large_file.threshold_mb is not None
                and size_mb >= large_file.threshold_mb
            ):
                probability *= large_file.multiplier

        low_speed = multipliers.get("low_speed")
        if isinstance(low_speed, ErrorConditionMultiplier):
            if (
                low_speed.threshold_mbps is not None
                and transfer_speed_mbps <= low_speed.threshold_mbps
            ):
                probability *= low_speed.multiplier

        long_transfer = multipliers.get("long_transfer")
        if isinstance(long_transfer, ErrorConditionMultiplier):
            if (
                long_transfer.threshold_sec is not None
                and transfer_duration_sec >= long_transfer.threshold_sec
            ):
                probability *= long_transfer.multiplier

        archive_tier = multipliers.get("archive_tier")
        if storage_tier == "archive" and isinstance(archive_tier, (int, float)):
            probability *= archive_tier

        peak_hour = multipliers.get("peak_hour")
        if time_slot.error_multiplier >= 1.5 and isinstance(peak_hour, (int, float)):
            probability *= peak_hour

        movement_multiplier = multipliers.get("movement_storage")
        if movement_storage == 1 and isinstance(movement_multiplier, (int, float)):
            probability *= movement_multiplier

        congestion_multiplier = multipliers.get("congestion")
        if congestion_factor > 1 and isinstance(congestion_multiplier, (int, float)):
            capacity = getattr(self.config, "capacity", None)
            penalty_factor = 1.0

            if capacity is not None:
                penalty_factor += (
                    (congestion_factor - 1)
                    * getattr(capacity, "error_penalty_factor", 0.0)
                )

            probability *= congestion_multiplier
            probability *= penalty_factor

        if queue_pressure > 1.25:
            probability *= 1.15

        if queue_pressure > 1.50:
            probability *= 1.25

        probability = self.apply_scenario_to_error_probability(
            probability=probability,
            transfer_duration_sec=transfer_duration_sec,
            queue_pressure=queue_pressure,
        )

        return min(max(probability, 0.0), 1.0)

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

        return min(probability, 0.65)

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