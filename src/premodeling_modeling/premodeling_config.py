from __future__ import annotations

from dataclasses import dataclass, field


IDENTIFIER_COLUMNS = {
    "file_id",
    "run_id",
    "sequence",
    "content_hash",
    "hash_head",
    "hash_tail",
    "source_file",
}

LEAKAGE_COLUMNS = {
    "severity",
    "is_duplicate",
    "error_duplicado",
    "error_orphan",
    "error_null",
    "error_blob_timeout",
}

DERIVED_COLUMNS = {
    "queue_pressure",
    "congestion_factor",
    "size_range",
}

TARGET_COLUMNS = {
    "transfer_duration_sec",
    "storage_cost",
    "has_error",
}

DEFAULT_CATEGORICAL_COLUMNS = {
    "file_type",
    "storage_tier",
    "time_slot",
    "day_of_week",
    "case_type",
    "case_group",
}

DEFAULT_LOG_TRANSFORM_COLUMNS = {
    "size_mb",
    "transfer_speed_mbps",
    "transfer_duration_sec",
    "storage_cost",
}


@dataclass(frozen=True)
class TargetSpec:
    target: str
    problem_type: str
    candidate_features: list[str]
    log_transform_target: bool = False
    notes: str = ""


@dataclass(frozen=True)
class PremodelingConfig:
    """Configuración declarativa para construir datasets por target.

    No ejecuta modelos. Solo define reglas de preparación.
    """

    target_specs: dict[str, TargetSpec] = field(default_factory=lambda: {
        "transfer_duration_sec": TargetSpec(
            target="transfer_duration_sec",
            problem_type="regression_positive",
            candidate_features=[
                "size_mb",
                "transfer_speed_mbps",
                "created_hour",
                "hourly_arrival_count",
                "hourly_capacity",
                "congestion_factor",
                "file_type",
                "storage_tier",
                "time_slot",
                "day_of_week",
            ],
            log_transform_target=True,
            notes="Duración positiva, asimétrica y sensible a tamaño/carga.",
        ),
        "storage_cost": TargetSpec(
            target="storage_cost",
            problem_type="regression_positive",
            candidate_features=[
                "size_mb",
                "days_stored",
                "days_since_last_access",
                "storage_tier",
                "file_type",
                "created_hour",
            ],
            log_transform_target=True,
            notes="Costo positivo y dominado por tamaño/retención.",
        ),
        "has_error": TargetSpec(
            target="has_error",
            problem_type="classification_binary",
            candidate_features=[
                "size_mb",
                "transfer_speed_mbps",
                "created_hour",
                "hourly_arrival_count",
                "hourly_capacity",
                "congestion_factor",
                "file_type",
                "storage_tier",
                "time_slot",
                "day_of_week",
            ],
            log_transform_target=False,
            notes="Evento binario; no usar componentes de error ni severidad.",
        ),
    })

    identifier_columns: set[str] = field(default_factory=lambda: set(IDENTIFIER_COLUMNS))
    leakage_columns: set[str] = field(default_factory=lambda: set(LEAKAGE_COLUMNS))
    derived_columns: set[str] = field(default_factory=lambda: set(DERIVED_COLUMNS))
    categorical_columns: set[str] = field(default_factory=lambda: set(DEFAULT_CATEGORICAL_COLUMNS))
    log_transform_columns: set[str] = field(default_factory=lambda: set(DEFAULT_LOG_TRANSFORM_COLUMNS))
