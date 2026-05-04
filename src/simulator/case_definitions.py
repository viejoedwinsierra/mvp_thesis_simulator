from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


# --------------------------------------------------
# Core domain model
# --------------------------------------------------

@dataclass(frozen=True, slots=True)
class CaseDefinition:
    """Logical case definition used to label simulated dataset rows.

    This class describes semantic labels and error indicators only.
    It does not define whether physical files are created.
    """

    case_type: str
    case_group: str

    error_duplicado: int = 0
    error_orphan: int = 0
    error_null: int = 0
    error_blob_timeout: int = 0

    is_duplicate: int = 0
    has_error: int = 0

    severity: float = 0.0
    base_weight: float = 1.0
    description: str = ""

    def __post_init__(self) -> None:
        # -------------------------
        # Binary validation
        # -------------------------
        flag_fields = (
            "error_duplicado",
            "error_orphan",
            "error_null",
            "error_blob_timeout",
            "is_duplicate",
            "has_error",
        )

        for field_name in flag_fields:
            value = getattr(self, field_name)
            if value not in (0, 1):
                raise ValueError(
                    f"{field_name} must be 0 or 1 for case_type={self.case_type}"
                )

        # -------------------------
        # Numeric validation
        # -------------------------
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(
                f"severity must be between 0 and 1 for case_type={self.case_type}"
            )

        if self.base_weight < 0:
            raise ValueError(
                f"base_weight cannot be negative for case_type={self.case_type}"
            )

        # -------------------------
        # Logical consistency
        # -------------------------
        expected_has_error = int(self.error_count > 0)

        if self.has_error != expected_has_error:
            raise ValueError(
                f"has_error={self.has_error} is inconsistent with error flags "
                f"for case_type={self.case_type}"
            )

        if self.is_duplicate != self.error_duplicado:
            raise ValueError(
                f"is_duplicate={self.is_duplicate} is inconsistent with "
                f"error_duplicado={self.error_duplicado} "
                f"for case_type={self.case_type}"
            )

    @property
    def error_count(self) -> int:
        return (
            self.error_duplicado
            + self.error_orphan
            + self.error_null
            + self.error_blob_timeout
        )

    @property
    def is_valid(self) -> bool:
        return self.has_error == 0


# --------------------------------------------------
# Dynamic catalog builder
# --------------------------------------------------

def build_case_catalog(error_families) -> Mapping[str, CaseDefinition]:
    """Build case definitions dynamically from configuration.

    Eliminates duplication between config and code.
    """

    catalog: dict[str, CaseDefinition] = {}

    # -------------------------
    # Base case (no error)
    # -------------------------
    catalog["UNIQUE_VALID"] = CaseDefinition(
        case_type="UNIQUE_VALID",
        case_group="CORRECT",
        severity=0.0,
        base_weight=1.0,
        description="Valid unique file without simulated errors.",
    )

    # -------------------------
    # Error-driven cases
    # -------------------------
    for family in error_families:
        for subtype in family.subtypes:

            is_dup = int(family.name == "DUPLICITY_ERRORS")
            is_orphan = int(family.name == "ORPHAN_ERRORS")
            is_null = int(family.name == "NULL_ERRORS")
            is_timeout = int(family.name == "BLOB_STORAGE_ERRORS")

            catalog[subtype.name] = CaseDefinition(
                case_type=subtype.name,
                case_group=family.name,
                error_duplicado=is_dup,
                error_orphan=is_orphan,
                error_null=is_null,
                error_blob_timeout=is_timeout,
                is_duplicate=is_dup,
                has_error=1,
                severity=subtype.severity,
                base_weight=subtype.weight_within_family,
                description=f"{family.name} - {subtype.name}",
            )

    return catalog