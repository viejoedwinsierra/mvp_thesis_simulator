from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CaseDefinition:
    """Logical case definition used to label simulated dataset rows.

    This class does not define whether physical files are created.
    Physical materialization belongs to the generator module.
    """

    case_type: str
    case_group: str

    error_duplicado: int
    error_orphan: int
    error_null: int
    error_blob_timeout: int

    is_duplicate: int
    has_error: int

    severity: float
    base_weight: float
    description: str


CASE_CATALOG: Dict[str, CaseDefinition] = {
    "UNIQUE_VALID": CaseDefinition(
        case_type="UNIQUE_VALID",
        case_group="CORRECT",
        error_duplicado=0,
        error_orphan=0,
        error_null=0,
        error_blob_timeout=0,
        is_duplicate=0,
        has_error=0,
        severity=0.0,
        base_weight=1.0,
        description="Valid unique file without simulated errors.",
    ),

    "DUP_SAME_CONTENT_DIFFERENT_NAME": CaseDefinition(
        case_type="DUP_SAME_CONTENT_DIFFERENT_NAME",
        case_group="DUPLICITY_ERRORS",
        error_duplicado=1,
        error_orphan=0,
        error_null=0,
        error_blob_timeout=0,
        is_duplicate=1,
        has_error=1,
        severity=0.3,
        base_weight=0.40,
        description="Duplicate content represented with a different file name.",
    ),

    "DUP_SAME_CONTENT_DIFFERENT_ROUTE": CaseDefinition(
        case_type="DUP_SAME_CONTENT_DIFFERENT_ROUTE",
        case_group="DUPLICITY_ERRORS",
        error_duplicado=1,
        error_orphan=0,
        error_null=0,
        error_blob_timeout=0,
        is_duplicate=1,
        has_error=1,
        severity=0.4,
        base_weight=0.35,
        description="Duplicate content represented in a different logical route.",
    ),

    "DUP_SAME_CONTENT_DIFFERENT_NAME_AND_ROUTE": CaseDefinition(
        case_type="DUP_SAME_CONTENT_DIFFERENT_NAME_AND_ROUTE",
        case_group="DUPLICITY_ERRORS",
        error_duplicado=1,
        error_orphan=0,
        error_null=0,
        error_blob_timeout=0,
        is_duplicate=1,
        has_error=1,
        severity=0.5,
        base_weight=0.25,
        description="Duplicate content represented with different name and route.",
    ),

    "JSON_WITHOUT_PDF": CaseDefinition(
        case_type="JSON_WITHOUT_PDF",
        case_group="ORPHAN_ERRORS",
        error_duplicado=0,
        error_orphan=1,
        error_null=0,
        error_blob_timeout=0,
        is_duplicate=0,
        has_error=1,
        severity=0.8,
        base_weight=0.60,
        description="Metadata record without its expected paired document.",
    ),

    "PDF_WITHOUT_JSON": CaseDefinition(
        case_type="PDF_WITHOUT_JSON",
        case_group="ORPHAN_ERRORS",
        error_duplicado=0,
        error_orphan=1,
        error_null=0,
        error_blob_timeout=0,
        is_duplicate=0,
        has_error=1,
        severity=0.9,
        base_weight=0.40,
        description="Document record without its expected metadata sidecar.",
    ),

    "NULL_JSON": CaseDefinition(
        case_type="NULL_JSON",
        case_group="NULL_ERRORS",
        error_duplicado=0,
        error_orphan=0,
        error_null=1,
        error_blob_timeout=0,
        is_duplicate=0,
        has_error=1,
        severity=0.7,
        base_weight=1.0,
        description="Record with null or invalid metadata representation.",
    ),

    "BLOB_TIMEOUT": CaseDefinition(
        case_type="BLOB_TIMEOUT",
        case_group="BLOB_STORAGE_ERRORS",
        error_duplicado=0,
        error_orphan=0,
        error_null=0,
        error_blob_timeout=1,
        is_duplicate=0,
        has_error=1,
        severity=0.9,
        base_weight=1.0,
        description="Simulated operational timeout affecting storage ingestion.",
    ),
}