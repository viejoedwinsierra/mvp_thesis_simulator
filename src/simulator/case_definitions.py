from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseDefinition:
    """Operational description of a representable case in the simulator."""

    case_type: str
    case_group: str

    pdf_should_exist: bool
    json_should_exist: bool
    json_should_be_valid: bool

    # 🔥 NUEVO
    severity: float  # impacto del error (0–1)
    base_weight: float  # probabilidad relativa dentro del universo

    description: str


CASE_CATALOG = {
    "UNIQUE_VALID": CaseDefinition(
        case_type="UNIQUE_VALID",
        case_group="CORRECT",
        pdf_should_exist=True,
        json_should_exist=True,
        json_should_be_valid=True,
        severity=0.0,
        base_weight=1.0,
        description="Valid unique file with matching PDF and JSON sidecar.",
    ),

    "DUP_SAME_CONTENT_DIFFERENT_NAME": CaseDefinition(
        case_type="DUP_SAME_CONTENT_DIFFERENT_NAME",
        case_group="ERROR",
        pdf_should_exist=True,
        json_should_exist=True,
        json_should_be_valid=True,
        severity=0.3,
        base_weight=0.8,
        description="Duplicate content with a different file name.",
    ),

    "DUP_SAME_CONTENT_DIFFERENT_ROUTE": CaseDefinition(
        case_type="DUP_SAME_CONTENT_DIFFERENT_ROUTE",
        case_group="ERROR",
        pdf_should_exist=True,
        json_should_exist=True,
        json_should_be_valid=True,
        severity=0.4,
        base_weight=0.7,
        description="Duplicate content written to a different logical route.",
    ),

    "DUP_SAME_CONTENT_DIFFERENT_NAME_AND_ROUTE": CaseDefinition(
        case_type="DUP_SAME_CONTENT_DIFFERENT_NAME_AND_ROUTE",
        case_group="ERROR",
        pdf_should_exist=True,
        json_should_exist=True,
        json_should_be_valid=True,
        severity=0.5,
        base_weight=0.6,
        description="Duplicate content with different name and different route.",
    ),

    "JSON_WITHOUT_PDF": CaseDefinition(
        case_type="JSON_WITHOUT_PDF",
        case_group="ERROR",
        pdf_should_exist=False,
        json_should_exist=True,
        json_should_be_valid=True,
        severity=0.8,
        base_weight=0.5,
        description="JSON metadata exists without the corresponding PDF.",
    ),

    "PDF_WITHOUT_JSON": CaseDefinition(
        case_type="PDF_WITHOUT_JSON",
        case_group="ERROR",
        pdf_should_exist=True,
        json_should_exist=False,
        json_should_be_valid=False,
        severity=0.9,
        base_weight=0.5,
        description="PDF exists without the JSON metadata sidecar.",
    ),

    "NULL_JSON": CaseDefinition(
        case_type="NULL_JSON",
        case_group="ERROR",
        pdf_should_exist=True,
        json_should_exist=True,
        json_should_be_valid=False,
        severity=0.7,
        base_weight=0.6,
        description="PDF exists with a null/invalid JSON sidecar.",
    ),

    "BLOB_TIMEOUT": CaseDefinition(
        case_type="BLOB_TIMEOUT",
        case_group="ERROR",
        pdf_should_exist=True,
        json_should_exist=True,
        json_should_be_valid=True,
        severity=0.9,
        base_weight=0.4,
        description="Represents an upload timeout that would affect blob storage.",
    ),
}