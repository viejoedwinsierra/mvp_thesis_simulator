from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CaseDefinition:
    """Operational description of a representable case in the simulator.

    The object states what physical artifacts should exist after execution.
    This is useful both for the thesis presentation and for the future replay
    module that will reconstruct scenarios from metadata.
    """

    case_type: str
    case_group: str
    pdf_should_exist: bool
    json_should_exist: bool
    json_should_be_valid: bool
    description: str


CASE_CATALOG = {
    "UNIQUE_VALID": CaseDefinition(
        case_type="UNIQUE_VALID",
        case_group="CORRECT",
        pdf_should_exist=True,
        json_should_exist=True,
        json_should_be_valid=True,
        description="Valid unique file with matching PDF and JSON sidecar.",
    ),
    "DUP_SAME_CONTENT_DIFFERENT_NAME": CaseDefinition(
        case_type="DUP_SAME_CONTENT_DIFFERENT_NAME",
        case_group="ERROR",
        pdf_should_exist=True,
        json_should_exist=True,
        json_should_be_valid=True,
        description="Duplicate content with a different file name.",
    ),
    "DUP_SAME_CONTENT_DIFFERENT_ROUTE": CaseDefinition(
        case_type="DUP_SAME_CONTENT_DIFFERENT_ROUTE",
        case_group="ERROR",
        pdf_should_exist=True,
        json_should_exist=True,
        json_should_be_valid=True,
        description="Duplicate content written to a different logical route.",
    ),
    "DUP_SAME_CONTENT_DIFFERENT_NAME_AND_ROUTE": CaseDefinition(
        case_type="DUP_SAME_CONTENT_DIFFERENT_NAME_AND_ROUTE",
        case_group="ERROR",
        pdf_should_exist=True,
        json_should_exist=True,
        json_should_be_valid=True,
        description="Duplicate content with different name and different route.",
    ),
    "JSON_WITHOUT_PDF": CaseDefinition(
        case_type="JSON_WITHOUT_PDF",
        case_group="ERROR",
        pdf_should_exist=False,
        json_should_exist=True,
        json_should_be_valid=True,
        description="JSON metadata exists without the corresponding PDF.",
    ),
    "PDF_WITHOUT_JSON": CaseDefinition(
        case_type="PDF_WITHOUT_JSON",
        case_group="ERROR",
        pdf_should_exist=True,
        json_should_exist=False,
        json_should_be_valid=False,
        description="PDF exists without the JSON metadata sidecar.",
    ),
    "NULL_JSON": CaseDefinition(
        case_type="NULL_JSON",
        case_group="ERROR",
        pdf_should_exist=True,
        json_should_exist=True,
        json_should_be_valid=False,
        description="PDF exists with a null/invalid JSON sidecar.",
    ),
    "BLOB_TIMEOUT": CaseDefinition(
        case_type="BLOB_TIMEOUT",
        case_group="ERROR",
        pdf_should_exist=True,
        json_should_exist=True,
        json_should_be_valid=True,
        description="Represents an upload timeout that would affect blob storage.",
    ),
}
