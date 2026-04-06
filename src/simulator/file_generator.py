from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import hashlib
import json


def build_file_stem(payload: Dict[str, Any], sequence: int) -> str:
    doc_type = payload["document_type"]
    logical_dt = payload["logical_creation_datetime"].replace(":", "").replace("-", "")[:15]
    return f"{doc_type}_{logical_dt}_{sequence:06d}"



def write_pdf_placeholder(pdf_path: Path, payload: Dict[str, Any]) -> str:
    """Create a lightweight placeholder file with .pdf extension.

    For the MVP thesis presentation this is sufficient because the focus is on
    the universe structure, the metadata, and the probabilistic allocation.
    """
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        "MVP SYNTHETIC PDF PLACEHOLDER\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
        + "\n"
    )
    pdf_path.write_text(content, encoding="utf-8")
    return hashlib.sha256(content.encode("utf-8")).hexdigest()
