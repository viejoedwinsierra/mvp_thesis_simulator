from __future__ import annotations

from pathlib import Path
from typing import Dict, Any
import hashlib
import json
import random

from .config_models import TimeRangeConfig


def build_file_stem(
    payload: Dict[str, Any],
    sequence: int,
    time_range: TimeRangeConfig | None = None,
) -> str:
    doc_type = payload["document_type"]
    logical_dt = payload["logical_creation_datetime"]
    logical_dt = logical_dt.replace(":", "").replace("-", "")[:15]

    if time_range is not None:
        return f"{doc_type}_{time_range.name}_{logical_dt}_{sequence:06d}"

    return f"{doc_type}_{logical_dt}_{sequence:06d}"


def write_pdf_placeholder(
    pdf_path: Path,
    payload: Dict[str, Any],
    time_range: TimeRangeConfig | None = None,
) -> str:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    size = random.randint(500, 1500)

    range_name = time_range.name if time_range is not None else "unknown"
    error_rate = time_range.error_rate if time_range is not None else 0.0

    content = (
        "MVP PDF PLACEHOLDER\n"
        f"time_range={range_name}\n"
        f"error_rate={error_rate}\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
        + "\n"
        + ("X" * size)
    )

    pdf_path.write_text(content, encoding="utf-8")

    return hashlib.sha256(content.encode("utf-8")).hexdigest()