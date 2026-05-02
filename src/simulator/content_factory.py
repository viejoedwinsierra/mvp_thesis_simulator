from __future__ import annotations

from datetime import datetime, timedelta
import hashlib
import json
import random
import string
from typing import Dict, Any

from .config_models import TimeRangeConfig


CUSTOMER_NAMES = [
    "Ana Torres", "Carlos Ruiz", "Laura Diaz", "Juan Perez", "Maria Gomez"
]

DOCUMENT_TYPES = ["invoice", "receipt", "report", "claim"]

CITIES = ["Bogota", "Medellin", "Cali", "Barranquilla", "Bucaramanga"]

COMMENTS = [
    "Synthetic event for thesis MVP.",
    "Generated for statistical and probability analysis.",
    "Structured content for daily simulation.",
    "Document emitted by the orchestrator.",
]


def _random_nonce(length: int = 16) -> str:
    alphabet = string.ascii_letters + string.digits
    return "".join(random.choice(alphabet) for _ in range(length))


def _random_datetime_in_range(
    base_dt: datetime,
    time_range: TimeRangeConfig,
) -> datetime:
    start_hour, start_minute = map(int, time_range.start.split(":"))
    end_hour, end_minute = map(int, time_range.end.split(":"))

    start_dt = base_dt.replace(
        hour=start_hour,
        minute=start_minute,
        second=0,
        microsecond=0,
    )

    end_dt = base_dt.replace(
        hour=end_hour,
        minute=end_minute,
        second=59,
        microsecond=0,
    )

    total_seconds = int((end_dt - start_dt).total_seconds())

    if total_seconds < 0:
        raise ValueError(
            f"Rango horario inválido: {time_range.start} - {time_range.end}"
        )

    return start_dt + timedelta(seconds=random.randint(0, total_seconds))


def build_payload(
    simulation_date: str,
    sequence: int,
    time_range: TimeRangeConfig | None = None,
) -> Dict[str, Any]:
    base_dt = datetime.fromisoformat(simulation_date)

    if time_range is not None:
        dt = _random_datetime_in_range(base_dt, time_range)
    else:
        dt = base_dt + timedelta(minutes=random.randint(0, 1439))

    return {
        "logical_document_id": f"DOC-{base_dt.strftime('%Y%m%d')}-{sequence:06d}",
        "document_type": random.choice(DOCUMENT_TYPES),
        "customer_name": random.choice(CUSTOMER_NAMES),
        "city": random.choice(CITIES),
        "amount": round(random.uniform(50.0, 5000.0), 2),
        "logical_creation_datetime": dt.isoformat(),
        "comment": random.choice(COMMENTS),
        "nonce": _random_nonce(),
    }


def compute_content_hash(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()