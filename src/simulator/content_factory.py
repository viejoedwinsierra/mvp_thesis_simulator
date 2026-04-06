from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta
import hashlib
import json
import random
import string
from typing import Dict, Any


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



def build_payload(simulation_date: str, sequence: int) -> Dict[str, Any]:
    """Create a canonical content payload for one logical document.

    The payload is the basis for the content hash. This is more suitable than
    relying on pure combinatorial enumeration because the simulator aims to
    generate a probabilistic universe with weighted categories rather than only
    count possible arrangements.
    """
    base_dt = datetime.fromisoformat(simulation_date)
    dt = base_dt + timedelta(minutes=random.randint(0, 1439))

    payload = {
        "logical_document_id": f"DOC-{base_dt.strftime('%Y%m%d')}-{sequence:06d}",
        "document_type": random.choice(DOCUMENT_TYPES),
        "customer_name": random.choice(CUSTOMER_NAMES),
        "city": random.choice(CITIES),
        "amount": round(random.uniform(50.0, 5000.0), 2),
        "logical_creation_datetime": dt.isoformat(),
        "comment": random.choice(COMMENTS),
        "nonce": _random_nonce(),
    }
    return payload



def compute_content_hash(payload: Dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
