from __future__ import annotations

from typing import Dict
import math


EPSILON = 1e-12


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize positive weights to sum to 1.0."""

    if not weights:
        return {}

    for k, v in weights.items():
        if v < 0:
            raise ValueError(f"Weight cannot be negative: {k}={v}")

    total = sum(weights.values())

    if total <= EPSILON:
        raise ValueError("The sum of weights must be positive and non-zero.")

    return {key: value / total for key, value in weights.items()}


def largest_remainder_allocation(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    """Allocate an integer total using the largest remainder method."""

    if total < 0:
        raise ValueError("Total cannot be negative.")

    if not weights:
        return {}

    normalized = normalize_weights(weights)

    raw = {k: total * w for k, w in normalized.items()}
    base = {k: math.floor(v) for k, v in raw.items()}

    allocated = sum(base.values())
    residual = total - allocated

    if residual == 0:
        return base

    remainders = sorted(
        ((k, raw[k] - base[k]) for k in raw),
        key=lambda pair: pair[1],
        reverse=True,
    )

    for i in range(residual):
        key = remainders[i % len(remainders)][0]  # 🔥 protección
        base[key] += 1

    return base