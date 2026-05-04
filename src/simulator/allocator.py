from __future__ import annotations

from typing import Dict
import math


EPSILON = 1e-12


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize non-negative weights so they sum to 1.0."""

    if not weights:
        return {}

    for key, value in weights.items():
        if value < 0:
            raise ValueError(f"Weight cannot be negative: {key}={value}")

    total = sum(weights.values())

    if total <= EPSILON:
        raise ValueError("The sum of weights must be positive and non-zero.")

    return {key: value / total for key, value in weights.items()}


def largest_remainder_allocation(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    """Allocate an integer total using the largest remainder method.

    This method preserves the requested total exactly while approximating
    the target proportions defined by the input weights.
    """

    if total < 0:
        raise ValueError("Total cannot be negative.")

    if not weights:
        return {}

    normalized = normalize_weights(weights)

    raw = {key: total * weight for key, weight in normalized.items()}
    base = {key: math.floor(value) for key, value in raw.items()}

    allocated = sum(base.values())
    residual = total - allocated

    if residual == 0:
        return base

    remainders = sorted(
        ((key, raw[key] - base[key]) for key in raw),
        key=lambda pair: pair[1],
        reverse=True,
    )

    for index in range(residual):
        key = remainders[index % len(remainders)][0]
        base[key] += 1

    return base