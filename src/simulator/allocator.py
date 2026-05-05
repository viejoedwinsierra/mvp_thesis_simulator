from __future__ import annotations

import math
from typing import Mapping


EPSILON = 1e-12


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Normalize non-negative finite weights so they sum to 1.0."""

    if not weights:
        return {}

    for key, value in weights.items():
        if not math.isfinite(value):
            raise ValueError(f"Weight must be finite: {key}={value}")

        if value < 0:
            raise ValueError(f"Weight cannot be negative: {key}={value}")

    total = sum(weights.values())

    if total <= EPSILON:
        raise ValueError("The sum of weights must be positive and non-zero.")

    return {key: value / total for key, value in weights.items()}


def largest_remainder_allocation(
    total: int,
    weights: Mapping[str, float],
) -> dict[str, int]:
    """Allocate an integer total using the largest remainder method.

    Use this when the total must be preserved exactly while approximating
    proportional weights. For Monte Carlo arrival counts, prefer stochastic
    sampling such as Poisson instead.
    """

    if not isinstance(total, int):
        raise TypeError("Total must be an integer.")

    if total < 0:
        raise ValueError("Total cannot be negative.")

    if not weights:
        return {}

    normalized = normalize_weights(weights)

    raw = {
        key: total * weight
        for key, weight in normalized.items()
    }

    base = {
        key: math.floor(value)
        for key, value in raw.items()
    }

    residual = total - sum(base.values())

    if residual == 0:
        return base

    if residual > len(base):
        raise RuntimeError(
            "Invalid largest remainder state: residual exceeds number of groups."
        )

    remainders = sorted(
        (
            (key, raw[key] - base[key])
            for key in raw
        ),
        key=lambda pair: (-pair[1], pair[0]),
    )

    for key, _ in remainders[:residual]:
        base[key] += 1

    return base