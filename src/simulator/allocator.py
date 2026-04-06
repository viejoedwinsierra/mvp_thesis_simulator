from __future__ import annotations

from typing import Dict
import math


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize positive weights to sum to 1.0.

    This function is intentionally isolated because it represents one of the
    core probabilistic operations of the MVP: converting user-provided weights
    into a valid discrete distribution.
    """
    if not weights:
        return {}

    total = sum(weights.values())
    if total <= 0:
        raise ValueError("The sum of weights must be positive.")

    return {key: value / total for key, value in weights.items()}



def largest_remainder_allocation(total: int, weights: Dict[str, float]) -> Dict[str, int]:
    """Allocate an integer total using the largest remainder method.

    Why this method matters in the thesis MVP:
    - probability distributions produce expected fractional values;
    - the simulator needs integer counts of files/cases;
    - naive rounding can break invariants and produce totals that do not match
      the global error universe;
    - the largest remainder method preserves the exact total while staying
      close to the theoretical distribution.
    """
    if total < 0:
        raise ValueError("Total cannot be negative.")

    if not weights:
        return {}

    normalized = normalize_weights(weights)
    raw = {k: total * w for k, w in normalized.items()}
    base = {k: math.floor(v) for k, v in raw.items()}
    residual = total - sum(base.values())

    remainders = sorted(
        ((k, raw[k] - base[k]) for k in raw),
        key=lambda pair: pair[1],
        reverse=True,
    )

    for i in range(residual):
        base[remainders[i][0]] += 1

    return base
