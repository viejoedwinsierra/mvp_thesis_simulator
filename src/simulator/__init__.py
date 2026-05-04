"""MVP thesis simulator package.

This package generates a synthetic dataset at file level for
probability/statistics modeling.

NOTE:
This module DOES NOT generate physical files.
It only produces a structured dataset (e.g., CSV/Parquet),
which acts as the source of truth for downstream processes.
"""

__all__ = [
    "config_models",
    "allocator",
    "case_definitions",
    "content_factory",
    "orchestrator",
]