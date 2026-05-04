"""MVP thesis simulator package.

This package generates a synthetic file-level dataset for
probability, statistics, cost modeling, and storage-behavior simulation.

The simulator does not generate physical files. It only produces
structured metadata datasets, such as CSV or Parquet files, which act
as the source of truth for downstream analysis and modeling.
"""

__version__ = "0.1.0"

__all__ = [
    "config_models",
    "allocator",
    "case_definitions",
    "content_factory",
    "orchestrator",
]