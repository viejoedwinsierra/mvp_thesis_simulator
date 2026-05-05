from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd

from premodeling_modeling.modeling_exporter import ModelingExporter


def build_modeling_report(
    df: pd.DataFrame,
    plot_files: list[Path] | None = None,
    output_path: str | Path = "output/html/report_modeling.html",
    enabled: bool = False,
    model_results: Dict[str, Dict[str, Any]] | None = None,
) -> Path | None:
    """Construye el reporte HTML de modelamiento estadístico.

    Mantiene compatibilidad:
    - conserva la función pública `build_modeling_report`;
    - respeta `enabled=False`;
    - mantiene los parámetros existentes;
    - agrega `model_results` como opcional para evitar reentrenamiento.
    """

    if not enabled:
        return None

    exporter_kwargs: dict[str, Any] = {
        "df": df,
        "plot_files": plot_files or [],
        "output_path": output_path,
    }

    # Compatible con el ModelingExporter mejorado.
    # Si por alguna razón el exporter local aún no soporta model_results,
    # se hace fallback sin romper ejecución.
    if model_results is not None:
        try:
            exporter = ModelingExporter(
                **exporter_kwargs,
                model_results=model_results,
            )
        except TypeError:
            exporter = ModelingExporter(**exporter_kwargs)
    else:
        exporter = ModelingExporter(**exporter_kwargs)

    return exporter.build_report()
