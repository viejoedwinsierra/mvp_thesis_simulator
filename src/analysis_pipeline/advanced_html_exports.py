from pathlib import Path

import pandas as pd

from analysis_pipeline.advanced_exporter import AdvancedExporter


def build_advanced_report(
    df: pd.DataFrame,
    plot_files: list[Path],
    output_path: str | Path = "output/html/report_advanced.html",
    enabled: bool = False,
) -> Path | None:
    """Construye el reporte avanzado multivariado.

    Conserva la interfaz existente para no romper el pipeline.
    """
    if not enabled:
        return None

    exporter = AdvancedExporter(
        df=df,
        plot_files=plot_files,
        output_path=output_path,
    )

    return exporter.build_report()
