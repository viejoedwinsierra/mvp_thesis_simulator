from pathlib import Path

import pandas as pd

from analysis_pipeline.advanced_analysis import AdvancedAnalysis


ALLOWED_ADVANCED_PLOT_PREFIXES = (
    "correlation_matrix_",
    "scatter_",
    "log_scatter_",
    "target_relationships_",
)


def _is_advanced_multivariate_plot(path: Path) -> bool:
    return path.suffix.lower() == ".png" and path.name.startswith(
        ALLOWED_ADVANCED_PLOT_PREFIXES
    )


def run_advanced_plots(
    df: pd.DataFrame,
    output_dir: str | Path = "output/plots/advanced",
    enabled: bool = False,
) -> list[Path]:
    """Genera únicamente gráficos avanzados multivariados.

    No genera ni retorna gráficos univariados:
    - histogramas
    - boxplots simples
    - distribuciones
    - barras de frecuencia

    Permitidos:
    - heatmaps de correlación
    - scatter plots de relaciones principales
    - scatter plots logarítmicos
    - gráficos de relación con targets
    """
    if not enabled:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analyzer = AdvancedAnalysis(df, output_dir=output_dir)
    analyzer.run_multivariate()

    plot_files = sorted(output_dir.glob("*.png"))

    return [
        path
        for path in plot_files
        if _is_advanced_multivariate_plot(path)
    ]