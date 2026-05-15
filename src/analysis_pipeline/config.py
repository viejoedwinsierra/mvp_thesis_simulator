from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AnalysisConfig:
    dataset_dir: Path = Path("output/dataset")
    output_dir: Path = Path("output")
    descriptive_plot_dir: Path = Path("output/plots/descriptive")
    advanced_plot_dir: Path = Path("output/plots/advanced")
    html_dir: Path = Path("output/html")
    report_name: str = "report_descriptive.html"
    advanced_report_name: str = "report_advanced.html"

    # Nuevo esquema recomendado para reportes por escenario.
    scenario_analysis_dir: Path = Path("output/analysis")

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.descriptive_plot_dir.mkdir(parents=True, exist_ok=True)
        self.advanced_plot_dir.mkdir(parents=True, exist_ok=True)
        self.html_dir.mkdir(parents=True, exist_ok=True)
        self.scenario_analysis_dir.mkdir(parents=True, exist_ok=True)
