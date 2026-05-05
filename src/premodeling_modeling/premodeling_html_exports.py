from pathlib import Path

import pandas as pd

from premodeling_modeling.premodeling_exporter import PremodelingExporter


def build_premodeling_report(
    df: pd.DataFrame,
    plot_files: list[Path] | None = None,
    output_path: str | Path = "output/html/report_premodeling.html",
    evidence_dir: str | Path = "output/premodeling",
    enabled: bool = False,
) -> Path | None:
    if not enabled:
        return None

    exporter = PremodelingExporter(
        df=df,
        plot_files=plot_files or [],
        output_path=output_path,
        evidence_dir=evidence_dir,
        export_evidence=True,
    )

    return exporter.build_report()
