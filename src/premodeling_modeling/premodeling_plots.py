from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from premodeling_modeling.premodeling_tables import build_premodeling_datasets


def _save_plot(path: Path) -> Path:
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return path


def plot_premodeling_target_distributions(
    df: pd.DataFrame,
    output_dir: str | Path = "output/plots/premodeling",
) -> list[Path]:
    """Genera gráficos ligeros de los targets preparados.

    No evalúa modelos. Sirve para evidencia de entrada a modelamiento.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = build_premodeling_datasets(df)
    paths = []

    for target, payload in datasets.items():
        y = payload["y"]

        plt.figure(figsize=(9, 5))

        if y.nunique(dropna=True) <= 2:
            y.value_counts(normalize=True).sort_index().plot(kind="bar")
            plt.ylabel("Proporción")
        else:
            y.hist(bins=50)
            plt.ylabel("Frecuencia")

        plt.title(f"Distribución preparada del target: {target}")
        plt.xlabel(payload["metadata"].get("target_output_name", target))

        paths.append(_save_plot(output_dir / f"premodeling_target_{target}.png"))

    return paths


def run_premodeling_plots(
    df: pd.DataFrame,
    output_dir: str | Path = "output/plots/premodeling",
    enabled: bool = False,
) -> list[Path]:
    if not enabled:
        return []

    return plot_premodeling_target_distributions(df, output_dir=output_dir)
