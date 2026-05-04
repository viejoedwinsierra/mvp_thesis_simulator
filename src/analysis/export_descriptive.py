from pathlib import Path

import pandas as pd
import numpy as np


EXCLUDED_NUMERIC_COLUMNS = {
    "content_hash",
}


class DescriptiveExporter:

    def __init__(self, df: pd.DataFrame, output_dir="output/descriptive_exports"):
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.numeric_df = (
            self.df
            .select_dtypes(include=np.number)
            .drop(columns=[c for c in EXCLUDED_NUMERIC_COLUMNS if c in self.df.columns])
        )

        self.categorical_df = self.df.select_dtypes(exclude=np.number)

    # ======================================================
    # 1. ESTADÍSTICAS POR VARIABLE
    # ======================================================
    def export_summary_stats(self):
        summary = []

        for col in self.numeric_df.columns:
            data = self.numeric_df[col].replace([np.inf, -np.inf], np.nan).dropna()

            if data.empty:
                continue

            summary.append({
                "variable": col,
                "mean": data.mean(),
                "median": data.median(),
                "std": data.std(),
                "min": data.min(),
                "max": data.max(),
                "skewness": data.skew(),
                "kurtosis": data.kurtosis(),
            })

        pd.DataFrame(summary).to_csv(
            self.output_dir / "summary_stats.csv",
            index=False,
            encoding="utf-8",
        )

        print("✔ summary_stats.csv generado")

    # ======================================================
    # 2. HISTOGRAMAS (BINS)
    # ======================================================
    def export_histograms(self):
        for col in self.numeric_df.columns:
            data = self.numeric_df[col].replace([np.inf, -np.inf], np.nan).dropna()

            if data.empty:
                continue

            counts, bins = np.histogram(data, bins=50)

            hist_df = pd.DataFrame({
                "bin_start": bins[:-1],
                "bin_end": bins[1:],
                "count": counts,
                "percentage": counts / len(data) * 100,
            })

            hist_df.to_csv(
                self.output_dir / f"{col}_histogram.csv",
                index=False,
                encoding="utf-8",
            )

        print("✔ histogramas exportados")

    # ======================================================
    # 3. PERCENTILES
    # ======================================================
    def export_percentiles(self):
        rows = []
        percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]

        for col in self.numeric_df.columns:
            data = self.numeric_df[col].replace([np.inf, -np.inf], np.nan).dropna()

            if data.empty:
                continue

            for p in percentiles:
                rows.append({
                    "variable": col,
                    "percentile": p,
                    "value": np.percentile(data, p),
                })

        pd.DataFrame(rows).to_csv(
            self.output_dir / "percentiles.csv",
            index=False,
            encoding="utf-8",
        )

        print("✔ percentiles.csv generado")

    # ======================================================
    # 4. FRECUENCIAS CATEGÓRICAS
    # ======================================================
    def export_categorical_frequencies(self):
        for col in self.categorical_df.columns:
            data = self.categorical_df[col].dropna()

            if data.empty:
                continue

            freq = (
                data
                .value_counts(normalize=True)
                .reset_index()
            )

            freq.columns = ["value", "percentage"]
            freq["percentage"] *= 100

            freq.to_csv(
                self.output_dir / f"{col}_frequency.csv",
                index=False,
                encoding="utf-8",
            )

        print("✔ frecuencias categóricas exportadas")

    # ======================================================
    # 5. DISTRIBUCIÓN NORMALIZADA (KDE APROX)
    # ======================================================
    def export_density(self):
        for col in self.numeric_df.columns:
            data = self.numeric_df[col].replace([np.inf, -np.inf], np.nan).dropna()

            if data.empty:
                continue

            counts, bins = np.histogram(data, bins=100, density=True)

            density_df = pd.DataFrame({
                "x": bins[:-1],
                "density": counts,
            })

            density_df.to_csv(
                self.output_dir / f"{col}_density.csv",
                index=False,
                encoding="utf-8",
            )

        print("✔ densidades exportadas")

    # ======================================================
    # EJECUCIÓN TOTAL
    # ======================================================
    def run_all(self):
        self.export_summary_stats()
        self.export_histograms()
        self.export_percentiles()
        self.export_categorical_frequencies()
        self.export_density()

        print("\n✅ Exportación descriptiva completa")