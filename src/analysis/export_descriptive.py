import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class DescriptiveExporter:

    def __init__(
        self,
        df: pd.DataFrame,
        output_dir="output/descriptive_exports",
        plots_dir="output/plots/descriptive"
    ):
        self.df = df.copy()
        self.output_dir = output_dir
        self.plots_dir = plots_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

        self.numeric_df = df.select_dtypes(include=np.number)
        self.categorical_df = df.select_dtypes(exclude=np.number)

    def export_summary_stats(self):
        summary = []

        for col in self.numeric_df.columns:
            data = self.numeric_df[col].dropna()

            summary.append({
                "variable": col,
                "count": data.count(),
                "mean": data.mean(),
                "median": data.median(),
                "std": data.std(),
                "min": data.min(),
                "q1": data.quantile(0.25),
                "q3": data.quantile(0.75),
                "max": data.max(),
                "skewness": data.skew(),
                "kurtosis": data.kurtosis()
            })

        pd.DataFrame(summary).to_csv(
            f"{self.output_dir}/summary_stats.csv",
            index=False
        )

    def export_histograms(self):
        for col in self.numeric_df.columns:
            data = self.numeric_df[col].dropna()

            counts, bins = np.histogram(data, bins=50)

            hist_df = pd.DataFrame({
                "variable": col,
                "bin_start": bins[:-1],
                "bin_end": bins[1:],
                "count": counts,
                "percentage": counts / len(data) * 100
            })

            hist_df.to_csv(
                f"{self.output_dir}/{col}_histogram.csv",
                index=False
            )

    def export_percentiles(self):
        percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
        rows = []

        for col in self.numeric_df.columns:
            data = self.numeric_df[col].dropna()

            for p in percentiles:
                rows.append({
                    "variable": col,
                    "percentile": p,
                    "value": np.percentile(data, p)
                })

        pd.DataFrame(rows).to_csv(
            f"{self.output_dir}/percentiles.csv",
            index=False
        )

    def export_categorical_frequencies(self):
        for col in self.categorical_df.columns:
            freq = self.df[col].value_counts(normalize=True).reset_index()
            freq.columns = ["value", "percentage"]
            freq["percentage"] *= 100

            freq.to_csv(
                f"{self.output_dir}/{col}_frequency.csv",
                index=False
            )

    def export_density(self):
        for col in self.numeric_df.columns:
            data = self.numeric_df[col].dropna()

            counts, bins = np.histogram(data, bins=100, density=True)

            density_df = pd.DataFrame({
                "variable": col,
                "x": bins[:-1],
                "density": counts
            })

            density_df.to_csv(
                f"{self.output_dir}/{col}_density.csv",
                index=False
            )

    # ======================================================
    # IMÁGENES: DISTRIBUCIÓN POR VARIABLE
    # ======================================================
    def export_distribution_plots(self):
        for col in self.numeric_df.columns:
            data = self.numeric_df[col].dropna()

            plt.figure(figsize=(9, 5))
            sns.histplot(data, bins=50, stat="percent", kde=True)
            plt.title(f"Distribución porcentual y densidad: {col}")
            plt.xlabel(col)
            plt.ylabel("Porcentaje")
            plt.tight_layout()
            plt.savefig(
                f"{self.plots_dir}/{col}_distribution.png",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()

    # ======================================================
    # IMÁGENES: BOXPLOT POR VARIABLE
    # ======================================================
    def export_boxplots(self):
        for col in self.numeric_df.columns:
            data = self.numeric_df[col].dropna()

            plt.figure(figsize=(9, 4))
            sns.boxplot(x=data)
            plt.title(f"Boxplot: {col}")
            plt.xlabel(col)
            plt.tight_layout()
            plt.savefig(
                f"{self.plots_dir}/{col}_boxplot.png",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()

    # ======================================================
    # IMÁGENES: CATEGÓRICAS
    # ======================================================
    def export_categorical_plots(self):
        for col in self.categorical_df.columns:
            freq = self.df[col].value_counts(normalize=True).head(15) * 100

            plt.figure(figsize=(10, 5))
            freq.plot(kind="bar")
            plt.title(f"Frecuencia porcentual: {col}")
            plt.ylabel("Porcentaje")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(
                f"{self.plots_dir}/{col}_frequency.png",
                dpi=300,
                bbox_inches="tight"
            )
            plt.close()

    # ======================================================
    # IMÁGENES: GRILLA RESUMEN
    # ======================================================
    def export_numeric_grid(self):
        cols = list(self.numeric_df.columns)

        n_cols = 3
        n_rows = math.ceil(len(cols) / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            data = self.numeric_df[col].dropna()

            sns.histplot(
                data,
                bins=50,
                stat="percent",
                kde=True,
                ax=axes[i]
            )

            axes[i].set_title(col)
            axes[i].set_ylabel("Porcentaje")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.savefig(
            f"{self.plots_dir}/all_numeric_distributions.png",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    def run_all(self):
        self.export_summary_stats()
        self.export_histograms()
        self.export_percentiles()
        self.export_categorical_frequencies()
        self.export_density()

        self.export_distribution_plots()
        self.export_boxplots()
        self.export_categorical_plots()
        self.export_numeric_grid()

        print("\n✅ Exportación descriptiva completa: CSV + imágenes")