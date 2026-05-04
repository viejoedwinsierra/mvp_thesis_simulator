import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm


class AdvancedAnalysis:

    def __init__(self, df: pd.DataFrame, output_dir="output/plots/advanced"):
        self.df = df.copy()
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.exclude_columns = [
            "file_id",
            "hash_head",
            "hash_tail",
            "sequence",
        ]

        self.numeric_df = (
            self.df
            .select_dtypes(include=np.number)
            .drop(columns=self.exclude_columns, errors="ignore")
        )

        self.categorical_df = (
            self.df
            .select_dtypes(exclude=np.number)
            .drop(columns=self.exclude_columns, errors="ignore")
        )

    def _save_plot(self, filename):
        plt.tight_layout()
        plt.savefig(
            f"{self.output_dir}/{filename}",
            dpi=300,
            bbox_inches="tight"
        )
        plt.close()

    # ======================================================
    # 1. DISTRIBUCIONES PORCENTUALES + KDE
    # ======================================================
    def plot_probability_distributions(self):
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

            axes[i].set_title(f"Distribución porcentual + KDE: {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Porcentaje")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        self._save_plot("probability_distributions_grid.png")

    # ======================================================
    # 2. DISTRIBUCIONES INDIVIDUALES
    # ======================================================
    def plot_individual_distributions(self):
        for col in self.numeric_df.columns:
            data = self.numeric_df[col].dropna()

            plt.figure(figsize=(9, 5))
            sns.histplot(data, bins=50, stat="percent", kde=True)
            plt.title(f"Distribución porcentual y KDE: {col}")
            plt.xlabel(col)
            plt.ylabel("Porcentaje")
            self._save_plot(f"{col}_distribution.png")

    # ======================================================
    # 3. DISTRIBUCIONES LOG PARA VARIABLES SESGADAS
    # ======================================================
    def plot_log_distributions(self):
        selected_cols = [
            "size_mb",
            "storage_cost",
            "transfer_speed_mbps",
            "transfer_duration_sec",
        ]

        selected_cols = [c for c in selected_cols if c in self.numeric_df.columns]

        for col in selected_cols:
            data = self.numeric_df[col].dropna()
            data = data[data > 0]

            if data.empty:
                continue

            plt.figure(figsize=(9, 5))
            sns.histplot(data, bins=50, stat="percent", kde=True)
            plt.xscale("log")
            plt.title(f"Distribución logarítmica: {col}")
            plt.xlabel(col)
            plt.ylabel("Porcentaje")
            self._save_plot(f"{col}_log_distribution.png")

    # ======================================================
    # 4. BOXPLOTS
    # ======================================================
    def plot_boxplots(self):
        cols = list(self.numeric_df.columns)
        n_cols = 3
        n_rows = math.ceil(len(cols) / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            data = self.numeric_df[col].dropna()
            sns.boxplot(x=data, ax=axes[i])
            axes[i].set_title(f"Boxplot: {col}")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        self._save_plot("boxplots_grid.png")

    # ======================================================
    # 5. VARIABLES CATEGÓRICAS ÚTILES
    # ======================================================
    def plot_categorical(self):
        useful_cols = [
            "case_type",
            "case_group",
            "file_type",
            "storage_tier",
            "day_of_week",
            "time_slot",
        ]

        useful_cols = [c for c in useful_cols if c in self.df.columns]

        n_cols = 2
        n_rows = math.ceil(len(useful_cols) / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(useful_cols):
            freq = self.df[col].value_counts(normalize=True).head(15) * 100
            freq.plot(kind="bar", ax=axes[i])

            axes[i].set_title(f"Frecuencia porcentual: {col}")
            axes[i].set_ylabel("Porcentaje")
            axes[i].tick_params(axis="x", rotation=45)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        self._save_plot("categorical_frequencies.png")

    # ======================================================
    # 6. MATRIZ DE CORRELACIÓN
    # ======================================================
    def correlation_matrix(self):
        corr = self.numeric_df.corr()

        plt.figure(figsize=(13, 10))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5
        )

        plt.title("Matriz de correlación")
        self._save_plot("correlation_matrix.png")

        corr.to_csv("output/correlation_matrix.csv")
        return corr

    # ======================================================
    # 7. MULTICOLINEALIDAD VIF
    # ======================================================
    def calculate_vif(self):
        X = self.numeric_df.copy()

        # Para VIF no se debe incluir la variable objetivo ni variables derivadas redundantes
        X = X.drop(
            columns=[
                "storage_cost",
                "has_error",
                "is_duplicate",
                "severity",
            ],
            errors="ignore"
        )

        X = X.dropna()
        X = X.loc[:, X.std() > 0]

        X = sm.add_constant(X)

        vif_data = pd.DataFrame()
        vif_data["variable"] = X.columns
        vif_data["VIF"] = [
            variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])
        ]

        vif_data = vif_data.sort_values("VIF", ascending=False)
        vif_data.to_csv("output/vif_analysis_clean.csv", index=False)

        print("\n=== VIF LIMPIO PARA MODELADO ===")
        print(vif_data)

        return vif_data

    # ======================================================
    # 8. RELACIONES CON COSTO
    # ======================================================
    def scatter_cost_relationships(self):
        if "storage_cost" not in self.df.columns:
            return

        cols = [
            "size_mb",
            "days_stored",
            "days_since_last_access",
            "transfer_duration_sec",
            "transfer_speed_mbps",
            "movement_storage",
        ]

        cols = [c for c in cols if c in self.df.columns]

        n_cols = 2
        n_rows = math.ceil(len(cols) / n_cols)

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
        axes = axes.flatten()

        for i, col in enumerate(cols):
            sns.scatterplot(
                data=self.df,
                x=col,
                y="storage_cost",
                alpha=0.35,
                ax=axes[i]
            )

            axes[i].set_title(f"{col} vs storage_cost")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("storage_cost")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        self._save_plot("cost_relationships.png")

    # ======================================================
    # 9. ANÁLISIS TEMPORAL
    # ======================================================
    def plot_time_analysis(self):
        if "created_at" not in self.df.columns:
            return

        self.df["created_at"] = pd.to_datetime(
            self.df["created_at"],
            errors="coerce"
        )

        self.df["hour"] = self.df["created_at"].dt.hour

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        self.df.groupby("hour").size().plot(kind="bar", ax=axes[0, 0])
        axes[0, 0].set_title("Cantidad de archivos por hora")
        axes[0, 0].set_ylabel("Cantidad")

        self.df.groupby("hour")["size_mb"].mean().plot(ax=axes[0, 1])
        axes[0, 1].set_title("Tamaño promedio por hora")
        axes[0, 1].set_ylabel("size_mb promedio")

        self.df.groupby("hour")["storage_cost"].sum().plot(ax=axes[1, 0])
        axes[1, 0].set_title("Costo total por hora")
        axes[1, 0].set_ylabel("storage_cost total")

        sns.boxplot(
            data=self.df,
            x="hour",
            y="size_mb",
            ax=axes[1, 1]
        )
        axes[1, 1].set_title("Distribución de size_mb por hora")

        self._save_plot("time_analysis.png")

    # ======================================================
    # 10. EJECUCIÓN COMPLETA
    # ======================================================
    def run_all(self):
        self.plot_probability_distributions()
        self.plot_individual_distributions()
        self.plot_log_distributions()
        self.plot_boxplots()
        self.plot_categorical()
        self.correlation_matrix()
        self.calculate_vif()
        self.scatter_cost_relationships()
        self.plot_time_analysis()

        print("\n✅ Análisis avanzado completo: imágenes + correlación + VIF")