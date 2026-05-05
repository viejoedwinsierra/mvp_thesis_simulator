from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from analysis_pipeline.advanced_tables import (
    IDENTIFIER_COLUMNS,
    LEAKAGE_COLUMNS,
    TARGET_COLUMNS,
    _numeric_multivariate_frame,
    build_correlation_matrix,
    build_nonlinearity_table,
    build_target_relationships,
    build_top_relationships,
)


class AdvancedAnalysis:
    """Análisis multivariado exploratorio previo al modelado.

    Alcance:
    - Matrices de correlación Pearson y Spearman.
    - Top relaciones multivariadas.
    - Relación exploratoria con targets.
    - Posibles relaciones monotónicas no lineales.
    - Scatters logarítmicos cuando aplica.

    Restricciones:
    - No calcula modelos predictivos.
    - No calcula regresiones.
    - No calcula VIF.
    - No hace inferencia causal.
    - No genera gráficos univariados.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        output_dir: str | Path = "output/plots/advanced",
    ):
        self.df = df.copy()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.exclude_columns = set(IDENTIFIER_COLUMNS) | set(LEAKAGE_COLUMNS)

        self.numeric_df = _numeric_multivariate_frame(
            self.df,
            include_targets=True,
            include_derived=True,
        )

    # ======================================================
    # UTILIDADES
    # ======================================================

    def _save_plot(self, filename: str) -> Path:
        path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    @staticmethod
    def _safe_filename(value: str) -> str:
        return (
            str(value)
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "_")
            .lower()
        )

    @staticmethod
    def _is_numeric_pair(df: pd.DataFrame, var1: str, var2: str) -> bool:
        if var1 not in df.columns or var2 not in df.columns:
            return False

        return (
            pd.api.types.is_numeric_dtype(df[var1])
            and pd.api.types.is_numeric_dtype(df[var2])
        )

    # ======================================================
    # 1. HEATMAP CORRELACIÓN
    # ======================================================

    def plot_correlation_heatmap(self, method: str = "spearman") -> Path | None:
        """Genera heatmap de correlación.

        Spearman debe interpretarse como matriz principal.
        Pearson se conserva como referencia lineal.
        """
        corr = build_correlation_matrix(self.df, method=method)

        if corr.empty:
            return None

        plt.figure(figsize=(13, 10))
        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
        )

        plt.title(f"Matriz de correlación {method.title()}")

        return self._save_plot(f"correlation_matrix_{method}.png")

    # ======================================================
    # 2. SCATTERS TOP RELACIONES
    # ======================================================

    def plot_top_relationship_scatters(
        self,
        threshold: float = 0.5,
        max_plots: int = 8,
        exclude_target_target_pairs: bool = True,
    ) -> list[Path]:
        """Genera scatters para las principales relaciones Spearman.

        Evita generar gráficos para pares que no sean numéricos.
        Opcionalmente evita pares target-target para no contaminar lectura.
        """
        top_pairs = build_top_relationships(
            self.df,
            method="spearman",
            threshold=threshold,
            include_targets=True,
            include_derived=True,
        )

        paths: list[Path] = []

        if top_pairs.empty:
            return paths

        selected = 0

        for _, row in top_pairs.iterrows():
            if selected >= max_plots:
                break

            var1 = row["variable_1"]
            var2 = row["variable_2"]

            if exclude_target_target_pairs and var1 in TARGET_COLUMNS and var2 in TARGET_COLUMNS:
                continue

            if not self._is_numeric_pair(self.df, var1, var2):
                continue

            data = (
                self.df[[var1, var2]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            if data.empty:
                continue

            plt.figure(figsize=(9, 5))
            sns.scatterplot(
                data=data,
                x=var1,
                y=var2,
                alpha=0.35,
            )

            plt.title(f"{var1} vs {var2} | Spearman={row['correlation']:.2f}")
            plt.xlabel(var1)
            plt.ylabel(var2)

            filename = (
                f"scatter_{self._safe_filename(var1)}"
                f"_vs_{self._safe_filename(var2)}.png"
            )

            paths.append(self._save_plot(filename))
            selected += 1

        return paths

    # ======================================================
    # 3. RELACIONES CON TARGETS
    # ======================================================

    def plot_target_relationships(
        self,
        max_variables_per_target: int = 5,
        exclude_other_targets: bool = True,
    ) -> list[Path]:
        """Genera gráficos de variables asociadas a cada target.

        Mejora frente a versión anterior:
        - Evita graficar un target contra otro target, salvo que se indique lo contrario.
        - Solo usa variables numéricas válidas para scatter.
        """
        target_relationships = build_target_relationships(self.df)

        paths: list[Path] = []

        if target_relationships.empty:
            return paths

        for target in sorted(TARGET_COLUMNS):
            if target not in self.df.columns:
                continue

            if not pd.api.types.is_numeric_dtype(self.df[target]):
                continue

            subset = target_relationships[
                target_relationships["target"] == target
            ].copy()

            if exclude_other_targets:
                subset = subset[~subset["variable"].isin(TARGET_COLUMNS)]

            variables: list[str] = []

            for variable in subset["variable"]:
                if len(variables) >= max_variables_per_target:
                    break

                if self._is_numeric_pair(self.df, variable, target):
                    variables.append(variable)

            if not variables:
                continue

            n_cols = 2
            n_rows = math.ceil(len(variables) / n_cols)

            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(14, 5 * n_rows),
            )

            axes = np.array(axes).reshape(-1)

            for i, variable in enumerate(variables):
                data = (
                    self.df[[variable, target]]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )

                sns.scatterplot(
                    data=data,
                    x=variable,
                    y=target,
                    alpha=0.35,
                    ax=axes[i],
                )

                axes[i].set_title(f"{variable} vs {target}")
                axes[i].set_xlabel(variable)
                axes[i].set_ylabel(target)

            for j in range(len(variables), len(axes)):
                axes[j].axis("off")

            filename = f"target_relationships_{self._safe_filename(target)}.png"
            paths.append(self._save_plot(filename))

        return paths

    # ======================================================
    # 4. SCATTERS LOG DESDE NO LINEALIDAD
    # ======================================================

    def plot_log_scatter_candidates(
        self,
        candidate_pairs: list[tuple[str, str]] | None = None,
        max_plots: int = 6,
    ) -> list[Path]:
        """Genera scatters log1p para pares candidatos.

        Si no se pasan pares manuales, usa la tabla de no linealidad
        Pearson vs Spearman para seleccionar relaciones candidatas.
        """
        if candidate_pairs is None:
            nonlinear = build_nonlinearity_table(self.df)

            if nonlinear.empty:
                candidate_pairs = [
                    ("size_mb", "transfer_duration_sec"),
                    ("size_mb", "storage_cost"),
                    ("transfer_duration_sec", "storage_cost"),
                    ("transfer_speed_mbps", "transfer_duration_sec"),
                ]
            else:
                candidate_pairs = list(
                    nonlinear[["variable_1", "variable_2"]]
                    .head(max_plots)
                    .itertuples(index=False, name=None)
                )

        paths: list[Path] = []
        selected = 0

        for var1, var2 in candidate_pairs:
            if selected >= max_plots:
                break

            if not self._is_numeric_pair(self.df, var1, var2):
                continue

            data = (
                self.df[[var1, var2]]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )

            data = data[(data[var1] > 0) & (data[var2] > 0)]

            if data.empty:
                continue

            plt.figure(figsize=(9, 5))
            sns.scatterplot(
                x=np.log1p(data[var1]),
                y=np.log1p(data[var2]),
                alpha=0.35,
            )

            plt.title(f"Relación log1p: {var1} vs {var2}")
            plt.xlabel(f"log1p({var1})")
            plt.ylabel(f"log1p({var2})")

            filename = (
                f"log_scatter_{self._safe_filename(var1)}"
                f"_vs_{self._safe_filename(var2)}.png"
            )

            paths.append(self._save_plot(filename))
            selected += 1

        return paths

    # ======================================================
    # 5. BOXPLOTS TARGET VS CATEGÓRICAS
    # ======================================================

    def plot_categorical_vs_targets(
        self,
        categorical_columns: list[str] | None = None,
        max_categories: int = 12,
    ) -> list[Path]:
        """Gráficos categóricos vs targets.

        Importante:
        Este método se conserva por compatibilidad, pero NO se ejecuta dentro
        de run_multivariate(), porque pertenece mejor a un reporte visual
        exploratorio separado y puede contaminar el reporte multivariado.
        """
        categorical_columns = categorical_columns or [
            "file_type",
            "storage_tier",
            "size_range",
            "time_slot",
            "day_of_week",
        ]

        numeric_targets = [
            target
            for target in ["transfer_duration_sec", "storage_cost"]
            if target in self.df.columns and pd.api.types.is_numeric_dtype(self.df[target])
        ]

        paths: list[Path] = []

        for cat in categorical_columns:
            if cat not in self.df.columns:
                continue

            top_categories = self.df[cat].value_counts().head(max_categories).index
            data = self.df[self.df[cat].isin(top_categories)].copy()

            for target in numeric_targets:
                plt.figure(figsize=(11, 5))
                sns.boxplot(
                    data=data,
                    x=cat,
                    y=target,
                )

                plt.title(f"{target} por {cat}")
                plt.xlabel(cat)
                plt.ylabel(target)
                plt.xticks(rotation=45)

                filename = (
                    f"boxplot_{self._safe_filename(target)}"
                    f"_by_{self._safe_filename(cat)}.png"
                )

                paths.append(self._save_plot(filename))

        return paths

    # ======================================================
    # 6. EJECUCIÓN MULTIVARIADA
    # ======================================================

    def run_multivariate(self) -> list[Path]:
        """Ejecuta solamente gráficos del reporte avanzado multivariado.

        No incluye:
        - histogramas
        - distribuciones
        - boxplots simples
        - boxplots categóricos
        - gráficos univariados
        """
        generated: list[Path] = []

        for method in ["spearman", "pearson"]:
            path = self.plot_correlation_heatmap(method=method)

            if path is not None:
                generated.append(path)

        generated.extend(self.plot_top_relationship_scatters())
        generated.extend(self.plot_target_relationships())
        generated.extend(self.plot_log_scatter_candidates())

        return generated
