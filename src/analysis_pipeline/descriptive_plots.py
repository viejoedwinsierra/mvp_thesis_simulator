from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def save_plot(fig, output_dir: str | Path, output_file: str):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / output_file

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return output_file


def _get_numeric_series(df: pd.DataFrame, column: str):
    if column not in df.columns:
        return None

    series = pd.to_numeric(df[column], errors="coerce").dropna()

    if series.empty:
        return None

    return series


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    output_dir,
    output_file: str,
    bins: int = 50,
):
    """
    Histograma univariado en escala original.
    Si la variable tiene alta asimetría, usa escala logarítmica en el eje Y
    para visualizar mejor la cola.
    """
    series = _get_numeric_series(df, column)

    if series is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(series, bins=bins)

    if abs(series.skew()) >= 1:
        ax.set_yscale("log")
        ylabel = "Frecuencia log"
    else:
        ylabel = "Frecuencia"

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    return save_plot(fig, output_dir, output_file)


def plot_log_histogram(
    df: pd.DataFrame,
    column: str,
    output_dir,
    output_file: str,
    bins: int = 50,
):
    """
    Histograma univariado de log(variable).

    Solo aplica para variables estrictamente positivas.
    Útil para size_mb, transfer_duration_sec, storage_cost
    y otras variables con colas largas.
    """
    series = _get_numeric_series(df, column)

    if series is None:
        return None

    series = series[series > 0]

    if series.empty:
        return None

    log_series = np.log1p(series)

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(log_series, bins=bins)

    ax.set_title(f"Distribución log1p({column})")
    ax.set_xlabel(f"log1p({column})")
    ax.set_ylabel("Frecuencia")
    ax.grid(True, alpha=0.3)

    return save_plot(fig, output_dir, output_file)


def plot_boxplot(
    df: pd.DataFrame,
    column: str,
    title: str,
    output_dir,
    output_file: str,
):
    """
    Boxplot univariado en escala original.
    """
    series = _get_numeric_series(df, column)

    if series is None:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.boxplot(series, vert=False, showmeans=True)

    ax.set_title(title)
    ax.set_xlabel(column)
    ax.grid(True, alpha=0.3)

    return save_plot(fig, output_dir, output_file)


def plot_trimmed_boxplot(
    df: pd.DataFrame,
    column: str,
    output_dir,
    output_file: str,
    lower_q: float = 0.01,
    upper_q: float = 0.99,
):
    """
    Boxplot univariado recortado entre percentiles.

    No elimina datos del dataset.
    Solo mejora la visualización de variables con colas largas.
    """
    series = _get_numeric_series(df, column)

    if series is None:
        return None

    lower = series.quantile(lower_q)
    upper = series.quantile(upper_q)
    trimmed = series[(series >= lower) & (series <= upper)]

    if trimmed.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.boxplot(trimmed, vert=False, showmeans=True)

    ax.set_title(f"Boxplot recortado p{int(lower_q * 100)}-p{int(upper_q * 100)} de {column}")
    ax.set_xlabel(column)
    ax.grid(True, alpha=0.3)

    return save_plot(fig, output_dir, output_file)


def plot_bar_chart(
    df: pd.DataFrame,
    column: str,
    title: str,
    xlabel: str,
    output_dir,
    output_file: str,
    top_n: int = 20,
):
    """
    Gráfico de barras univariado para variables categóricas.
    """
    if column not in df.columns:
        return None

    series = df[column].dropna().astype(str)

    if series.empty:
        return None

    counts = series.value_counts().head(top_n)

    fig, ax = plt.subplots(figsize=(10, 5))

    counts.plot(kind="bar", ax=ax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Frecuencia")
    ax.grid(True, axis="y", alpha=0.3)

    plt.xticks(rotation=45, ha="right")

    return save_plot(fig, output_dir, output_file)


def plot_samples_over_time(
    df: pd.DataFrame,
    output_dir,
    output_file: str = "samples_over_time.png",
):
    """
    Gráfico univariado temporal: cantidad de registros por fecha.

    No cruza variables de negocio contra tiempo.
    Solo muestra la frecuencia de muestras por unidad temporal.
    """
    date_column = None

    for candidate in ["simulation_date", "created_date"]:
        if candidate in df.columns:
            date_column = candidate
            break

    if date_column is None:
        return None

    temp = df[[date_column]].dropna().copy()

    if temp.empty:
        return None

    temp[date_column] = pd.to_datetime(temp[date_column], errors="coerce")
    temp = temp.dropna()

    if temp.empty:
        return None

    counts = (
        temp[date_column]
        .dt.date
        .value_counts()
        .sort_index()
    )

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(counts.index, counts.values, marker="o", linewidth=2)

    ax.set_title("Frecuencia de muestras por fecha")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Número de registros")
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45, ha="right")

    return save_plot(fig, output_dir, output_file)


def generate_descriptive_plots(
    df: pd.DataFrame,
    output_dir,
    histogram_vars: list[str] | None = None,
    log_histogram_vars: list[str] | None = None,
    boxplot_vars: list[str] | None = None,
    trimmed_boxplot_vars: list[str] | None = None,
    categorical_vars: list[str] | None = None,
    include_samples_over_time: bool = True,
) -> list[tuple[str, str]]:
    """
    Genera gráficos descriptivos estrictamente univariados.

    Incluye:
    - histogramas
    - histogramas log1p
    - boxplots
    - boxplots recortados para visualización
    - barras de frecuencia categóricas
    - conteo de muestras en el tiempo

    No incluye:
    - correlaciones
    - heatmaps
    - scatter plots
    - comparaciones entre variables
    - boxplots por grupo
    """

    plots = []

    if histogram_vars is None:
        histogram_vars = [
            "size_mb",
            "transfer_duration_sec",
            "storage_cost",
            "transfer_speed_mbps",
        ]

    if log_histogram_vars is None:
        log_histogram_vars = [
            "size_mb",
            "transfer_duration_sec",
            "storage_cost",
            "transfer_speed_mbps",
        ]

    if boxplot_vars is None:
        boxplot_vars = [
            "size_mb",
            "transfer_duration_sec",
            "storage_cost",
        ]

    if trimmed_boxplot_vars is None:
        trimmed_boxplot_vars = [
            "size_mb",
            "transfer_duration_sec",
            "storage_cost",
        ]

    if categorical_vars is None:
        categorical_vars = [
            "file_type",
            "storage_tier",
            "has_error",
            "severity",
        ]

    for column in histogram_vars:
        output_file = plot_histogram(
            df=df,
            column=column,
            title=f"Distribución de {column}",
            xlabel=column,
            output_dir=output_dir,
            output_file=f"{column}_histogram.png",
        )

        if output_file:
            plots.append((f"Histograma - {column}", output_file))

    for column in log_histogram_vars:
        output_file = plot_log_histogram(
            df=df,
            column=column,
            output_dir=output_dir,
            output_file=f"{column}_log_histogram.png",
        )

        if output_file:
            plots.append((f"Histograma log1p - {column}", output_file))

    for column in boxplot_vars:
        output_file = plot_boxplot(
            df=df,
            column=column,
            title=f"Boxplot de {column}",
            output_dir=output_dir,
            output_file=f"{column}_boxplot.png",
        )

        if output_file:
            plots.append((f"Boxplot - {column}", output_file))

    for column in trimmed_boxplot_vars:
        output_file = plot_trimmed_boxplot(
            df=df,
            column=column,
            output_dir=output_dir,
            output_file=f"{column}_trimmed_boxplot.png",
        )

        if output_file:
            plots.append((f"Boxplot recortado - {column}", output_file))

    for column in categorical_vars:
        output_file = plot_bar_chart(
            df=df,
            column=column,
            title=f"Frecuencia de {column}",
            xlabel=column,
            output_dir=output_dir,
            output_file=f"{column}_bar_chart.png",
        )

        if output_file:
            plots.append((f"Barras - {column}", output_file))

    if include_samples_over_time:
        output_file = plot_samples_over_time(
            df=df,
            output_dir=output_dir,
        )

        if output_file:
            plots.append(("Frecuencia de muestras en el tiempo", output_file))

    return plots