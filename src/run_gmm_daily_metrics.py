import argparse
import glob
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.patches import Ellipse
from scipy.stats import norm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore", category=UserWarning)


REQUIRED_COLUMNS = [
    "simulation_date",
    "size_mb",
    "transfer_speed_mbps",
    "storage_cost",
    "queue_pressure",
]


GMM_FEATURES = [
    "log_total_size_mb",
    "log_file_count",
    "avg_transfer_speed_mbps",
    "avg_queue_pressure",
    "log_storage_cost",
]


FEATURE_EXPLANATIONS = {
    "log_total_size_mb": {
        "original": "total_size_mb",
        "theory": "Transformación logarítmica",
        "formula": "log(1 + total_size_mb)",
        "why": (
            "El tamaño total puede crecer mucho en días de alta carga. "
            "El logaritmo reduce el impacto de valores extremos y permite que "
            "el modelo compare mejor días normales contra días pesados."
        ),
    },
    "log_file_count": {
        "original": "file_count",
        "theory": "Transformación logarítmica",
        "formula": "log(1 + file_count)",
        "why": (
            "La cantidad de archivos representa volumen operativo. "
            "Al aplicar logaritmo se evita que días con muchísimos archivos "
            "dominen completamente el clustering."
        ),
    },
    "avg_transfer_speed_mbps": {
        "original": "transfer_speed_mbps",
        "theory": "Promedio aritmético",
        "formula": "sum(transfer_speed_mbps) / n",
        "why": (
            "Representa el rendimiento promedio de transferencia del BlobStorage. "
            "Ayuda a separar días rápidos de días lentos o congestionados."
        ),
    },
    "avg_queue_pressure": {
        "original": "queue_pressure",
        "theory": "Promedio aritmético",
        "formula": "sum(queue_pressure) / n",
        "why": (
            "Representa la presión promedio de cola. "
            "Sirve para identificar congestión, acumulación de procesos o saturación."
        ),
    },
    "log_storage_cost": {
        "original": "total_storage_cost",
        "theory": "Transformación logarítmica",
        "formula": "log(1 + total_storage_cost)",
        "why": (
            "El costo puede variar mucho entre días. "
            "El logaritmo estabiliza la escala y permite comparar costo con carga y rendimiento."
        ),
    },
}


GRAPH_EXPLANATIONS = {
    "kmeans_elbow_method.png": {
        "title": "Método del codo aplicado con K-Means",
        "why_selected": "Justifica visualmente una cantidad razonable de clusters.",
        "theory": "La inercia mide compactación interna. El codo aparece cuando agregar clusters deja de aportar mucho.",
        "how_to_read": "El punto donde la curva se aplana sugiere un número razonable de clusters.",
    },
    "kmeans_silhouette_method.png": {
        "title": "Evaluación Silhouette aplicada",
        "why_selected": "Complementa el codo midiendo separación entre grupos.",
        "theory": "Silhouette compara cercanía al propio cluster contra cercanía a otros clusters.",
        "how_to_read": "Valores más altos indican clusters más separados y coherentes.",
    },
    "gmm_bic_selection.png": {
        "title": "Selección de componentes GMM usando BIC",
        "why_selected": "Permite justificar el número final de componentes gaussianos.",
        "theory": "BIC equilibra ajuste del modelo y complejidad.",
        "how_to_read": "El menor BIC indica el modelo más conveniente.",
    },
    "size_mb_original_log_mixed.png": {
        "title": "Comparación mixta: size_mb original vs logarítmico",
        "why_selected": "Muestra por qué se usa logaritmo antes del clustering.",
        "theory": "El logaritmo comprime extremos y reduce sesgo positivo.",
        "how_to_read": "La distribución logarítmica debe verse más compacta que la original.",
    },
    "daily_log_total_size_gmm_components.png": {
        "title": "Componentes gaussianos GMM sobre log_total_size_mb",
        "why_selected": "Es el gráfico más claro para explicar que GMM mezcla varias gaussianas.",
        "theory": "Cada curva es una gaussiana marginal del modelo para la variable log_total_size_mb.",
        "how_to_read": "Curvas separadas sugieren patrones distintos de carga diaria.",
    },
    "gmm_pca_mixture_ellipses.png": {
        "title": "Visualización PCA con elipses gaussianas",
        "why_selected": "Reduce las variables del GMM a dos dimensiones y muestra la forma de los clusters.",
        "theory": "PCA proyecta las variables normalizadas y las elipses muestran dispersión gaussiana.",
        "how_to_read": "Cada punto es un día; cada elipse representa la zona de mayor densidad de un componente GMM.",
    },
    "gmm_probability_heatmap.png": {
        "title": "Mapa de calor de probabilidades GMM",
        "why_selected": "Explica que GMM no asigna de forma rígida como K-Means.",
        "theory": "GMM calcula probabilidad de pertenencia a cada componente.",
        "how_to_read": "Colores altos indican mayor probabilidad de pertenecer a un cluster.",
    },
    "gmm_cluster_profile_radar.png": {
        "title": "Perfil mixto promedio por cluster",
        "why_selected": "Resume varias variables en un solo gráfico interpretativo.",
        "theory": "Normaliza promedios de variables por cluster para comparar perfiles operativos.",
        "how_to_read": "Cada línea representa un cluster y muestra su perfil de carga, costo, velocidad y presión.",
    },
    "gmm_scatter_size_count.png": {
        "title": "GMM - Tamaño total vs cantidad de archivos",
        "why_selected": "Muestra la relación entre volumen y número de archivos.",
        "theory": "Usa variables logarítmicas para comparar días de distinta escala.",
        "how_to_read": "Cada punto es un día; color = cluster; tamaño = costo.",
    },
    "gmm_anomalies.png": {
        "title": "Detección de anomalías GMM",
        "why_selected": "Muestra días poco probables para el modelo.",
        "theory": "El log-likelihood bajo indica baja probabilidad bajo la mezcla gaussiana.",
        "how_to_read": "Los puntos marcados como anomalía son los días menos probables.",
    },
}


def df_to_markdown_safe(df: pd.DataFrame, index: bool = False) -> str:
    """
    Evita que el programa falle si no está instalada la dependencia opcional tabulate.
    """
    try:
        return df.to_markdown(index=index)
    except ImportError:
        return "```text\n" + df.to_string(index=index) + "\n```"


def load_csv_files(input_dir: str) -> pd.DataFrame:
    pattern = os.path.join(input_dir, "*.csv")
    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No se encontraron archivos CSV en: {input_dir}")

    frames = []

    for file in files:
        try:
            temp = pd.read_csv(file)
            temp["source_file"] = os.path.basename(file)
            frames.append(temp)
            print(f"[OK] Cargado: {file}")
        except Exception as exc:
            print(f"[WARN] No se pudo leer {file}: {exc}")

    if not frames:
        raise ValueError("No se pudo cargar ningún CSV válido.")

    return pd.concat(frames, ignore_index=True)


def validate_columns(df: pd.DataFrame) -> None:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(f"Columnas faltantes en dataset: {missing}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    validate_columns(df)

    df["simulation_date"] = pd.to_datetime(df["simulation_date"], errors="coerce")

    numeric_columns = [
        "size_mb",
        "transfer_speed_mbps",
        "storage_cost",
        "queue_pressure",
    ]

    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["simulation_date", *numeric_columns])

    df = df[df["size_mb"] >= 0]
    df = df[df["transfer_speed_mbps"] >= 0]
    df = df[df["storage_cost"] >= 0]
    df = df[df["queue_pressure"] >= 0]

    return df


def build_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby("simulation_date")
        .agg(
            total_size_mb=("size_mb", "sum"),
            avg_size_mb=("size_mb", "mean"),
            median_size_mb=("size_mb", "median"),
            max_size_mb=("size_mb", "max"),
            file_count=("size_mb", "count"),
            avg_transfer_speed_mbps=("transfer_speed_mbps", "mean"),
            min_transfer_speed_mbps=("transfer_speed_mbps", "min"),
            total_storage_cost=("storage_cost", "sum"),
            avg_storage_cost=("storage_cost", "mean"),
            avg_queue_pressure=("queue_pressure", "mean"),
            max_queue_pressure=("queue_pressure", "max"),
        )
        .reset_index()
    )

    daily["simulation_date"] = daily["simulation_date"].dt.date

    daily["log_total_size_mb"] = np.log1p(daily["total_size_mb"])
    daily["log_file_count"] = np.log1p(daily["file_count"])
    daily["log_storage_cost"] = np.log1p(daily["total_storage_cost"])

    return daily


def build_scaled_matrix(daily_metrics: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    X = daily_metrics[GMM_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def calculate_elbow_method(
    daily_metrics: pd.DataFrame,
    min_clusters: int,
    max_clusters: int,
) -> pd.DataFrame:
    X_scaled, _ = build_scaled_matrix(daily_metrics)

    scores = []
    max_valid_clusters = min(max_clusters, len(daily_metrics))

    for k in range(min_clusters, max_valid_clusters + 1):
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10,
        )

        labels = kmeans.fit_predict(X_scaled)
        inertia = kmeans.inertia_

        if k > 1 and len(set(labels)) > 1:
            silhouette = silhouette_score(X_scaled, labels)
        else:
            silhouette = np.nan

        scores.append(
            {
                "n_clusters": k,
                "inertia": inertia,
                "silhouette_score": silhouette,
            }
        )

    return pd.DataFrame(scores)


def select_best_gmm_components(
    daily_metrics: pd.DataFrame,
    min_components: int,
    max_components: int,
) -> tuple[int, pd.DataFrame]:
    X_scaled, _ = build_scaled_matrix(daily_metrics)

    scores = []
    max_valid_components = min(max_components, len(daily_metrics))

    for k in range(min_components, max_valid_components + 1):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=42,
        )

        gmm.fit(X_scaled)

        scores.append(
            {
                "n_components": k,
                "bic": gmm.bic(X_scaled),
                "aic": gmm.aic(X_scaled),
            }
        )

    scores_df = pd.DataFrame(scores)
    best_k = int(scores_df.sort_values("bic").iloc[0]["n_components"])

    return best_k, scores_df


def apply_gmm(
    daily_metrics: pd.DataFrame,
    n_components: int,
) -> tuple[pd.DataFrame, GaussianMixture, StandardScaler, np.ndarray]:
    result = daily_metrics.copy()
    X_scaled, scaler = build_scaled_matrix(result)

    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type="full",
        random_state=42,
    )

    gmm.fit(X_scaled)

    result["gmm_cluster"] = gmm.predict(X_scaled)
    result["gmm_log_likelihood"] = gmm.score_samples(X_scaled)

    probabilities = gmm.predict_proba(X_scaled)

    for i in range(n_components):
        result[f"prob_cluster_{i}"] = probabilities[:, i]

    result["max_cluster_probability"] = probabilities.max(axis=1)
    result["is_low_confidence"] = result["max_cluster_probability"] < 0.70

    anomaly_threshold = result["gmm_log_likelihood"].quantile(0.05)
    result["is_anomaly"] = result["gmm_log_likelihood"] <= anomaly_threshold

    return result, gmm, scaler, X_scaled


def build_cluster_summary(result: pd.DataFrame) -> pd.DataFrame:
    summary = (
        result.groupby("gmm_cluster")
        .agg(
            days=("simulation_date", "count"),
            avg_total_size_mb=("total_size_mb", "mean"),
            avg_file_count=("file_count", "mean"),
            avg_transfer_speed_mbps=("avg_transfer_speed_mbps", "mean"),
            avg_storage_cost=("total_storage_cost", "mean"),
            avg_queue_pressure=("avg_queue_pressure", "mean"),
            avg_likelihood=("gmm_log_likelihood", "mean"),
            anomalies=("is_anomaly", "sum"),
            low_confidence_days=("is_low_confidence", "sum"),
            avg_cluster_probability=("max_cluster_probability", "mean"),
        )
        .reset_index()
    )

    return summary


def classify_cluster(row: pd.Series, global_means: pd.Series) -> str:
    labels = []

    if row["avg_total_size_mb"] > global_means["total_size_mb"]:
        labels.append("alta carga")
    else:
        labels.append("carga normal o baja")

    if row["avg_storage_cost"] > global_means["total_storage_cost"]:
        labels.append("alto costo")

    if row["avg_transfer_speed_mbps"] < global_means["avg_transfer_speed_mbps"]:
        labels.append("menor velocidad")

    if row["avg_queue_pressure"] > global_means["avg_queue_pressure"]:
        labels.append("mayor presión de cola")

    if row["anomalies"] > 0:
        labels.append("incluye anomalías")

    return ", ".join(labels)


def plot_covariance_ellipse(mean: np.ndarray, covariance: np.ndarray, ax, label: str) -> None:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

    # 2 desviaciones estándar aprox. cubren buena parte de la densidad gaussiana.
    width, height = 2 * 2 * np.sqrt(eigenvalues)

    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        fill=False,
        linewidth=2,
        label=label,
    )

    ax.add_patch(ellipse)


def generate_mixed_gmm_visualizations(
    clean_df: pd.DataFrame,
    result: pd.DataFrame,
    gmm: GaussianMixture,
    scaler: StandardScaler,
    X_scaled: np.ndarray,
    output_dir: str,
) -> None:
    report_path = Path(output_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    # 1. Distribución original y logarítmica de size_mb.
    sample_size = min(len(clean_df), 100_000)
    plot_df = clean_df.sample(sample_size, random_state=42) if len(clean_df) > sample_size else clean_df.copy()
    plot_df["log_size_mb"] = np.log1p(plot_df["size_mb"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.histplot(plot_df["size_mb"], bins=80, kde=True, ax=axes[0])
    axes[0].set_title("Distribución original de size_mb")
    axes[0].set_xlabel("size_mb")
    axes[0].set_ylabel("Frecuencia")

    sns.histplot(plot_df["log_size_mb"], bins=80, kde=True, ax=axes[1])
    axes[1].set_title("Distribución logarítmica de size_mb")
    axes[1].set_xlabel("log(1 + size_mb)")
    axes[1].set_ylabel("Frecuencia")

    plt.suptitle("Comportamiento mixto: variable original vs transformación logarítmica")
    plt.tight_layout()
    plt.savefig(report_path / "size_mb_original_log_mixed.png", dpi=150)
    plt.close()

    # 2. Componentes gaussianos marginales para log_total_size_mb.
    feature_idx = GMM_FEATURES.index("log_total_size_mb")
    log_total_size = result["log_total_size_mb"].to_numpy()

    scaled_feature_values = X_scaled[:, feature_idx]
    x_scaled = np.linspace(scaled_feature_values.min() - 0.8, scaled_feature_values.max() + 0.8, 1000)

    plt.figure(figsize=(12, 7))
    sns.histplot(
        scaled_feature_values,
        bins=18,
        stat="density",
        alpha=0.35,
        label="Distribución real normalizada",
    )

    total_density = np.zeros_like(x_scaled)

    for i in range(gmm.n_components):
        mean_i = gmm.means_[i, feature_idx]
        variance_i = gmm.covariances_[i, feature_idx, feature_idx]
        std_i = np.sqrt(variance_i)
        weight_i = gmm.weights_[i]

        density_i = weight_i * norm.pdf(x_scaled, mean_i, std_i)
        total_density += density_i

        plt.plot(
            x_scaled,
            density_i,
            linewidth=2,
            label=f"Gaussiana {i} | peso={weight_i:.2f}",
        )

    plt.plot(
        x_scaled,
        total_density,
        linestyle="--",
        linewidth=3,
        label="Mezcla total GMM",
    )

    plt.title("Componentes gaussianos GMM sobre log_total_size_mb normalizado")
    plt.xlabel("z-score de log_total_size_mb")
    plt.ylabel("Densidad")
    plt.legend()
    plt.tight_layout()
    plt.savefig(report_path / "daily_log_total_size_gmm_components.png", dpi=150)
    plt.close()

    # 3. PCA con elipses gaussianas proyectadas.
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    means_pca = pca.transform(gmm.means_)
    covariances_pca = []

    for cov in gmm.covariances_:
        cov_pca = pca.components_ @ cov @ pca.components_.T
        covariances_pca.append(cov_pca)

    pca_df = pd.DataFrame(
        {
            "pc1": X_pca[:, 0],
            "pc2": X_pca[:, 1],
            "gmm_cluster": result["gmm_cluster"],
            "is_anomaly": result["is_anomaly"],
            "max_cluster_probability": result["max_cluster_probability"],
            "simulation_date": result["simulation_date"].astype(str),
        }
    )

    fig, ax = plt.subplots(figsize=(11, 8))
    sns.scatterplot(
        data=pca_df,
        x="pc1",
        y="pc2",
        hue="gmm_cluster",
        style="is_anomaly",
        size="max_cluster_probability",
        sizes=(80, 250),
        ax=ax,
    )

    for i, cov_pca in enumerate(covariances_pca):
        plot_covariance_ellipse(
            means_pca[i],
            cov_pca,
            ax,
            label=f"Elipse gaussiana {i}",
        )

    ax.set_title("PCA + elipses gaussianas del modelo GMM")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% varianza)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% varianza)")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(report_path / "gmm_pca_mixture_ellipses.png", dpi=150)
    plt.close()

    # 4. Heatmap de probabilidades por día y componente.
    probability_columns = [f"prob_cluster_{i}" for i in range(gmm.n_components)]
    probability_matrix = result[probability_columns].copy()
    probability_matrix.index = result["simulation_date"].astype(str)

    plt.figure(figsize=(12, 8))
    sns.heatmap(
        probability_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        cbar_kws={"label": "Probabilidad"},
    )
    plt.title("Mapa de calor de probabilidades de pertenencia GMM")
    plt.xlabel("Componente GMM")
    plt.ylabel("Día")
    plt.tight_layout()
    plt.savefig(report_path / "gmm_probability_heatmap.png", dpi=150)
    plt.close()

    # 5. Radar de perfil promedio por cluster.
    profile_features = [
        "total_size_mb",
        "file_count",
        "avg_transfer_speed_mbps",
        "avg_queue_pressure",
        "total_storage_cost",
    ]

    cluster_profile = (
        result.groupby("gmm_cluster")[profile_features]
        .mean()
        .reset_index()
    )

    normalized_profile = cluster_profile.copy()

    for feature in profile_features:
        min_value = normalized_profile[feature].min()
        max_value = normalized_profile[feature].max()
        if max_value == min_value:
            normalized_profile[feature] = 0.5
        else:
            normalized_profile[feature] = (
                normalized_profile[feature] - min_value
            ) / (max_value - min_value)

    labels = [
        "Tamaño",
        "Archivos",
        "Velocidad",
        "Presión cola",
        "Costo",
    ]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9, 8))
    ax = plt.subplot(111, polar=True)

    for _, row in normalized_profile.iterrows():
        values = [row[feature] for feature in profile_features]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=f"Cluster {int(row['gmm_cluster'])}")
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Perfil mixto promedio por cluster GMM", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig(report_path / "gmm_cluster_profile_radar.png", dpi=150)
    plt.close()


def generate_markdown_report(
    result: pd.DataFrame,
    bic_scores: pd.DataFrame,
    elbow_scores: pd.DataFrame,
    output_dir: str,
    best_k: int,
) -> None:
    report_path = Path(output_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    cluster_summary = build_cluster_summary(result)

    global_means = pd.Series(
        {
            "total_size_mb": result["total_size_mb"].mean(),
            "total_storage_cost": result["total_storage_cost"].mean(),
            "avg_transfer_speed_mbps": result["avg_transfer_speed_mbps"].mean(),
            "avg_queue_pressure": result["avg_queue_pressure"].mean(),
        }
    )

    cluster_summary["interpretacion"] = cluster_summary.apply(
        lambda row: classify_cluster(row, global_means),
        axis=1,
    )

    report_file = report_path / "reporte_explicativo_gmm_blobstorage.md"

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Reporte explicativo - Clustering GMM aplicado a BlobStorage\n\n")

        f.write("## 1. Objetivo del análisis\n\n")
        f.write(
            "Este reporte analiza métricas simuladas u operativas de Azure Blob Storage "
            "para identificar patrones de comportamiento diario, agrupar días similares "
            "mediante clustering probabilístico y detectar posibles anomalías.\n\n"
        )

        f.write("## 2. Flujo general del código\n\n")
        f.write("```text\n")
        f.write("CSV de entrada\n")
        f.write("  ↓\n")
        f.write("Validación y limpieza de datos\n")
        f.write("  ↓\n")
        f.write("Agrupación diaria de métricas\n")
        f.write("  ↓\n")
        f.write("Transformaciones logarítmicas\n")
        f.write("  ↓\n")
        f.write("Normalización con StandardScaler\n")
        f.write("  ↓\n")
        f.write("Selección de componentes con BIC\n")
        f.write("  ↓\n")
        f.write("Aplicación de Gaussian Mixture Model\n")
        f.write("  ↓\n")
        f.write("Asignación probabilística de clusters\n")
        f.write("  ↓\n")
        f.write("Visualizaciones mixtas: histogramas, gaussianas, PCA, heatmap y radar\n")
        f.write("```\n\n")

        f.write("## 3. Variables usadas en el modelo\n\n")
        for feature, info in FEATURE_EXPLANATIONS.items():
            f.write(f"### {feature}\n\n")
            f.write(f"- Variable original: `{info['original']}`\n")
            f.write(f"- Teoría aplicada: {info['theory']}\n")
            f.write(f"- Fórmula: `{info['formula']}`\n")
            f.write(f"- Justificación: {info['why']}\n\n")

        f.write("## 4. Teoría del modelo GMM\n\n")
        f.write(
            "GMM significa Gaussian Mixture Model. Es un modelo de clustering probabilístico. "
            "A diferencia de K-Means, que asigna cada observación a un único grupo de forma rígida, "
            "GMM calcula la probabilidad de pertenencia de cada día a cada cluster.\n\n"
        )
        f.write(
            "La idea central es que los datos se explican como una mezcla de distribuciones gaussianas:\n\n"
        )
        f.write("```text\n")
        f.write("P(x) = peso_1 * Gaussiana_1 + peso_2 * Gaussiana_2 + ... + peso_k * Gaussiana_k\n")
        f.write("```\n\n")
        f.write(
            "Esto es útil para BlobStorage porque los días no siempre son completamente normales "
            "o completamente anómalos. Un día puede tener comportamiento mixto: carga media, "
            "costo alto, baja velocidad o presión de cola elevada.\n\n"
        )

        f.write("## 5. Método del codo aplicado\n\n")
        f.write(
            "El método del codo se aplica con K-Means como herramienta visual para analizar "
            "cuántos grupos naturales podrían existir. Aunque el modelo final usa GMM, "
            "el codo ayuda a explicar la segmentación.\n\n"
        )
        f.write("### Resultados del método del codo\n\n")
        f.write(df_to_markdown_safe(elbow_scores, index=False))
        f.write("\n\n")

        f.write("## 6. Selección del número de clusters con GMM\n\n")
        f.write(f"El mejor número de componentes seleccionado por BIC fue: **{best_k}**.\n\n")
        f.write(
            "BIC penaliza modelos demasiado complejos. Por eso, no siempre gana el modelo "
            "con más clusters, sino el que logra mejor equilibrio entre ajuste y simplicidad.\n\n"
        )
        f.write("### Resultados BIC/AIC\n\n")
        f.write(df_to_markdown_safe(bic_scores, index=False))
        f.write("\n\n")

        f.write("## 7. Interpretación automática de clusters\n\n")
        f.write(df_to_markdown_safe(cluster_summary, index=False))
        f.write("\n\n")

        f.write("## 8. Explicación de gráficos generados\n\n")
        for graph_file, explanation in GRAPH_EXPLANATIONS.items():
            f.write(f"### {explanation['title']}\n\n")
            f.write(f"Archivo generado: `{graph_file}`\n\n")
            f.write(f"![{explanation['title']}]({graph_file})\n\n")
            f.write(f"**Por qué se seleccionó:** {explanation['why_selected']}\n\n")
            f.write(f"**Teoría aplicada:** {explanation['theory']}\n\n")
            f.write(f"**Cómo interpretarlo:** {explanation['how_to_read']}\n\n")

        f.write("## 9. Detección de anomalías\n\n")
        f.write(
            "El código calcula `gmm_log_likelihood`, que representa qué tan probable es un día "
            "según el modelo aprendido. Los días con menor probabilidad son los más extraños.\n\n"
        )
        f.write(
            "Se usa el percentil 5 como umbral. Esto significa que el 5% de días menos probables "
            "se marcan como anomalías.\n\n"
        )

        top_anomalies = result.sort_values("gmm_log_likelihood").head(10)
        f.write("### Top 10 días más anómalos\n\n")
        f.write(
            df_to_markdown_safe(
                top_anomalies[
                    [
                        "simulation_date",
                        "total_size_mb",
                        "file_count",
                        "total_storage_cost",
                        "avg_queue_pressure",
                        "gmm_cluster",
                        "gmm_log_likelihood",
                        "max_cluster_probability",
                    ]
                ],
                index=False,
            )
        )
        f.write("\n\n")

        f.write("## 10. Conclusión para exposición\n\n")
        f.write(
            "Este código convierte datos operativos de BlobStorage en información analítica. "
            "Primero consolida CSV, limpia datos, construye métricas diarias, aplica transformaciones "
            "logarítmicas y normalización, selecciona componentes mediante BIC y aplica GMM para "
            "encontrar patrones de comportamiento.\n\n"
        )
        f.write(
            "Los nuevos gráficos mixtos permiten entender mejor el modelo: muestran la transformación "
            "de `size_mb`, las gaussianas que componen el GMM, la separación PCA, la probabilidad de "
            "pertenencia a cada cluster y el perfil operativo de cada grupo.\n\n"
        )

    print(f"[OK] Reporte explicativo generado en: {report_file}")


def generate_reports(
    clean_df: pd.DataFrame,
    result: pd.DataFrame,
    bic_scores: pd.DataFrame,
    elbow_scores: pd.DataFrame,
    output_dir: str,
    best_k: int,
    gmm: GaussianMixture,
    scaler: StandardScaler,
    X_scaled: np.ndarray,
) -> None:
    report_path = Path(output_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    cluster_summary = build_cluster_summary(result)
    cluster_summary.to_csv(report_path / "gmm_cluster_summary.csv", index=False)

    anomalies = result.sort_values("gmm_log_likelihood").head(10)
    anomalies.to_csv(report_path / "gmm_top_anomalies.csv", index=False)

    bic_scores.to_csv(report_path / "gmm_model_selection_bic_aic.csv", index=False)
    elbow_scores.to_csv(report_path / "kmeans_elbow_scores.csv", index=False)

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=elbow_scores, x="n_clusters", y="inertia", marker="o")
    plt.title("Método del codo aplicado con K-Means")
    plt.xlabel("Número de clusters")
    plt.ylabel("Inercia")
    plt.tight_layout()
    plt.savefig(report_path / "kmeans_elbow_method.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=elbow_scores, x="n_clusters", y="silhouette_score", marker="o")
    plt.title("Evaluación Silhouette por número de clusters")
    plt.xlabel("Número de clusters")
    plt.ylabel("Silhouette Score")
    plt.tight_layout()
    plt.savefig(report_path / "kmeans_silhouette_method.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.lineplot(data=bic_scores, x="n_components", y="bic", marker="o")
    plt.title("Selección de componentes GMM usando BIC")
    plt.xlabel("Número de componentes")
    plt.ylabel("BIC")
    plt.tight_layout()
    plt.savefig(report_path / "gmm_bic_selection.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=result,
        x="log_total_size_mb",
        y="log_file_count",
        hue="gmm_cluster",
        size="total_storage_cost",
        sizes=(80, 400),
        palette="viridis",
    )
    plt.title("GMM - Tamaño total vs cantidad de archivos")
    plt.xlabel("log_total_size_mb")
    plt.ylabel("log_file_count")
    plt.tight_layout()
    plt.savefig(report_path / "gmm_scatter_size_count.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.countplot(
        data=result,
        x="gmm_cluster",
        hue="gmm_cluster",
        palette="viridis",
        legend=False,
    )
    plt.title("Cantidad de días por cluster")
    plt.xlabel("Cluster GMM")
    plt.ylabel("Cantidad de días")
    plt.tight_layout()
    plt.savefig(report_path / "gmm_cluster_counts.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=cluster_summary,
        x="gmm_cluster",
        y="avg_total_size_mb",
        hue="gmm_cluster",
        palette="viridis",
        legend=False,
    )
    plt.title("Tamaño promedio por cluster")
    plt.xlabel("Cluster GMM")
    plt.ylabel("Tamaño promedio MB")
    plt.tight_layout()
    plt.savefig(report_path / "gmm_avg_total_size_by_cluster.png", dpi=150)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.barplot(
        data=cluster_summary,
        x="gmm_cluster",
        y="avg_storage_cost",
        hue="gmm_cluster",
        palette="viridis",
        legend=False,
    )
    plt.title("Costo promedio por cluster")
    plt.xlabel("Cluster GMM")
    plt.ylabel("Costo promedio")
    plt.tight_layout()
    plt.savefig(report_path / "gmm_avg_storage_cost_by_cluster.png", dpi=150)
    plt.close()

    plt.figure(figsize=(12, 5))
    sns.lineplot(
        data=result,
        x="simulation_date",
        y="gmm_log_likelihood",
        marker="o",
    )
    plt.xticks(rotation=45)
    plt.title("Log-Likelihood GMM por día")
    plt.xlabel("Fecha")
    plt.ylabel("Log-Likelihood")
    plt.tight_layout()
    plt.savefig(report_path / "gmm_likelihood_by_day.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=result,
        x="total_size_mb",
        y="total_storage_cost",
        hue="is_anomaly",
        style="gmm_cluster",
        s=120,
    )
    plt.title("Detección de anomalías GMM")
    plt.xlabel("Tamaño total MB")
    plt.ylabel("Costo total storage")
    plt.tight_layout()
    plt.savefig(report_path / "gmm_anomalies.png", dpi=150)
    plt.close()

    generate_mixed_gmm_visualizations(
        clean_df=clean_df,
        result=result,
        gmm=gmm,
        scaler=scaler,
        X_scaled=X_scaled,
        output_dir=output_dir,
    )

    generate_markdown_report(result, bic_scores, elbow_scores, output_dir, best_k)

    print(f"[OK] Reportes generados en: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolida CSV diarios, genera métricas, aplica GMM y crea reporte explicativo."
    )

    parser.add_argument(
        "--input-dir",
        default="output/dataset",
        help="Carpeta de entrada con archivos CSV.",
    )

    parser.add_argument(
        "--output-file",
        default="output/gmm_daily_metrics.csv",
        help="Archivo CSV final.",
    )

    parser.add_argument(
        "--report-dir",
        default="output/reports",
        help="Carpeta para reportes, gráficos y explicación.",
    )

    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Número fijo de componentes GMM. Si no se pasa, se selecciona con BIC.",
    )

    parser.add_argument(
        "--min-components",
        type=int,
        default=2,
        help="Mínimo de componentes para búsqueda BIC.",
    )

    parser.add_argument(
        "--max-components",
        type=int,
        default=6,
        help="Máximo de componentes para búsqueda BIC.",
    )

    args = parser.parse_args()

    print("[INFO] Leyendo CSV...")
    raw_df = load_csv_files(args.input_dir)
    print(f"[INFO] Registros cargados: {len(raw_df):,}")

    print("[INFO] Limpiando datos...")
    clean_df = clean_data(raw_df)
    print(f"[INFO] Registros válidos: {len(clean_df):,}")

    print("[INFO] Construyendo métricas diarias...")
    daily_metrics = build_daily_metrics(clean_df)
    print(f"[INFO] Días procesados: {len(daily_metrics):,}")

    if len(daily_metrics) < 2:
        raise ValueError("Se necesitan al menos 2 días de datos para aplicar clustering GMM.")

    print("[INFO] Calculando método del codo aplicado con K-Means...")
    elbow_scores = calculate_elbow_method(
        daily_metrics,
        args.min_components,
        args.max_components,
    )

    if args.n_components is None:
        print("[INFO] Seleccionando mejor número de componentes con BIC...")
        best_k, bic_scores = select_best_gmm_components(
            daily_metrics,
            args.min_components,
            args.max_components,
        )
        print(f"[OK] Mejor número de componentes según BIC: {best_k}")
    else:
        best_k = args.n_components
        _, bic_scores = select_best_gmm_components(
            daily_metrics,
            max(1, best_k),
            best_k,
        )

    print("[INFO] Aplicando GMM...")
    result, gmm, scaler, X_scaled = apply_gmm(daily_metrics, best_k)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)

    print(f"[OK] Archivo generado: {output_path}")

    print("[INFO] Generando reportes visuales mixtos y explicación...")
    generate_reports(
        clean_df=clean_df,
        result=result,
        bic_scores=bic_scores,
        elbow_scores=elbow_scores,
        output_dir=args.report_dir,
        best_k=best_k,
        gmm=gmm,
        scaler=scaler,
        X_scaled=X_scaled,
    )

    print("[INFO] Resumen por cluster:")
    print(build_cluster_summary(result))

    print("[OK] Proceso finalizado correctamente.")


if __name__ == "__main__":
    main()
