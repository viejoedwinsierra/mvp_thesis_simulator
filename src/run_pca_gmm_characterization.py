from __future__ import annotations

"""
Runner de caracterización probabilística del simulador.

Objetivo:
    1. Construir métricas diarias por escenario.
    2. Aplicar PCA para validar estructura latente.
    3. Aplicar GMM para identificar composición probabilística.
    4. Ejecutar análisis global y/o por escenario.
    5. Exportar tablas, gráficos, HTML y manifiestos.

Uso recomendado:
    python src/run_pca_gmm_characterization.py --mode all
"""

import argparse
import json
import math
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = [
    "simulation_date",
    "size_mb",
    "transfer_speed_mbps",
    "storage_cost",
    "queue_pressure",
]

DATASET_PATTERN = "blob_inventory_*.csv"

DAILY_FEATURES = [
    "log_total_size_mb",
    "log_file_count",
    "avg_transfer_speed_mbps",
    "avg_queue_pressure",
    "log_storage_cost",
]

FEATURE_EXPLANATIONS = {
    "log_total_size_mb": "Logaritmo del tamaño total diario. Representa carga volumétrica reduciendo efecto de extremos.",
    "log_file_count": "Logaritmo de cantidad diaria de archivos. Representa transaccionalidad operativa.",
    "avg_transfer_speed_mbps": "Velocidad promedio diaria. Representa rendimiento operativo de transferencia.",
    "avg_queue_pressure": "Presión promedio diaria de cola. Representa congestión o saturación.",
    "log_storage_cost": "Logaritmo del costo diario total. Representa impacto financiero estabilizando escala.",
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_table(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_json(path: str | Path, data: dict) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def safe_rel(path: str | Path, start: str | Path) -> str:
    return os.path.relpath(Path(path), start=Path(start)).replace("\\", "/")


def fmt(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        if pd.isna(value):
            return "N/A"
    except Exception:
        pass
    if isinstance(value, (int, float, np.floating)):
        if math.isfinite(float(value)):
            return f"{float(value):,.{digits}f}"
    return str(value)


def discover_dataset_files(dataset_root: str | Path) -> list[Path]:
    dataset_root = Path(dataset_root)

    if not dataset_root.exists():
        raise FileNotFoundError(f"No existe dataset_root: {dataset_root}")

    files = []

    for file in sorted(dataset_root.rglob(DATASET_PATTERN)):
        try:
            rel = file.relative_to(dataset_root)
        except ValueError:
            continue

        # Regla estricta del proyecto:
        # output/dataset/<escenario>/blob_inventory_<escenario>_<fecha>.csv
        if len(rel.parts) != 2:
            continue

        if file.name == "blob_inventory.csv":
            continue

        files.append(file)

    if not files:
        raise FileNotFoundError(
            "No se encontraron datasets finales. "
            "Estructura esperada: output/dataset/<escenario>/blob_inventory_<escenario>_<fecha>.csv"
        )

    return files


def load_datasets(dataset_root: str | Path) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    files = discover_dataset_files(dataset_root)

    frames = []

    for file in files:
        df = pd.read_csv(file)
        rel = file.relative_to(dataset_root)
        scenario = rel.parts[0]
        df["scenario_name"] = scenario
        df["source_file"] = file.name
        df["source_path"] = str(file)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    print("=== ARCHIVOS CARGADOS PARA PCA/GMM ===")
    for file in files:
        print(file)

    return raw


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Columnas requeridas faltantes: {missing}")

    df = df.copy()

    df["simulation_date"] = pd.to_datetime(df["simulation_date"], errors="coerce")

    for col in ["size_mb", "transfer_speed_mbps", "storage_cost", "queue_pressure"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["simulation_date", "size_mb", "transfer_speed_mbps", "storage_cost", "queue_pressure"])
    df = df[df["size_mb"] >= 0]
    df = df[df["transfer_speed_mbps"] >= 0]
    df = df[df["storage_cost"] >= 0]
    df = df[df["queue_pressure"] >= 0]

    if "scenario_name" not in df.columns:
        df["scenario_name"] = "global"

    return df


def build_daily_metrics(df: pd.DataFrame) -> pd.DataFrame:
    daily = (
        df.groupby(["scenario_name", "simulation_date"])
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

    daily["simulation_date"] = pd.to_datetime(daily["simulation_date"]).dt.date

    daily["log_total_size_mb"] = np.log1p(daily["total_size_mb"])
    daily["log_file_count"] = np.log1p(daily["file_count"])
    daily["log_storage_cost"] = np.log1p(daily["total_storage_cost"])

    return daily


def build_scaled_matrix(daily_metrics: pd.DataFrame) -> tuple[np.ndarray, StandardScaler]:
    X = daily_metrics[DAILY_FEATURES].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def run_pca(daily_metrics: pd.DataFrame) -> tuple[PCA, np.ndarray, pd.DataFrame, StandardScaler]:
    X_scaled, scaler = build_scaled_matrix(daily_metrics)

    if X_scaled.shape[0] < 2:
        raise ValueError("Se requieren al menos 2 observaciones para PCA 2D.")

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    loadings = pd.DataFrame(
        pca.components_.T,
        columns=["PC1", "PC2"],
        index=DAILY_FEATURES,
    ).reset_index(names="feature")

    loadings["abs_PC1"] = loadings["PC1"].abs()
    loadings["abs_PC2"] = loadings["PC2"].abs()

    return pca, X_pca, loadings, scaler


def calculate_kmeans_reference(daily_metrics: pd.DataFrame, min_k: int, max_k: int) -> pd.DataFrame:
    X_scaled, _ = build_scaled_matrix(daily_metrics)
    max_valid = min(max_k, len(daily_metrics) - 1)

    rows = []

    if max_valid < min_k:
        return pd.DataFrame(columns=["n_clusters", "inertia", "silhouette_score"])

    for k in range(min_k, max_valid + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)

        sil = np.nan
        if len(set(labels)) > 1:
            sil = silhouette_score(X_scaled, labels)

        rows.append({
            "n_clusters": k,
            "inertia": km.inertia_,
            "silhouette_score": sil,
        })

    return pd.DataFrame(rows)


def select_gmm_components(daily_metrics: pd.DataFrame, min_components: int, max_components: int) -> tuple[int, pd.DataFrame]:
    X_scaled, _ = build_scaled_matrix(daily_metrics)
    max_valid = min(max_components, len(daily_metrics) - 1)

    if max_valid < min_components:
        best_k = 1
        gmm = GaussianMixture(n_components=1, covariance_type="full", random_state=42)
        gmm.fit(X_scaled)
        return best_k, pd.DataFrame([{
            "n_components": 1,
            "bic": gmm.bic(X_scaled),
            "aic": gmm.aic(X_scaled),
        }])

    rows = []

    for k in range(min_components, max_valid + 1):
        gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=42)
        gmm.fit(X_scaled)

        rows.append({
            "n_components": k,
            "bic": gmm.bic(X_scaled),
            "aic": gmm.aic(X_scaled),
        })

    scores = pd.DataFrame(rows)
    best_k = int(scores.sort_values("bic").iloc[0]["n_components"])

    return best_k, scores


def apply_gmm(daily_metrics: pd.DataFrame, n_components: int) -> tuple[pd.DataFrame, GaussianMixture, np.ndarray, StandardScaler]:
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

    threshold = result["gmm_log_likelihood"].quantile(0.05)
    result["is_anomaly"] = result["gmm_log_likelihood"] <= threshold

    return result, gmm, X_scaled, scaler


def build_cluster_summary(result: pd.DataFrame) -> pd.DataFrame:
    return (
        result.groupby("gmm_cluster")
        .agg(
            observations=("simulation_date", "count"),
            scenarios=("scenario_name", "nunique"),
            avg_total_size_mb=("total_size_mb", "mean"),
            avg_file_count=("file_count", "mean"),
            avg_transfer_speed_mbps=("avg_transfer_speed_mbps", "mean"),
            avg_storage_cost=("total_storage_cost", "mean"),
            avg_queue_pressure=("avg_queue_pressure", "mean"),
            avg_likelihood=("gmm_log_likelihood", "mean"),
            avg_cluster_probability=("max_cluster_probability", "mean"),
            anomalies=("is_anomaly", "sum"),
            low_confidence=("is_low_confidence", "sum"),
        )
        .reset_index()
    )


def classify_cluster(row: pd.Series, global_means: pd.Series) -> str:
    labels = []

    if row["avg_total_size_mb"] > global_means["total_size_mb"]:
        labels.append("alta carga")
    else:
        labels.append("carga normal/baja")

    if row["avg_storage_cost"] > global_means["total_storage_cost"]:
        labels.append("alto costo")

    if row["avg_transfer_speed_mbps"] < global_means["avg_transfer_speed_mbps"]:
        labels.append("menor velocidad")

    if row["avg_queue_pressure"] > global_means["avg_queue_pressure"]:
        labels.append("mayor presión de cola")

    if row["anomalies"] > 0:
        labels.append("incluye anomalías")

    return ", ".join(labels)


def add_cluster_interpretation(result: pd.DataFrame, cluster_summary: pd.DataFrame) -> pd.DataFrame:
    means = pd.Series({
        "total_size_mb": result["total_size_mb"].mean(),
        "total_storage_cost": result["total_storage_cost"].mean(),
        "avg_transfer_speed_mbps": result["avg_transfer_speed_mbps"].mean(),
        "avg_queue_pressure": result["avg_queue_pressure"].mean(),
    })

    cluster_summary = cluster_summary.copy()
    cluster_summary["interpretation"] = cluster_summary.apply(
        lambda row: classify_cluster(row, means),
        axis=1,
    )

    return cluster_summary


def save_fig(path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_pca_scatter(scope_result: pd.DataFrame, X_pca: np.ndarray, pca: PCA, output_dir: Path, title: str) -> Path:
    plot_df = scope_result.copy()
    plot_df["PC1"] = X_pca[:, 0]
    plot_df["PC2"] = X_pca[:, 1]

    plt.figure(figsize=(11, 7))

    clusters = sorted(plot_df["gmm_cluster"].unique())

    for cluster in clusters:
        subset = plot_df[plot_df["gmm_cluster"] == cluster]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            s=90,
            alpha=0.75,
            label=f"Cluster {cluster}",
        )

    anomalies = plot_df[plot_df["is_anomaly"]]
    if not anomalies.empty:
        plt.scatter(
            anomalies["PC1"],
            anomalies["PC2"],
            s=180,
            facecolors="none",
            edgecolors="black",
            linewidths=1.8,
            label="Anomalía",
        )

    plt.title(title)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% varianza)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% varianza)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    return save_fig(output_dir / "pca_gmm_scatter.png")


def plot_bic(bic_scores: pd.DataFrame, output_dir: Path) -> Path | None:
    if bic_scores.empty:
        return None

    plt.figure(figsize=(9, 5))
    plt.plot(bic_scores["n_components"], bic_scores["bic"], marker="o")
    plt.title("Selección de componentes GMM con BIC")
    plt.xlabel("Número de componentes")
    plt.ylabel("BIC")
    plt.grid(True, alpha=0.3)
    return save_fig(output_dir / "gmm_bic_selection.png")


def plot_kmeans_reference(elbow_scores: pd.DataFrame, output_dir: Path) -> list[Path]:
    paths = []

    if elbow_scores.empty:
        return paths

    plt.figure(figsize=(9, 5))
    plt.plot(elbow_scores["n_clusters"], elbow_scores["inertia"], marker="o")
    plt.title("Referencia K-Means: método del codo")
    plt.xlabel("Número de clusters")
    plt.ylabel("Inercia")
    plt.grid(True, alpha=0.3)
    paths.append(save_fig(output_dir / "kmeans_elbow_reference.png"))

    if "silhouette_score" in elbow_scores.columns:
        plt.figure(figsize=(9, 5))
        plt.plot(elbow_scores["n_clusters"], elbow_scores["silhouette_score"], marker="o")
        plt.title("Referencia K-Means: silhouette")
        plt.xlabel("Número de clusters")
        plt.ylabel("Silhouette")
        plt.grid(True, alpha=0.3)
        paths.append(save_fig(output_dir / "kmeans_silhouette_reference.png"))

    return paths


def plot_probability_heatmap(result: pd.DataFrame, n_components: int, output_dir: Path) -> Path:
    prob_cols = [f"prob_cluster_{i}" for i in range(n_components)]
    matrix = result[prob_cols].copy()

    labels = result["scenario_name"].astype(str) + " | " + result["simulation_date"].astype(str)
    matrix.index = labels

    max_rows = 80
    if len(matrix) > max_rows:
        matrix = matrix.head(max_rows)

    fig_height = max(6, len(matrix) * 0.25)
    plt.figure(figsize=(10, fig_height))
    plt.imshow(matrix.values, aspect="auto")
    plt.colorbar(label="Probabilidad")
    plt.xticks(range(len(prob_cols)), prob_cols, rotation=45)
    plt.yticks(range(len(matrix)), matrix.index, fontsize=7)
    plt.title("Mapa de calor de probabilidades GMM")
    return save_fig(output_dir / "gmm_probability_heatmap.png")


def plot_cluster_profile(cluster_summary: pd.DataFrame, output_dir: Path) -> Path:
    features = [
        "avg_total_size_mb",
        "avg_file_count",
        "avg_transfer_speed_mbps",
        "avg_storage_cost",
        "avg_queue_pressure",
    ]

    profile = cluster_summary[["gmm_cluster", *features]].copy()

    for col in features:
        min_v = profile[col].min()
        max_v = profile[col].max()
        if max_v == min_v:
            profile[col] = 0.5
        else:
            profile[col] = (profile[col] - min_v) / (max_v - min_v)

    labels = ["Tamaño", "Archivos", "Velocidad", "Costo", "Presión"]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure(figsize=(9, 8))
    ax = plt.subplot(111, polar=True)

    for _, row in profile.iterrows():
        values = [row[col] for col in features]
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=f"Cluster {int(row['gmm_cluster'])}")
        ax.fill(angles, values, alpha=0.08)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    ax.set_title("Perfil normalizado por cluster GMM", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))

    return save_fig(output_dir / "gmm_cluster_profile_radar.png")


def plot_covariance_ellipses(X_pca: np.ndarray, result: pd.DataFrame, gmm: GaussianMixture, output_dir: Path) -> Path:
    gmm_pca = GaussianMixture(n_components=gmm.n_components, covariance_type="full", random_state=42)
    gmm_pca.fit(X_pca)

    plt.figure(figsize=(11, 7))
    ax = plt.gca()

    for cluster in sorted(result["gmm_cluster"].unique()):
        subset = result[result["gmm_cluster"] == cluster]
        idx = subset.index.to_numpy()
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], s=90, alpha=0.7, label=f"Cluster {cluster}")

    for i in range(gmm_pca.n_components):
        mean = gmm_pca.means_[i]
        cov = gmm_pca.covariances_[i]

        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
        width, height = 4 * np.sqrt(eigvals)

        ellipse = Ellipse(mean, width, height, angle=angle, fill=False, linewidth=2)
        ax.add_patch(ellipse)

    plt.title("PCA con elipses gaussianas GMM")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)

    return save_fig(output_dir / "pca_gmm_gaussian_ellipses.png")


def plot_anomalies(result: pd.DataFrame, output_dir: Path) -> Path:
    plt.figure(figsize=(10, 6))

    normal = result[~result["is_anomaly"]]
    anomalous = result[result["is_anomaly"]]

    plt.scatter(normal["total_size_mb"], normal["total_storage_cost"], alpha=0.65, s=90, label="Normal")
    if not anomalous.empty:
        plt.scatter(anomalous["total_size_mb"], anomalous["total_storage_cost"], s=180, facecolors="none", edgecolors="black", linewidths=1.8, label="Anomalía")

    plt.title("Anomalías según baja probabilidad GMM")
    plt.xlabel("Tamaño total diario MB")
    plt.ylabel("Costo total diario")
    plt.grid(True, alpha=0.3)
    plt.legend()

    return save_fig(output_dir / "gmm_anomalies.png")


def html_table(df: pd.DataFrame, title: str, description: str = "", max_rows: int = 80) -> str:
    if df is None or df.empty:
        return f"<section class='card'><h2>{title}</h2><p class='muted'>No disponible.</p></section>"

    data = df.copy()

    for col in data.select_dtypes(include="number").columns:
        data[col] = data[col].round(5)

    if len(data) > max_rows:
        data = data.head(max_rows)
        description += f" Se muestran las primeras {max_rows} filas."

    return f"""
    <section class="card">
        <h2>{title}</h2>
        <p class="section-desc">{description}</p>
        <div class="table-container">
            {data.to_html(index=False, border=0, classes="data-table")}
        </div>
    </section>
    """


def plot_grid(plot_files: list[Path], html_dir: Path) -> str:
    if not plot_files:
        return "<section class='card'><h2>Gráficos</h2><p class='muted'>No se generaron gráficos.</p></section>"

    items = ""

    for path in plot_files:
        if path is None:
            continue
        rel = safe_rel(path, html_dir)
        title = Path(path).stem.replace("_", " ").title()
        items += f"""
        <figure class="plot-card">
            <img src="{rel}" alt="{title}">
            <figcaption>{title}</figcaption>
        </figure>
        """

    return f"""
    <section class="card">
        <h2>Gráficos</h2>
        <div class="plot-grid">{items}</div>
    </section>
    """


def metric_card(label: str, value, hint: str = "") -> str:
    return f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{fmt(value)}</div>
        <div class="metric-hint">{hint}</div>
    </div>
    """


def html_page(title: str, body: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
:root {{
  --bg:#f3f6fa; --card:#fff; --ink:#17212b; --muted:#667085;
  --primary:#17324d; --accent:#2c7be5; --border:#e5edf3;
}}
body {{ font-family: Arial, sans-serif; margin:0; background:var(--bg); color:var(--ink); }}
.page {{ max-width:1360px; margin:0 auto; padding:32px; }}
.hero {{ background:linear-gradient(135deg,#17324d,#245a89); color:white; padding:28px; border-radius:18px; margin-bottom:24px; }}
.hero h1 {{ margin:0 0 8px 0; font-size:30px; }}
.hero p {{ margin:0; opacity:.92; }}
.card {{ background:white; padding:22px; margin-bottom:22px; border-radius:16px; border:1px solid var(--border); box-shadow:0 2px 10px rgba(0,0,0,.05); }}
.note {{ background:#eef6ff; border-left:6px solid var(--accent); padding:16px 18px; border-radius:12px; margin-bottom:22px; line-height:1.5; }}
h2 {{ color:var(--primary); margin-top:0; border-bottom:1px solid var(--border); padding-bottom:9px; }}
.section-desc,.muted {{ color:var(--muted); font-size:14px; }}
.metrics-grid {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; margin-bottom:22px; }}
.metric-card {{ background:white; border:1px solid var(--border); border-radius:14px; padding:18px; box-shadow:0 1px 6px rgba(0,0,0,.05); }}
.metric-label {{ font-size:13px; color:var(--muted); margin-bottom:8px; }}
.metric-value {{ font-size:25px; font-weight:700; color:var(--primary); }}
.metric-hint {{ font-size:12px; color:var(--muted); margin-top:8px; }}
.table-container {{ overflow-x:auto; }}
table.data-table, table.dataframe {{ border-collapse:collapse; width:100%; font-size:13px; }}
table.data-table th, table.dataframe th {{ background-color:var(--primary); color:white; padding:8px; text-align:left; }}
table.data-table td, table.dataframe td {{ border:1px solid #dde5ed; padding:7px; vertical-align:top; }}
table.data-table tr:nth-child(even), table.dataframe tr:nth-child(even) {{ background-color:#f8fafc; }}
.plot-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:20px; }}
.plot-card {{ margin:0; background:#fbfdff; border:1px solid var(--border); border-radius:14px; padding:12px; }}
.plot-card img {{ width:100%; max-width:100%; display:block; border-radius:10px; border:1px solid #d9e2ec; background:white; }}
.plot-card figcaption {{ margin-top:8px; font-size:13px; color:var(--muted); text-align:center; }}
</style>
</head>
<body>
<main class="page">
<section class="hero">
<h1>{title}</h1>
<p>PCA para estructura latente y GMM para composición probabilística del simulador.</p>
</section>
{body}
</main>
</body>
</html>"""


def build_scope_findings(
    scope_name: str,
    result: pd.DataFrame,
    pca: PCA,
    best_k: int,
    cluster_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    rows.append({
        "hallazgo": "varianza_pc1",
        "valor": pca.explained_variance_ratio_[0],
        "lectura": "PC1 representa la dirección dominante de variabilidad operacional.",
    })

    rows.append({
        "hallazgo": "varianza_pc1_pc2",
        "valor": pca.explained_variance_ratio_.sum(),
        "lectura": "Varianza acumulada explicada por las dos primeras componentes.",
    })

    rows.append({
        "hallazgo": "componentes_gmm",
        "valor": best_k,
        "lectura": "Número de componentes gaussianos seleccionado por BIC.",
    })

    rows.append({
        "hallazgo": "anomalias",
        "valor": int(result["is_anomaly"].sum()),
        "lectura": "Observaciones diarias con baja probabilidad bajo la mezcla GMM.",
    })

    rows.append({
        "hallazgo": "baja_confianza",
        "valor": int(result["is_low_confidence"].sum()),
        "lectura": "Días con pertenencia ambigua a componentes gaussianos.",
    })

    if not cluster_summary.empty:
        top_cluster = cluster_summary.sort_values("observations", ascending=False).iloc[0]
        rows.append({
            "hallazgo": "cluster_dominante",
            "valor": int(top_cluster["gmm_cluster"]),
            "lectura": f"Cluster con más observaciones: {top_cluster.get('interpretation', '')}",
        })

    return pd.DataFrame(rows)


def build_html_report(
    scope_name: str,
    result: pd.DataFrame,
    daily_metrics: pd.DataFrame,
    pca: PCA,
    loadings: pd.DataFrame,
    bic_scores: pd.DataFrame,
    elbow_scores: pd.DataFrame,
    cluster_summary: pd.DataFrame,
    findings: pd.DataFrame,
    plot_files: list[Path],
    output_dir: Path,
) -> Path:
    html_dir = ensure_dir(output_dir / "html")

    cards = ""
    cards += metric_card("Observaciones", len(result), "Días/escenario-día")
    cards += metric_card("PC1+PC2", pca.explained_variance_ratio_.sum(), "Varianza explicada")
    cards += metric_card("Clusters GMM", int(result["gmm_cluster"].nunique()), "Componentes encontrados")
    cards += metric_card("Anomalías", int(result["is_anomaly"].sum()), "Baja probabilidad")

    feature_explanation_df = pd.DataFrame([
        {"feature": key, "explicacion": value}
        for key, value in FEATURE_EXPLANATIONS.items()
    ])

    body = f"""
    <section class="note">
        <strong>Alcance:</strong> {scope_name}<br><br>
        Este análisis aplica PCA antes de GMM. PCA valida si existe estructura latente
        y GMM modela la composición probabilística como una mezcla de comportamientos gaussianos.
    </section>

    <section class="metrics-grid">
        {cards}
    </section>
    """

    body += html_table(findings, "1. Hallazgos principales", "Lectura ejecutiva del análisis PCA/GMM.")
    body += html_table(feature_explanation_df, "2. Variables utilizadas", "Variables agregadas diariamente antes del escalamiento.")
    body += html_table(loadings.sort_values("abs_PC1", ascending=False), "3. Cargas PCA", "Variables que más explican PC1 y PC2.")
    body += html_table(cluster_summary, "4. Perfil de clusters GMM", "Interpretación operacional de cada componente gaussiano.")
    body += html_table(bic_scores, "5. Selección GMM por BIC/AIC", "Menor BIC indica mejor equilibrio entre ajuste y complejidad.")
    body += html_table(elbow_scores, "6. Referencia K-Means", "Usado solo como referencia explicativa, no como clustering final.")
    body += plot_grid(plot_files, html_dir)
    body += html_table(result.sort_values("gmm_log_likelihood").head(15), "7. Observaciones más anómalas", "Menor log-likelihood bajo GMM.")

    report_path = html_dir / f"report_pca_gmm_{scope_name}.html"
    report_path.write_text(
        html_page(f"Reporte PCA + GMM - {scope_name}", body),
        encoding="utf-8",
    )

    return report_path


def run_scope(
    daily_metrics: pd.DataFrame,
    scope_name: str,
    output_root: Path,
    min_components: int,
    max_components: int,
    fixed_components: int | None = None,
) -> dict:
    if len(daily_metrics) < 2:
        raise ValueError(f"El alcance {scope_name} tiene menos de 2 observaciones.")

    daily_metrics = daily_metrics.reset_index(drop=True)

    scope_dir = ensure_dir(output_root / scope_name)
    table_dir = ensure_dir(scope_dir / "tables")
    plot_dir = ensure_dir(scope_dir / "plots")

    pca, X_pca, loadings, scaler = run_pca(daily_metrics)

    elbow_scores = calculate_kmeans_reference(daily_metrics, min_components, max_components)

    if fixed_components is None:
        best_k, bic_scores = select_gmm_components(daily_metrics, min_components, max_components)
    else:
        best_k = max(1, min(fixed_components, len(daily_metrics) - 1))
        _, bic_scores = select_gmm_components(daily_metrics, best_k, best_k)

    result, gmm, X_scaled, scaler = apply_gmm(daily_metrics, best_k)

    cluster_summary = build_cluster_summary(result)
    cluster_summary = add_cluster_interpretation(result, cluster_summary)

    findings = build_scope_findings(scope_name, result, pca, best_k, cluster_summary)

    plot_files = []
    plot_files.append(plot_pca_scatter(result, X_pca, pca, plot_dir, f"PCA + GMM - {scope_name}"))
    plot_files.append(plot_bic(bic_scores, plot_dir))
    plot_files.extend(plot_kmeans_reference(elbow_scores, plot_dir))
    plot_files.append(plot_probability_heatmap(result, best_k, plot_dir))
    plot_files.append(plot_cluster_profile(cluster_summary, plot_dir))
    plot_files.append(plot_covariance_ellipses(X_pca, result, gmm, plot_dir))
    plot_files.append(plot_anomalies(result, plot_dir))
    plot_files = [p for p in plot_files if p is not None]

    save_table(daily_metrics, table_dir / "daily_metrics.csv")
    save_table(result, table_dir / "pca_gmm_assignments.csv")
    save_table(loadings, table_dir / "pca_loadings.csv")
    save_table(bic_scores, table_dir / "gmm_bic_aic_scores.csv")
    save_table(elbow_scores, table_dir / "kmeans_reference_scores.csv")
    save_table(cluster_summary, table_dir / "gmm_cluster_summary.csv")
    save_table(findings, table_dir / "pca_gmm_findings.csv")

    report_path = build_html_report(
        scope_name=scope_name,
        result=result,
        daily_metrics=daily_metrics,
        pca=pca,
        loadings=loadings,
        bic_scores=bic_scores,
        elbow_scores=elbow_scores,
        cluster_summary=cluster_summary,
        findings=findings,
        plot_files=plot_files,
        output_dir=scope_dir,
    )

    return {
        "scope": scope_name,
        "observations": len(result),
        "scenarios": int(result["scenario_name"].nunique()),
        "best_gmm_components": int(best_k),
        "pca_explained_variance_pc1": float(pca.explained_variance_ratio_[0]),
        "pca_explained_variance_pc1_pc2": float(pca.explained_variance_ratio_.sum()),
        "anomalies": int(result["is_anomaly"].sum()),
        "low_confidence": int(result["is_low_confidence"].sum()),
        "report": str(report_path),
        "tables_dir": str(table_dir),
        "plots_dir": str(plot_dir),
    }


def build_master_index(results: list[dict], output_root: Path) -> Path:
    html_dir = ensure_dir(output_root / "html")
    df = pd.DataFrame(results)

    if df.empty:
        body = "<section class='card'><h2>Sin resultados</h2></section>"
    else:
        links = df.copy()
        links["report_link"] = links["report"].apply(
            lambda p: f"<a href='{safe_rel(p, html_dir)}'>Abrir reporte</a>"
        )

        body = """
        <section class="note">
            Índice general de caracterización probabilística.
            Incluye análisis global y análisis por escenario cuando están disponibles.
        </section>
        """
        body += html_table(links, "Índice de reportes PCA/GMM", "Resumen de alcances generados.", max_rows=200)

    html = html_page("Índice general PCA + GMM", body)
    html = html.replace("&lt;a href=", "<a href=").replace("&lt;/a&gt;", "</a>").replace("&gt;Abrir reporte", ">Abrir reporte")

    index_path = html_dir / "index_pca_gmm_reports.html"
    index_path.write_text(html, encoding="utf-8")

    return index_path


def main():
    parser = argparse.ArgumentParser(
        description="Caracterización probabilística del simulador usando PCA + GMM."
    )

    parser.add_argument("--dataset-dir", default="output/dataset")
    parser.add_argument("--output-dir", default="output/probabilistic_characterization")
    parser.add_argument("--mode", choices=["global", "scenarios", "all"], default="all")
    parser.add_argument("--scenario-filter", default=None)
    parser.add_argument("--min-components", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=6)
    parser.add_argument("--n-components", type=int, default=None)

    args = parser.parse_args()

    output_root = ensure_dir(args.output_dir)

    raw = load_datasets(args.dataset_dir)
    clean = clean_data(raw)
    daily = build_daily_metrics(clean)

    save_table(daily, output_root / "tables" / "daily_metrics_all.csv")

    results = []

    if args.mode in {"global", "all"}:
        print("[PCA/GMM] Ejecutando análisis global...")
        results.append(
            run_scope(
                daily_metrics=daily,
                scope_name="global",
                output_root=output_root,
                min_components=args.min_components,
                max_components=args.max_components,
                fixed_components=args.n_components,
            )
        )

    if args.mode in {"scenarios", "all"}:
        print("[PCA/GMM] Ejecutando análisis por escenario...")

        scenario_names = sorted(daily["scenario_name"].unique())

        if args.scenario_filter:
            scenario_names = [
                s for s in scenario_names
                if args.scenario_filter.lower() in s.lower()
            ]

        for scenario in scenario_names:
            scenario_daily = daily[daily["scenario_name"] == scenario].copy()

            if len(scenario_daily) < 2:
                print(f"[WARN] Escenario omitido por pocas observaciones: {scenario}")
                continue

            print(f"[PCA/GMM] Escenario: {scenario}")

            results.append(
                run_scope(
                    daily_metrics=scenario_daily,
                    scope_name=f"scenario_{scenario}",
                    output_root=output_root,
                    min_components=args.min_components,
                    max_components=args.max_components,
                    fixed_components=args.n_components,
                )
            )

    results_df = pd.DataFrame(results)
    save_table(results_df, output_root / "tables" / "pca_gmm_scope_summary.csv")

    index_path = build_master_index(results, output_root)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_dir": str(args.dataset_dir),
        "output_dir": str(output_root),
        "mode": args.mode,
        "total_scopes": len(results),
        "index_report": str(index_path),
        "results": results,
    }

    write_json(output_root / "manifest_pca_gmm_characterization.json", manifest)

    print("[OK] Caracterización PCA/GMM completada")
    print(f"[OK] Índice: {index_path}")
    print(f"[OK] Manifest: {output_root / 'manifest_pca_gmm_characterization.json'}")


if __name__ == "__main__":
    main()
