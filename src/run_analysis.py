from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from analysis.descriptive_analysis import DescriptiveAnalysis
from analysis.advanced_analysis import AdvancedAnalysis
from analysis.export_descriptive import DescriptiveExporter


# ======================================================
# 1. CARGA DATASETS DESDE CARPETA
# ======================================================

def load_datasets_from_folder(dataset_dir: str = "output/dataset") -> pd.DataFrame:
    dataset_path = Path(dataset_dir)

    files = sorted(dataset_path.glob("blob_inventory*.csv"))

    # Evita duplicar el último archivo compatible
    files = [
        file for file in files
        if file.name != "blob_inventory.csv"
    ]

    if not files:
        raise FileNotFoundError(
            f"No se encontraron archivos blob_inventory*.csv en {dataset_path}"
        )

    frames = []

    for file in files:
        df_part = pd.read_csv(file)
        df_part["source_file"] = file.name
        frames.append(df_part)

    df_all = pd.concat(frames, ignore_index=True)

    return df_all


df = load_datasets_from_folder("output/dataset")

print("\n=== DATASETS CARGADOS ===")
print(df["source_file"].nunique())

print("\n=== FILAS TOTALES ===")
print(len(df))

print("\n=== COLUMNAS DEL DATASET ===")
print(df.columns.tolist())


# ======================================================
# 2. DICCIONARIO DE VARIABLES
# ======================================================

data_dictionary = {
    "file_id": "Identificador único del archivo",
    "run_id": "Identificador de la corrida de simulación",
    "simulation_date": "Fecha lógica de simulación",
    "weekday_base_load": "Peso de carga semanal asociado al día",
    "source_file": "Archivo CSV origen del registro",
    "sequence": "Número secuencial del registro",
    "case_type": "Tipo de caso asociado",
    "case_group": "Grupo de casos",
    "file_type": "Tipo de archivo",
    "size_mb": "Tamaño del archivo en MB",
    "storage_tier": "Nivel de almacenamiento",
    "days_stored": "Días almacenado",
    "days_since_last_access": "Días desde último acceso",
    "movement_storage": "Indicador de movimiento de almacenamiento",
    "transfer_duration_sec": "Duración de transferencia en segundos",
    "transfer_speed_mbps": "Velocidad de transferencia",
    "day_of_week": "Día de la semana",
    "time_slot": "Franja horaria",
    "created_at": "Fecha de creación",
    "content_hash": "Hash completo del contenido si está habilitado",
    "hash_head": "Hash inicial",
    "hash_tail": "Hash final",
    "error_duplicado": "Error por duplicado",
    "error_orphan": "Error de archivo huérfano",
    "error_null": "Error por datos nulos",
    "error_blob_timeout": "Error por timeout",
    "has_error": "Indicador de error",
    "is_duplicate": "Indicador de duplicado",
    "severity": "Severidad del error",
    "storage_cost": "Costo de almacenamiento",
}

print("\n=== DICCIONARIO DE VARIABLES ===")
for k, v in data_dictionary.items():
    if k in df.columns:
        print(f"{k}: {v}")


# ======================================================
# 3. LIMPIEZA Y TIPOS
# ======================================================

numeric_columns = [
    "sequence",
    "weekday_base_load",
    "size_mb",
    "days_stored",
    "days_since_last_access",
    "movement_storage",
    "transfer_duration_sec",
    "transfer_speed_mbps",
    "error_duplicado",
    "error_orphan",
    "error_null",
    "error_blob_timeout",
    "has_error",
    "is_duplicate",
    "severity",
    "storage_cost",
]

for col in numeric_columns:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

if "simulation_date" in df.columns:
    df["simulation_date"] = pd.to_datetime(df["simulation_date"], errors="coerce")

if "created_at" in df.columns:
    df["created_hour"] = df["created_at"].dt.hour
    df["created_date"] = df["created_at"].dt.date


# ======================================================
# 4. ANÁLISIS DESCRIPTIVO
# ======================================================

analysis = DescriptiveAnalysis(df)
results = analysis.run()

print("\n=== INFO GENERAL ===")
print(results["general_info"])

print("\n=== NULOS ===")
print(results["nulls"])

print("\n=== DESCRIPTIVO ===")
print(results["descriptive_stats"])

print("\n=== DISTRIBUCIÓN ===")
print(results["distribution"])

print("\n=== OUTLIERS ===")
print(results["outliers"])


# ======================================================
# 5. TABLAS CLAVE
# ======================================================

print("\n=== TOP COSTOS ===")
print(df.sort_values("storage_cost", ascending=False).head(10))

print("\n=== COSTO PROMEDIO POR STORAGE TIER ===")
print(df.groupby("storage_tier")["storage_cost"].mean())

print("\n=== ERRORES ===")
print(df["has_error"].value_counts())

if "simulation_date" in df.columns:
    print("\n=== RESUMEN POR FECHA ===")
    daily_summary = (
        df.groupby(df["simulation_date"].dt.date)
        .agg(
            records=("file_id", "count"),
            total_cost=("storage_cost", "sum"),
            avg_cost=("storage_cost", "mean"),
            error_rate=("has_error", "mean"),
            avg_size_mb=("size_mb", "mean"),
            avg_transfer_duration=("transfer_duration_sec", "mean"),
        )
        .reset_index()
    )

    print(daily_summary)


if "day_of_week" in df.columns:
    print("\n=== RESUMEN POR DÍA DE SEMANA ===")
    weekday_summary = (
        df.groupby("day_of_week")
        .agg(
            records=("file_id", "count"),
            total_cost=("storage_cost", "sum"),
            avg_cost=("storage_cost", "mean"),
            error_rate=("has_error", "mean"),
            avg_size_mb=("size_mb", "mean"),
        )
        .reset_index()
    )

    print(weekday_summary)


# ======================================================
# 6. EXPORTAR RESULTADOS
# ======================================================

os.makedirs("output", exist_ok=True)
os.makedirs("output/plots", exist_ok=True)
os.makedirs("output/plots/descriptive", exist_ok=True)
os.makedirs("output/descriptive_exports", exist_ok=True)

results["nulls"].to_csv("output/analysis_nulls.csv", index=False)
results["distribution"].to_csv("output/analysis_distribution.csv", index=False)
results["outliers"].to_csv("output/analysis_outliers.csv", index=False)

if "simulation_date" in df.columns:
    daily_summary.to_csv("output/analysis_daily_summary.csv", index=False)

if "day_of_week" in df.columns:
    weekday_summary.to_csv("output/analysis_weekday_summary.csv", index=False)


# ======================================================
# 7. GRÁFICOS
# ======================================================

plt.figure()
df["size_mb"].hist(bins=50)
plt.title("Distribución de tamaño (MB)")
plt.xlabel("size_mb")
plt.ylabel("Frecuencia")
plt.savefig("output/plots/descriptive/size_mb_histogram.png", bbox_inches="tight")
plt.close()

plt.figure()
sns.boxplot(x=df["size_mb"])
plt.title("Outliers en size_mb")
plt.savefig("output/plots/descriptive/size_mb_boxplot.png", bbox_inches="tight")
plt.close()

plt.figure()
df["storage_cost"].hist(bins=50)
plt.title("Distribución de costos")
plt.xlabel("storage_cost")
plt.ylabel("Frecuencia")
plt.savefig("output/plots/descriptive/storage_cost_histogram.png", bbox_inches="tight")
plt.close()

numeric_df = df.select_dtypes(include="number")

if "content_hash" in numeric_df.columns:
    numeric_df = numeric_df.drop(columns=["content_hash"])

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm")
plt.title("Matriz de correlación")
plt.savefig("output/plots/descriptive/correlation_matrix.png", bbox_inches="tight")
plt.close()

if "simulation_date" in df.columns:
    plt.figure(figsize=(12, 5))
    daily_summary.plot(
        x="simulation_date",
        y="records",
        kind="line",
        marker="o",
        legend=False,
    )
    plt.title("Registros por fecha de simulación")
    plt.xlabel("simulation_date")
    plt.ylabel("records")
    plt.savefig("output/plots/descriptive/daily_records.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 5))
    daily_summary.plot(
        x="simulation_date",
        y="error_rate",
        kind="line",
        marker="o",
        legend=False,
    )
    plt.title("Tasa de error por fecha")
    plt.xlabel("simulation_date")
    plt.ylabel("error_rate")
    plt.savefig("output/plots/descriptive/daily_error_rate.png", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 5))
    daily_summary.plot(
        x="simulation_date",
        y="total_cost",
        kind="line",
        marker="o",
        legend=False,
    )
    plt.title("Costo total por fecha")
    plt.xlabel("simulation_date")
    plt.ylabel("total_cost")
    plt.savefig("output/plots/descriptive/daily_total_cost.png", bbox_inches="tight")
    plt.close()


# ======================================================
# 8. ANÁLISIS AVANZADO Y EXPORTS
# ======================================================

advanced = AdvancedAnalysis(df)
advanced.run_all()

exporter = DescriptiveExporter(df)
exporter.run_all()

print("\n✅ Análisis completado con carga multi-dataset desde output/dataset")