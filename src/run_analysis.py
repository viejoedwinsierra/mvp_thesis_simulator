import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from analysis.descriptive_analysis import DescriptiveAnalysis
from analysis.advanced_analysis import AdvancedAnalysis
from analysis.export_descriptive import DescriptiveExporter

# ======================================================
# 📥 1. CARGA DATASET
# ======================================================

df = pd.read_csv("output/dataset/blob_inventory.csv")

print("\n=== COLUMNAS DEL DATASET ===")
print(df.columns.tolist())

# ======================================================
# 🧾 2. DICCIONARIO DE VARIABLES (TESIS)
# ======================================================

data_dictionary = {
    "file_id": "Identificador único del archivo",
    "sequence": "Número secuencial del registro",
    "case_type": "Tipo de caso asociado",
    "case_group": "Grupo de casos",
    "file_type": "Tipo de archivo",
    "size_mb": "Tamaño del archivo en MB",
    "storage_tier": "Nivel de almacenamiento",
    "days_stored": "Días almacenado",
    "days_since_last_access": "Días desde último acceso",
    "read_level": "Nivel de lectura (actividad)",
    "modify_level": "Nivel de modificación",
    "movement_storage": "Cantidad de movimientos de almacenamiento",
    "transfer_duration_sec": "Duración de transferencia en segundos",
    "transfer_speed_mbps": "Velocidad de transferencia",
    "day_of_week": "Día de la semana",
    "time_slot": "Franja horaria",
    "created_at": "Fecha de creación",
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
    print(f"{k}: {v}")

# ======================================================
# ⚙️ 3. LIMPIEZA Y TIPOS
# ======================================================

numeric_columns = [
    "sequence",
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

# ======================================================
# 📊 4. ANÁLISIS DESCRIPTIVO
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
# 📊 5. TABLAS CLAVE
# ======================================================

print("\n=== TOP COSTOS ===")
print(df.sort_values("storage_cost", ascending=False).head(10))

print("\n=== COSTO PROMEDIO POR STORAGE TIER ===")
print(df.groupby("storage_tier")["storage_cost"].mean())

print("\n=== ERRORES ===")
print(df["has_error"].value_counts())

# ======================================================
# 📈 6. GRÁFICOS
# ======================================================

# ---- Histograma tamaño ----
plt.figure()
df["size_mb"].hist(bins=50)
plt.title("Distribución de tamaño (MB)")
plt.xlabel("size_mb")
plt.ylabel("Frecuencia")
plt.show()

# ---- Boxplot ----
plt.figure()
sns.boxplot(x=df["size_mb"])
plt.title("Outliers en size_mb")
plt.show()

# ---- Costos ----
plt.figure()
df["storage_cost"].hist(bins=50)
plt.title("Distribución de costos")
plt.show()

# ---- Correlación ----
numeric_df = df.select_dtypes(include="number")

plt.figure(figsize=(10, 8))
sns.heatmap(numeric_df.corr(), cmap="coolwarm")
plt.title("Matriz de correlación")
plt.show()

# ======================================================
# 💾 7. EXPORTAR RESULTADOS (OPCIONAL)
# ======================================================

results["nulls"].to_csv("output/analysis_nulls.csv")
results["distribution"].to_csv("output/analysis_distribution.csv")
results["outliers"].to_csv("output/analysis_outliers.csv")

print("\n✅ Análisis completado")

os.makedirs("output/plots", exist_ok=True)
os.makedirs("output/descriptive_exports", exist_ok=True)
advanced = AdvancedAnalysis(df)
advanced.run_all()

exporter = DescriptiveExporter(df)
exporter.run_all()