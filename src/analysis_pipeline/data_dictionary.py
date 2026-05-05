import pandas as pd


def get_data_dictionary() -> dict[str, str]:
    return {
        "file_id": "Identificador único del archivo",
        "run_id": "Identificador de la corrida de simulación",
        "simulation_date": "Fecha lógica de simulación",
        "simulation_datetime": "Fecha-hora lógica reconstruida desde simulation_date y created_hour",
        "weekday_base_load": "Peso de carga semanal asociado al día",
        "source_file": "Archivo CSV origen del registro",
        "sequence": "Número secuencial del registro",
        "case_type": "Tipo de caso asociado",
        "case_group": "Grupo de casos",
        "file_type": "Tipo de archivo",
        "size_mb": "Tamaño del archivo en MB",
        "size_range": "Rango de tamaño calculado desde size_mb",
        "storage_tier": "Nivel de almacenamiento",
        "days_stored": "Días almacenado",
        "days_since_last_access": "Días desde último acceso",
        "movement_storage": "Indicador de movimiento de almacenamiento",
        "transfer_duration_sec": "Duración de transferencia en segundos",
        "transfer_speed_mbps": "Velocidad de transferencia",
        "day_of_week": "Día de la semana",
        "time_slot": "Franja horaria",
        "created_at": "Fecha de creación",
        "created_hour": "Hora de creación",
        "created_date": "Fecha derivada desde created_at",
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


def build_data_dictionary_table(df: pd.DataFrame) -> pd.DataFrame:
    dictionary = get_data_dictionary()
    rows = []

    for col in df.columns:
        rows.append({
            "variable": col,
            "description": dictionary.get(col, "Sin descripción"),
            "dtype": str(df[col].dtype),
            "non_null_count": int(df[col].notna().sum()),
            "null_count": int(df[col].isna().sum()),
            "unique_values": int(df[col].nunique(dropna=True)),
        })

    return pd.DataFrame(rows)
