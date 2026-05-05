import pandas as pd


NUMERIC_COLUMNS = [
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
    "hourly_arrival_count",
    "hourly_capacity",
    "queue_pressure",
    "congestion_factor",
]


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        df["created_hour"] = df["created_at"].dt.hour
        df["created_date"] = df["created_at"].dt.date

    if "simulation_date" in df.columns:
        df["simulation_date"] = pd.to_datetime(df["simulation_date"], errors="coerce")

    if "simulation_date" in df.columns and "created_hour" in df.columns:
        df["simulation_datetime"] = (
            pd.to_datetime(df["simulation_date"], errors="coerce")
            + pd.to_timedelta(df["created_hour"].fillna(0), unit="h")
        )

    if "size_mb" in df.columns:
        bins = [0, 1, 10, 100, 1000, float("inf")]
        labels = ["tiny", "small", "medium", "large", "xlarge"]
        df["size_range"] = pd.cut(
            df["size_mb"],
            bins=bins,
            labels=labels,
            include_lowest=True,
        )

    if "created_hour" in df.columns and "time_slot" not in df.columns:
        df["time_slot"] = pd.cut(
            df["created_hour"],
            bins=[-1, 6, 12, 18, 23],
            labels=["night", "morning", "afternoon", "evening"],
        )

    return df
