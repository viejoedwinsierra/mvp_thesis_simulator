import numpy as np
import pandas as pd


NUMERIC_VARS = [
    "size_mb",
    "days_stored",
    "days_since_last_access",
    "transfer_duration_sec",
    "transfer_speed_mbps",
    "created_hour",
    "storage_cost",
    "hourly_arrival_count",
    "hourly_capacity",
    "queue_pressure",
    "congestion_factor",
]

CATEGORICAL_VARS = [
    "file_type",
    "storage_tier",
    "has_error",
    "severity",
]


def classify_skewness(skewness: float) -> str:
    if pd.isna(skewness):
        return "NA"
    if abs(skewness) < 0.5:
        return "simetrica"
    if abs(skewness) < 1:
        return "moderada"
    return "alta"


def classify_outlier_level(outlier_pct: float) -> str:
    if pd.isna(outlier_pct):
        return "NA"
    if outlier_pct < 1:
        return "bajo"
    if outlier_pct < 5:
        return "moderado"
    return "alto"


def classify_cardinality(n_categories: int) -> str:
    if n_categories <= 5:
        return "baja"
    if n_categories <= 20:
        return "media"
    return "alta"


def classify_distribution(skewness: float, kurtosis: float) -> str:
    if pd.isna(skewness) or pd.isna(kurtosis):
        return "NA"

    if abs(skewness) < 0.5 and abs(kurtosis) < 3:
        return "aprox_normal"

    if skewness > 1 and kurtosis > 10:
        return "heavy_tail"

    if skewness > 1:
        return "log_normal_candidate"

    if skewness < -1:
        return "left_skewed"

    return "skewed"


def detect_numeric_variable_type(series: pd.Series) -> str:
    unique_count = series.nunique(dropna=True)

    if unique_count <= 2:
        return "binaria"

    if unique_count <= 25:
        return "discreta"

    return "continua"


def classify_scale(series: pd.Series) -> str:
    median = series.median()

    if pd.isna(median):
        return "NA"

    if abs(median) < 1:
        return "small_scale"

    if abs(median) <= 1000:
        return "medium_scale"

    return "large_scale"


def numeric_modeling_recommendation(
    series: pd.Series,
    skewness: float,
    outlier_pct: float,
) -> str:
    unique_count = series.nunique(dropna=True)

    if unique_count <= 1:
        return "CONSTANT"

    # En variables numéricas continuas, muchos valores únicos NO es alta cardinalidad.
    if abs(skewness) >= 1 or outlier_pct >= 5:
        return "NEEDS_LOG"

    return "OK"


def outlier_decision(outlier_pct: float, skewness: float) -> str:
    # En simulación Monte Carlo, los extremos suelen representar comportamiento válido.
    if outlier_pct >= 20:
        return "REVIEW"

    if outlier_pct >= 5 or abs(skewness) >= 1:
        return "TRANSFORM"

    return "KEEP"


def build_numeric_summary(
    df: pd.DataFrame,
    numeric_vars: list[str] | None = None,
) -> pd.DataFrame:
    if numeric_vars is None:
        numeric_vars = NUMERIC_VARS

    rows = []

    for variable in numeric_vars:
        if variable not in df.columns:
            continue

        series = pd.to_numeric(df[variable], errors="coerce").dropna()

        if series.empty:
            rows.append({
                "variable": variable,
                "variable_type": "NO_DATA",
                "count": 0,
                "mean": np.nan,
                "median": np.nan,
                "std": np.nan,
                "min": np.nan,
                "max": np.nan,
                "p25": np.nan,
                "p75": np.nan,
                "p90": np.nan,
                "p95": np.nan,
                "p99": np.nan,
                "range": np.nan,
                "IQR": np.nan,
                "cv": np.nan,
                "skewness": np.nan,
                "kurtosis": np.nan,
                "tipo_asimetria": "NA",
                "distribution_type": "NA",
                "scale_type": "NA",
                "outlier_pct": np.nan,
                "riesgo_outliers": "NA",
                "modeling_recommendation": "NO_DATA",
            })
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
        outlier_pct = outlier_count / len(series) * 100

        mean = series.mean()
        std = series.std()
        skewness = series.skew()
        kurtosis = series.kurtosis()

        rows.append({
            "variable": variable,
            "variable_type": detect_numeric_variable_type(series),
            "count": len(series),
            "mean": mean,
            "median": series.median(),
            "std": std,
            "min": series.min(),
            "max": series.max(),
            "p25": q1,
            "p75": q3,
            "p90": series.quantile(0.90),
            "p95": series.quantile(0.95),
            "p99": series.quantile(0.99),
            "range": series.max() - series.min(),
            "IQR": iqr,
            "cv": std / mean if mean != 0 else np.nan,
            "skewness": skewness,
            "kurtosis": kurtosis,
            "tipo_asimetria": classify_skewness(skewness),
            "distribution_type": classify_distribution(skewness, kurtosis),
            "scale_type": classify_scale(series),
            "outlier_pct": outlier_pct,
            "riesgo_outliers": classify_outlier_level(outlier_pct),
            "modeling_recommendation": numeric_modeling_recommendation(
                series=series,
                skewness=skewness,
                outlier_pct=outlier_pct,
            ),
        })

    return pd.DataFrame(rows)


def build_outlier_summary(
    df: pd.DataFrame,
    numeric_vars: list[str] | None = None,
) -> pd.DataFrame:
    if numeric_vars is None:
        numeric_vars = NUMERIC_VARS

    rows = []

    for variable in numeric_vars:
        if variable not in df.columns:
            continue

        series = pd.to_numeric(df[variable], errors="coerce").dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()
        outlier_pct = outlier_count / len(series) * 100
        skewness = series.skew()

        rows.append({
            "variable": variable,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outlier_count": outlier_count,
            "outlier_pct": outlier_pct,
            "clasificacion": classify_outlier_level(outlier_pct),
            "decision_modelado": outlier_decision(outlier_pct, skewness),
            "nota": "No eliminar automaticamente; revisar si representa comportamiento real del sistema.",
        })

    return pd.DataFrame(rows)


def build_categorical_summary(
    df: pd.DataFrame,
    categorical_vars: list[str] | None = None,
) -> pd.DataFrame:
    if categorical_vars is None:
        categorical_vars = CATEGORICAL_VARS

    rows = []

    for variable in categorical_vars:
        if variable not in df.columns:
            continue

        series = df[variable].dropna().astype(str)

        if series.empty:
            rows.append({
                "variable": variable,
                "n_categories": 0,
                "most_frequent_value": "NO_DATA",
                "dominant_pct": np.nan,
                "cardinality": "NA",
                "recommendation": "excluir",
            })
            continue

        counts = series.value_counts()
        n_categories = series.nunique()
        most_frequent_value = counts.index[0]
        dominant_pct = counts.iloc[0] / len(series) * 100
        cardinality = classify_cardinality(n_categories)

        if n_categories <= 1:
            recommendation = "excluir"
        elif dominant_pct >= 95:
            recommendation = "excluir"
        elif variable == "severity":
            recommendation = "excluir"
        elif cardinality == "alta":
            recommendation = "agrupar"
        elif cardinality == "baja":
            recommendation = "dummy"
        else:
            recommendation = "usable"

        rows.append({
            "variable": variable,
            "n_categories": n_categories,
            "most_frequent_value": most_frequent_value,
            "dominant_pct": dominant_pct,
            "cardinality": cardinality,
            "recommendation": recommendation,
        })

    return pd.DataFrame(rows)


def build_data_quality_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for variable in df.columns:
        null_count = df[variable].isna().sum()
        null_pct = null_count / len(df) * 100 if len(df) > 0 else np.nan
        unique_count = df[variable].nunique(dropna=True)

        flags = []

        if null_pct >= 30:
            flags.append("HIGH_NULLS")
        elif null_pct >= 5:
            flags.append("MODERATE_NULLS")

        if unique_count <= 1:
            flags.append("CONSTANT")

        if df[variable].dtype == "object" and unique_count > 0.9 * len(df):
            flags.append("HIGH_CARDINALITY")

        if variable in ["file_id", "hash_head", "hash_tail"]:
            flags.append("IDENTIFIER")

        if variable in ["queue_pressure", "congestion_factor", "size_range"]:
            flags.append("DERIVED")

        if "error_" in variable and variable != "has_error":
            flags.append("ERROR_COMPONENT")

        if "severity" == variable:
            flags.append("POTENTIAL_LEAKAGE")

        if "HIGH_NULLS" in flags or "CONSTANT" in flags:
            severity = "CRITICAL"
        elif "MODERATE_NULLS" in flags or "HIGH_CARDINALITY" in flags:
            severity = "WARNING"
        else:
            severity = "OK"

        rows.append({
            "variable": variable,
            "null_count": null_count,
            "null_pct": null_pct,
            "unique_count": unique_count,
            "flags": ", ".join(flags) if flags else "OK",
            "quality_severity": severity,
        })

    return pd.DataFrame(rows)


def build_variable_classification() -> pd.DataFrame:
    rows = [
        ("file_id", "identificador", "no_usar", "NO", "Identificador único; no aporta señal estadística generalizable"),
        ("run_id", "identificador_simulacion", "control", "NO", "Identifica corrida; usar solo para trazabilidad"),
        ("simulation_date", "temporal", "control", "NO", "Fecha lógica; útil para auditoría temporal, no predictor directo en fase univariada"),
        ("source_file", "origen", "control", "NO", "Archivo fuente; usar solo para trazabilidad"),

        ("file_type", "categorica", "predictor", "SI", "Tipo de archivo operacionalmente relevante"),
        ("size_mb", "continua", "predictor", "SI", "Variable principal de tamaño; requiere revisar transformación log"),
        ("storage_tier", "categorica", "predictor", "SI", "Nivel de almacenamiento asociado a costo y acceso"),
        ("days_stored", "continua", "predictor", "SI", "Tiempo de permanencia del blob"),
        ("days_since_last_access", "continua", "predictor", "SI", "Recencia de acceso"),
        ("transfer_duration_sec", "continua", "target", "SI", "Tiempo de transferencia a explicar o simular"),
        ("transfer_speed_mbps", "continua", "predictor", "SI", "Velocidad observada de transferencia"),
        ("created_hour", "discreta", "predictor", "SI", "Hora de creación del blob"),

        ("has_error", "binaria", "target", "SI", "Indicador principal de evento de error"),
        ("severity", "categorica", "no_usar", "NO", "Posible fuga de información posterior al error"),
        ("error_duplicado", "binaria", "no_usar", "NO", "Componente del error; puede inducir fuga si el target es has_error"),
        ("error_orphan", "binaria", "no_usar", "NO", "Componente del error; puede inducir fuga si el target es has_error"),
        ("error_null", "binaria", "no_usar", "NO", "Componente del error; puede inducir fuga si el target es has_error"),
        ("error_blob_timeout", "binaria", "no_usar", "NO", "Componente del error; puede inducir fuga si el target es has_error"),

        ("storage_cost", "continua", "target/predictor", "SI", "Costo operacional individual; requiere revisar transformación log"),
        ("hourly_arrival_count", "discreta", "predictor", "SI", "Carga horaria de llegada"),
        ("hourly_capacity", "discreta", "predictor", "SI", "Capacidad horaria disponible"),

        ("queue_pressure", "derivada", "no_usar", "NO", "Variable derivada; revisar solo en fase avanzada"),
        ("congestion_factor", "derivada", "no_usar", "NO", "Variable derivada; puede inducir dependencia artificial"),
        ("size_range", "derivada", "no_usar", "NO", "Variable derivada desde size_mb; evitar duplicidad"),
        ("content_hash", "tecnica", "no_usar", "NO", "100% nula o sin utilidad predictiva"),
        ("hash_head", "identificador", "no_usar", "NO", "Alta cardinalidad técnica; no generalizable"),
        ("hash_tail", "identificador", "no_usar", "NO", "Alta cardinalidad técnica; no generalizable"),
    ]

    return pd.DataFrame(
        rows,
        columns=["variable", "tipo", "rol_modelo", "usar", "motivo"],
    )


def build_automatic_conclusions(
    numeric_summary: pd.DataFrame,
    categorical_summary: pd.DataFrame,
    data_quality: pd.DataFrame,
) -> list[str]:
    conclusions = []

    if numeric_summary is not None and not numeric_summary.empty:
        for _, row in numeric_summary.iterrows():
            variable = row["variable"]
            skewness = row.get("skewness", np.nan)
            kurtosis = row.get("kurtosis", np.nan)
            outlier_pct = row.get("outlier_pct", np.nan)
            distribution_type = row.get("distribution_type", "NA")
            recommendation = row.get("modeling_recommendation", "OK")

            if recommendation == "NEEDS_LOG":
                conclusions.append(
                    f"{variable} presenta asimetría/outliers relevantes "
                    f"(skew={skewness:.2f}, kurtosis={kurtosis:.2f}, outliers={outlier_pct:.2f}%). "
                    f"Distribución sugerida: {distribution_type}. "
                    "Recomendación: evaluar log1p o transformación robusta."
                )

            elif recommendation == "CONSTANT":
                conclusions.append(
                    f"{variable} es constante. Recomendación: excluir del modelamiento."
                )

        high_outliers = numeric_summary[
            numeric_summary["riesgo_outliers"].eq("alto")
        ]["variable"].tolist()

        if high_outliers:
            conclusions.append(
                "Variables con alto porcentaje de outliers: "
                + ", ".join(high_outliers)
                + ". En simulación Monte Carlo no deben eliminarse automáticamente; primero validar si representan escenarios extremos reales."
            )

    if categorical_summary is not None and not categorical_summary.empty:
        exclude_vars = categorical_summary[
            categorical_summary["recommendation"].eq("excluir")
        ]["variable"].tolist()

        high_card = categorical_summary[
            categorical_summary["cardinality"].eq("alta")
        ]["variable"].tolist()

        if exclude_vars:
            conclusions.append(
                "Variables categóricas sugeridas para exclusión: "
                + ", ".join(exclude_vars)
                + "."
            )

        if high_card:
            conclusions.append(
                "Variables categóricas de alta cardinalidad: "
                + ", ".join(high_card)
                + ". Evaluar agrupación antes de codificar."
            )

    if data_quality is not None and not data_quality.empty:
        critical = data_quality[
            data_quality["quality_severity"].eq("CRITICAL")
        ]["variable"].tolist()

        warning = data_quality[
            data_quality["quality_severity"].eq("WARNING")
        ]["variable"].tolist()

        leakage = data_quality[
            data_quality["flags"].str.contains("POTENTIAL_LEAKAGE", na=False)
        ]["variable"].tolist()

        derived = data_quality[
            data_quality["flags"].str.contains("DERIVED", na=False)
        ]["variable"].tolist()

        if critical:
            conclusions.append(
                "Variables con problemas críticos de calidad: "
                + ", ".join(critical)
                + ". Requieren exclusión, imputación o revisión de origen."
            )

        if warning:
            conclusions.append(
                "Variables con alertas de calidad: "
                + ", ".join(warning)
                + ". Revisar antes de modelar."
            )

        if leakage:
            conclusions.append(
                "Variables con posible fuga de información: "
                + ", ".join(leakage)
                + ". No usarlas como predictoras si el objetivo es predecir error."
            )

        if derived:
            conclusions.append(
                "Variables derivadas detectadas: "
                + ", ".join(derived)
                + ". Evitar duplicidad en fases de modelamiento."
            )

    if not conclusions:
        conclusions.append(
            "No se detectaron alertas críticas en la fase descriptiva univariada."
        )

    return conclusions


def build_key_tables(df: pd.DataFrame) -> dict:
    numeric_summary = build_numeric_summary(df)
    categorical_summary = build_categorical_summary(df)
    outlier_summary = build_outlier_summary(df)
    data_quality = build_data_quality_table(df)
    variable_classification = build_variable_classification()
    automatic_conclusions = build_automatic_conclusions(
        numeric_summary=numeric_summary,
        categorical_summary=categorical_summary,
        data_quality=data_quality,
    )

    return {
        "numeric_summary": numeric_summary,
        "categorical_summary": categorical_summary,
        "outlier_summary": outlier_summary,
        "data_quality": data_quality,
        "variable_classification": variable_classification,
        "automatic_conclusions": automatic_conclusions,
    }


def run_descriptive_analysis(df: pd.DataFrame) -> dict:
    return build_key_tables(df)