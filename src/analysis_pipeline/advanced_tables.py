from __future__ import annotations

import numpy as np
import pandas as pd


# ==========================================================
# CONFIGURACIÓN CENTRAL
# ==========================================================

LEAKAGE_COLUMNS = {
    "severity",
    "is_duplicate",
    "error_duplicado",
    "error_orphan",
    "error_null",
    "error_blob_timeout",
}

IDENTIFIER_COLUMNS = {
    "file_id",
    "run_id",
    "sequence",
    "content_hash",
    "hash_head",
    "hash_tail",
}

DERIVED_COLUMNS = {
    "queue_pressure",
    "congestion_factor",
    "size_range",
}

TARGET_COLUMNS = {
    "transfer_duration_sec",
    "storage_cost",
    "has_error",
}


# Variables binarias que no deben recibir recomendaciones de transformación continua.
BINARY_COLUMNS = {
    "has_error",
    "movement_storage",
    "is_duplicate",
    "error_duplicado",
    "error_orphan",
    "error_null",
    "error_blob_timeout",
}


# Relaciones derivadas conocidas.
DERIVED_RELATION_RULES = {
    "queue_pressure": {
        "derived_from": "hourly_arrival_count, hourly_capacity",
        "reason": "representa presión de cola calculada desde carga y capacidad",
        "recommendation": "usar como resumen operativo o usar componentes originales, no ambos sin revisión",
    },
    "congestion_factor": {
        "derived_from": "hourly_arrival_count, hourly_capacity, queue_pressure",
        "reason": "representa congestión derivada de presión/capacidad del sistema",
        "recommendation": "revisar redundancia con queue_pressure y hourly_arrival_count",
    },
    "size_range": {
        "derived_from": "size_mb",
        "reason": "variable categórica derivada del tamaño del archivo",
        "recommendation": "usar para segmentación descriptiva; evitar combinar con size_mb sin justificación",
    },
}


# ==========================================================
# HELPERS
# ==========================================================

def _empty_df(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _is_binary_series(series: pd.Series) -> bool:
    values = set(series.dropna().unique())
    return values.issubset({0, 1, False, True}) and len(values) <= 2


def _numeric_multivariate_frame(
    df: pd.DataFrame,
    include_targets: bool = True,
    include_derived: bool = True,
    include_binary: bool = True,
) -> pd.DataFrame:
    """Devuelve variables numéricas válidas para análisis multivariado exploratorio.

    No incluye identificadores ni variables con fuga conocida.
    No calcula modelos, VIF ni inferencia.
    """
    numeric_df = df.select_dtypes(include="number").copy()

    drop_cols = set(IDENTIFIER_COLUMNS) | set(LEAKAGE_COLUMNS)

    if not include_targets:
        drop_cols |= TARGET_COLUMNS

    if not include_derived:
        drop_cols |= DERIVED_COLUMNS

    if not include_binary:
        binary_cols = {
            col
            for col in numeric_df.columns
            if col in BINARY_COLUMNS or _is_binary_series(numeric_df[col])
        }
        drop_cols |= binary_cols

    numeric_df = numeric_df.drop(
        columns=[c for c in drop_cols if c in numeric_df.columns],
        errors="ignore",
    )

    numeric_df = numeric_df.replace([np.inf, -np.inf], np.nan)
    numeric_df = numeric_df.dropna(axis=1, how="all")
    numeric_df = numeric_df.loc[:, numeric_df.nunique(dropna=True) > 1]

    return numeric_df


def _relationship_strength(correlation: float) -> str:
    value = abs(correlation)

    if value >= 0.7:
        return "fuerte"
    if value >= 0.5:
        return "moderada"
    if value >= 0.3:
        return "débil"

    return "muy débil"


def _interpret_correlation(correlation: float) -> str:
    value = abs(correlation)

    if value >= 0.7:
        return "relación alta; revisar como variable clave para fase posterior"
    if value >= 0.5:
        return "relación moderada; útil para priorización exploratoria"
    if value >= 0.3:
        return "relación débil; puede aportar contexto"
    return "relación baja; baja prioridad exploratoria"


def _relationship_role(var1: str, var2: str) -> str:
    if var1 in TARGET_COLUMNS and var2 in TARGET_COLUMNS:
        return "target_target"

    if var1 in TARGET_COLUMNS or var2 in TARGET_COLUMNS:
        return "target_variable"

    if var1 in DERIVED_COLUMNS or var2 in DERIVED_COLUMNS:
        return "derived_relation"

    return "variable_variable"


# ==========================================================
# MATRICES DE CORRELACIÓN
# ==========================================================

def build_correlation_matrix(
    df: pd.DataFrame,
    method: str = "spearman",
    include_targets: bool = True,
    include_derived: bool = True,
) -> pd.DataFrame:
    """Construye matriz de correlación Pearson o Spearman.

    Spearman debe usarse como referencia principal por robustez ante relaciones
    monotónicas no necesariamente lineales.
    """
    numeric_df = _numeric_multivariate_frame(
        df,
        include_targets=include_targets,
        include_derived=include_derived,
        include_binary=True,
    )

    if numeric_df.shape[1] < 2:
        return pd.DataFrame()

    return numeric_df.corr(method=method)


def build_predictor_correlation_matrix(
    df: pd.DataFrame,
    method: str = "spearman",
    include_derived: bool = True,
) -> pd.DataFrame:
    """Matriz solo entre variables explicativas/pre-modelado.

    Excluye targets para evitar mezclar relaciones target-target en una matriz
    que debe servir para revisar redundancia entre variables de entrada.
    """
    return build_correlation_matrix(
        df,
        method=method,
        include_targets=False,
        include_derived=include_derived,
    )


# ==========================================================
# TOP RELACIONES ENTRE VARIABLES
# ==========================================================

def build_top_relationships(
    df: pd.DataFrame,
    method: str = "spearman",
    threshold: float = 0.5,
    include_targets: bool = True,
    include_derived: bool = True,
    exclude_target_target_pairs: bool = True,
) -> pd.DataFrame:
    corr = build_correlation_matrix(
        df,
        method=method,
        include_targets=include_targets,
        include_derived=include_derived,
    )

    columns = [
        "variable_1",
        "variable_2",
        "correlation",
        "abs_correlation",
        "tipo_relacion",
        "relationship_role",
        "method",
    ]

    rows = []

    if corr.empty:
        return _empty_df(columns)

    corr_columns = list(corr.columns)

    for i, var1 in enumerate(corr_columns):
        for var2 in corr_columns[i + 1:]:
            role = _relationship_role(var1, var2)

            if exclude_target_target_pairs and role == "target_target":
                continue

            value = corr.loc[var1, var2]

            if pd.notna(value) and abs(value) >= threshold:
                rows.append({
                    "variable_1": var1,
                    "variable_2": var2,
                    "correlation": value,
                    "abs_correlation": abs(value),
                    "tipo_relacion": _relationship_strength(value),
                    "relationship_role": role,
                    "method": method,
                })

    if not rows:
        return _empty_df(columns)

    return (
        pd.DataFrame(rows)
        .sort_values("abs_correlation", ascending=False)
        .reset_index(drop=True)
    )


# ==========================================================
# RELACIÓN CON TARGETS
# ==========================================================

def build_target_relationships(
    df: pd.DataFrame,
    targets: set[str] | None = None,
    method: str = "spearman",
    exclude_other_targets: bool = True,
) -> pd.DataFrame:
    """Relaciones exploratorias contra targets.

    Por defecto excluye otros targets del ranking para que la tabla no mezcle
    objetivos entre sí.
    """
    targets = targets or TARGET_COLUMNS

    corr = build_correlation_matrix(
        df,
        method=method,
        include_targets=True,
        include_derived=True,
    )

    columns = [
        "target",
        "variable",
        "correlation",
        "abs_correlation",
        "ranking",
        "interpretacion",
        "method",
    ]

    rows = []

    if corr.empty:
        return _empty_df(columns)

    for target in sorted(targets):
        if target not in corr.columns:
            continue

        drop_labels = [target]

        if exclude_other_targets:
            drop_labels = list(TARGET_COLUMNS)

        target_corr = (
            corr[target]
            .drop(labels=drop_labels, errors="ignore")
            .dropna()
            .sort_values(key=lambda s: s.abs(), ascending=False)
        )

        for ranking, (variable, value) in enumerate(target_corr.items(), start=1):
            rows.append({
                "target": target,
                "variable": variable,
                "correlation": value,
                "abs_correlation": abs(value),
                "ranking": ranking,
                "interpretacion": _interpret_correlation(value),
                "method": method,
            })

    if not rows:
        return _empty_df(columns)

    return pd.DataFrame(rows)


# ==========================================================
# REDUNDANCIA
# ==========================================================

def build_redundancy_table(
    df: pd.DataFrame,
    threshold: float = 0.9,
    method: str = "spearman",
) -> pd.DataFrame:
    """Detecta redundancia entre variables.

    Usa correlación numérica y complementa con reglas de derivadas conocidas.
    """
    top = build_top_relationships(
        df,
        method=method,
        threshold=threshold,
        include_targets=True,
        include_derived=True,
        exclude_target_target_pairs=True,
    )

    columns = [
        "variable_a",
        "variable_b",
        "correlation",
        "razon",
        "accion_sugerida",
    ]

    rows = []

    for _, row in top.iterrows():
        var1 = row["variable_1"]
        var2 = row["variable_2"]

        reason = "correlación muy alta"
        action = "revisar si una variable puede excluirse antes del modelado"

        if var1 in DERIVED_COLUMNS or var2 in DERIVED_COLUMNS:
            reason = "variable derivada altamente relacionada con otra variable"
            action = "preferir variable original o derivada, pero no ambas"

        if {var1, var2} == {"queue_pressure", "congestion_factor"}:
            reason = "ambas representan presión/congestión operativa"
            action = "conservar solo la más interpretable para la fase posterior"

        rows.append({
            "variable_a": var1,
            "variable_b": var2,
            "correlation": row["correlation"],
            "razon": reason,
            "accion_sugerida": action,
        })

    # Regla explícita para derivada categórica size_range.
    if "size_range" in df.columns and "size_mb" in df.columns:
        already_exists = any(
            {row["variable_a"], row["variable_b"]} == {"size_range", "size_mb"}
            for row in rows
        )

        if not already_exists:
            rows.append({
                "variable_a": "size_mb",
                "variable_b": "size_range",
                "correlation": np.nan,
                "razon": "size_range es una derivación categórica de size_mb",
                "accion_sugerida": "usar size_mb para análisis continuo o size_range para segmentación, no ambas sin justificación",
            })

    if not rows:
        return _empty_df(columns)

    return pd.DataFrame(rows)


# ==========================================================
# DERIVADAS
# ==========================================================

def build_derived_variable_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "variable",
        "derived_from",
        "motivo",
        "decision_pre_modeling",
    ]

    rows = []

    for variable, rule in DERIVED_RELATION_RULES.items():
        if variable not in df.columns:
            continue

        rows.append({
            "variable": variable,
            "derived_from": rule["derived_from"],
            "motivo": rule["reason"],
            "decision_pre_modeling": rule["recommendation"],
        })

    if not rows:
        return _empty_df(columns)

    return pd.DataFrame(rows)


# ==========================================================
# FUGA DE INFORMACIÓN Y EXCLUSIONES
# ==========================================================

def build_leakage_and_exclusion_table(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "variable",
        "type",
        "motivo",
        "decision_pre_modeling",
    ]

    rows = []

    for col in df.columns:
        if col in LEAKAGE_COLUMNS:
            rows.append({
                "variable": col,
                "type": "leakage",
                "motivo": "codifica información posterior al evento o depende del error",
                "decision_pre_modeling": "NO_USAR",
            })
        elif col in IDENTIFIER_COLUMNS:
            rows.append({
                "variable": col,
                "type": "identifier",
                "motivo": "identificador/hash sin valor generalizable",
                "decision_pre_modeling": "EXCLUIR",
            })
        elif col in DERIVED_COLUMNS:
            rows.append({
                "variable": col,
                "type": "derived",
                "motivo": "variable derivada; revisar redundancia con componentes originales",
                "decision_pre_modeling": "REVISAR",
            })

    if not rows:
        return _empty_df(columns)

    return pd.DataFrame(rows)


# ==========================================================
# NO LINEALIDAD: PEARSON VS SPEARMAN
# ==========================================================

def build_nonlinearity_table(
    df: pd.DataFrame,
    min_spearman: float = 0.4,
    delta_threshold: float = 0.2,
    exclude_target_target_pairs: bool = True,
) -> pd.DataFrame:
    pearson = build_correlation_matrix(df, method="pearson")
    spearman = build_correlation_matrix(df, method="spearman")

    columns = [
        "variable_1",
        "variable_2",
        "pearson",
        "spearman",
        "delta_abs",
        "relationship_role",
        "interpretacion",
    ]

    rows = []

    if pearson.empty or spearman.empty:
        return _empty_df(columns)

    common_columns = [c for c in pearson.columns if c in spearman.columns]

    for i, var1 in enumerate(common_columns):
        for var2 in common_columns[i + 1:]:
            role = _relationship_role(var1, var2)

            if exclude_target_target_pairs and role == "target_target":
                continue

            p = pearson.loc[var1, var2]
            s = spearman.loc[var1, var2]

            if pd.isna(p) or pd.isna(s):
                continue

            delta = abs(s) - abs(p)

            if abs(s) >= min_spearman and delta >= delta_threshold:
                rows.append({
                    "variable_1": var1,
                    "variable_2": var2,
                    "pearson": p,
                    "spearman": s,
                    "delta_abs": delta,
                    "relationship_role": role,
                    "interpretacion": "posible relación monotónica no lineal; evaluar transformación o segmentación",
                })

    if not rows:
        return _empty_df(columns)

    return (
        pd.DataFrame(rows)
        .sort_values("delta_abs", ascending=False)
        .reset_index(drop=True)
    )


# ==========================================================
# CANDIDATAS A TRANSFORMACIÓN
# ==========================================================

def build_transformation_candidates(
    df: pd.DataFrame,
    include_targets: bool = False,
    include_binary: bool = False,
) -> pd.DataFrame:
    """Detecta candidatas a transformación.

    Por defecto excluye targets y variables binarias para evitar recomendar
    transformaciones continuas sobre `has_error` u otros flags.
    """
    numeric_df = _numeric_multivariate_frame(
        df,
        include_targets=include_targets,
        include_derived=True,
        include_binary=include_binary,
    )

    columns = [
        "variable",
        "skewness",
        "zero_pct",
        "positive_pct",
        "suggested_transformation",
        "reason",
    ]

    rows = []

    for col in numeric_df.columns:
        data = numeric_df[col].replace([np.inf, -np.inf], np.nan).dropna()

        if data.empty:
            continue

        if not include_binary and (col in BINARY_COLUMNS or _is_binary_series(data)):
            continue

        positive_pct = (data > 0).mean() * 100
        zero_pct = (data == 0).mean() * 100
        skewness = data.skew()

        if abs(skewness) > 1 and positive_pct >= 95:
            transformation = "log1p"
            reason = "asimetría alta y mayoría de valores positivos"
        elif abs(skewness) > 1:
            transformation = "winsorizar, discretizar o segmentar"
            reason = "asimetría alta con ceros o valores no positivos"
        else:
            transformation = "sin transformación obligatoria"
            reason = "asimetría controlada"

        rows.append({
            "variable": col,
            "skewness": skewness,
            "zero_pct": zero_pct,
            "positive_pct": positive_pct,
            "suggested_transformation": transformation,
            "reason": reason,
        })

    if not rows:
        return _empty_df(columns)

    return (
        pd.DataFrame(rows)
        .sort_values("skewness", key=lambda s: s.abs(), ascending=False)
        .reset_index(drop=True)
    )


# ==========================================================
# RESUMEN EJECUTIVO AUTOMÁTICO
# ==========================================================

def build_executive_summary(df: pd.DataFrame) -> pd.DataFrame:
    top_relationships = build_top_relationships(df, threshold=0.5)
    target_relationships = build_target_relationships(df)
    redundancy = build_redundancy_table(df)
    leakage = build_leakage_and_exclusion_table(df)
    nonlinear = build_nonlinearity_table(df)
    derived = build_derived_variable_table(df)
    transformation_candidates = build_transformation_candidates(df)

    leakage_count = 0
    if not leakage.empty and "type" in leakage.columns:
        leakage_count = len(leakage[leakage["type"] == "leakage"])

    rows = [
        {
            "hallazgo": "relaciones_multivariadas",
            "detalle": f"Se detectaron {len(top_relationships)} relaciones con |Spearman| >= 0.5, excluyendo pares target-target.",
        },
        {
            "hallazgo": "targets",
            "detalle": f"Se evaluaron relaciones contra targets disponibles: {', '.join(sorted(TARGET_COLUMNS & set(df.columns)))}.",
        },
        {
            "hallazgo": "redundancia",
            "detalle": f"Se detectaron {len(redundancy)} relaciones redundantes o derivadas que requieren revisión.",
        },
        {
            "hallazgo": "fuga_informacion",
            "detalle": f"Se identificaron {leakage_count} variables con fuga o riesgo directo.",
        },
        {
            "hallazgo": "variables_derivadas",
            "detalle": f"Se documentaron {len(derived)} variables derivadas para control pre-modelado.",
        },
        {
            "hallazgo": "no_linealidad",
            "detalle": f"Se detectaron {len(nonlinear)} relaciones con Spearman considerablemente mayor que Pearson.",
        },
        {
            "hallazgo": "transformaciones",
            "detalle": f"Se identificaron {len(transformation_candidates)} variables continuas para revisar transformación; se excluyen targets y binarias por defecto.",
        },
    ]

    return pd.DataFrame(rows)


# ==========================================================
# BUILDER PRINCIPAL
# ==========================================================

def build_advanced_tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "executive_summary": build_executive_summary(df),
        "pearson_correlation": build_correlation_matrix(df, method="pearson"),
        "spearman_correlation": build_correlation_matrix(df, method="spearman"),
        "predictor_spearman_correlation": build_predictor_correlation_matrix(df, method="spearman"),
        "top_relationships": build_top_relationships(df, method="spearman", threshold=0.5),
        "target_relationships": build_target_relationships(df, method="spearman"),
        "redundancy": build_redundancy_table(df, threshold=0.9),
        "derived_variables": build_derived_variable_table(df),
        "leakage_and_exclusions": build_leakage_and_exclusion_table(df),
        "nonlinear_relationships": build_nonlinearity_table(df),
        "transformation_candidates": build_transformation_candidates(df),
    }
