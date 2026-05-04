import pandas as pd
import numpy as np


class DescriptiveAnalysis:

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.numeric_df = self.df.select_dtypes(include=[np.number])
        self.categorical_df = self.df.select_dtypes(exclude=[np.number])

    def general_info(self):
        return {
            "rows": self.df.shape[0],
            "columns": self.df.shape[1],
            "numeric_columns": list(self.numeric_df.columns),
            "categorical_columns": list(self.categorical_df.columns),
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "duplicated_rows": int(self.df.duplicated().sum()),
            "memory_mb": round(self.df.memory_usage(deep=True).sum() / (1024 ** 2), 4),
        }

    def null_analysis(self):
        nulls = self.df.isnull().sum()
        null_pct = (nulls / len(self.df)) * 100

        return (
            pd.DataFrame({
                "variable": nulls.index,
                "null_count": nulls.values,
                "null_pct": null_pct.values,
            })
            .sort_values(by="null_pct", ascending=False)
            .reset_index(drop=True)
        )

    def descriptive_stats(self):
        return self.df.describe(include="all").T

    def numeric_summary(self):
        rows = []

        for col in self.numeric_df.columns:
            data = self.numeric_df[col].dropna()

            if data.empty:
                continue

            rows.append({
                "variable": col,
                "count": data.count(),
                "mean": data.mean(),
                "median": data.median(),
                "std": data.std(),
                "min": data.min(),
                "q1": data.quantile(0.25),
                "q3": data.quantile(0.75),
                "max": data.max(),
                "range": data.max() - data.min(),
                "iqr": data.quantile(0.75) - data.quantile(0.25),
                "cv": data.std() / data.mean() if data.mean() != 0 else np.nan,
                "skewness": data.skew(),
                "kurtosis": data.kurtosis(),
                "unique_values": data.nunique(),
                "zero_count": int((data == 0).sum()),
                "zero_pct": float((data == 0).mean() * 100),
            })

        return pd.DataFrame(rows)

    def categorical_summary(self):
        rows = []

        for col in self.categorical_df.columns:
            data = self.categorical_df[col].dropna()

            if data.empty:
                continue

            mode_value = data.mode().iloc[0] if not data.mode().empty else None
            mode_count = int((data == mode_value).sum()) if mode_value is not None else 0

            rows.append({
                "variable": col,
                "count": data.count(),
                "unique_values": data.nunique(),
                "most_frequent": mode_value,
                "most_frequent_count": mode_count,
                "most_frequent_pct": mode_count / len(data) * 100,
            })

        return pd.DataFrame(rows)

    def distribution_shape(self):
        rows = []

        for col in self.numeric_df.columns:
            data = self.numeric_df[col].dropna()

            if data.empty:
                continue

            skew = data.skew()
            kurt = data.kurtosis()

            rows.append({
                "variable": col,
                "skewness": skew,
                "kurtosis": kurt,
                "skewness_interpretation": self._interpret_skewness(skew),
                "kurtosis_interpretation": self._interpret_kurtosis(kurt),
            })

        return pd.DataFrame(rows)

    def outlier_analysis(self):
        rows = []

        for col in self.numeric_df.columns:
            data = self.numeric_df[col].dropna()

            if data.empty:
                continue

            q1 = data.quantile(0.25)
            q3 = data.quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers = data[(data < lower) | (data > upper)]

            rows.append({
                "variable": col,
                "outlier_count": len(outliers),
                "outlier_pct": len(outliers) / len(data) * 100,
                "lower_bound": lower,
                "upper_bound": upper,
                "outlier_severity": self._interpret_outliers(len(outliers) / len(data) * 100),
            })

        return pd.DataFrame(rows).sort_values(
            by="outlier_pct",
            ascending=False
        ).reset_index(drop=True)

    def data_quality_flags(self):
        rows = []

        nulls = self.null_analysis().set_index("variable")
        outliers = self.outlier_analysis().set_index("variable")

        for col in self.df.columns:
            null_pct = nulls.loc[col, "null_pct"] if col in nulls.index else 0
            unique_pct = self.df[col].nunique(dropna=True) / len(self.df) * 100

            flags = []

            if null_pct > 20:
                flags.append("HIGH_NULLS")
            elif null_pct > 5:
                flags.append("MODERATE_NULLS")

            if unique_pct > 95:
                flags.append("HIGH_CARDINALITY")

            if col in outliers.index and outliers.loc[col, "outlier_pct"] > 10:
                flags.append("HIGH_OUTLIERS")

            if self.df[col].nunique(dropna=True) <= 1:
                flags.append("CONSTANT")

            rows.append({
                "variable": col,
                "null_pct": null_pct,
                "unique_pct": unique_pct,
                "flags": ", ".join(flags) if flags else "OK",
            })

        return pd.DataFrame(rows)

    @staticmethod
    def _interpret_skewness(value):
        if pd.isna(value):
            return "No evaluable"
        if abs(value) < 0.5:
            return "Aproximadamente simétrica"
        if abs(value) < 1:
            return "Asimetría moderada"
        return "Asimetría fuerte"

    @staticmethod
    def _interpret_kurtosis(value):
        if pd.isna(value):
            return "No evaluable"
        if value > 3:
            return "Colas pesadas / alta concentración de extremos"
        if value < -1:
            return "Distribución plana"
        return "Curtosis moderada"

    @staticmethod
    def _interpret_outliers(value):
        if value == 0:
            return "Sin outliers"
        if value < 5:
            return "Bajo"
        if value < 10:
            return "Moderado"
        return "Alto"

    def run(self):
        return {
            "general_info": self.general_info(),
            "nulls": self.null_analysis(),
            "descriptive_stats": self.descriptive_stats(),
            "numeric_summary": self.numeric_summary(),
            "categorical_summary": self.categorical_summary(),
            "distribution": self.distribution_shape(),
            "outliers": self.outlier_analysis(),
            "quality_flags": self.data_quality_flags(),
        }