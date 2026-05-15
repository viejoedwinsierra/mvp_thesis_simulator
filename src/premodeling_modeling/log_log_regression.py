from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from premodeling_modeling.common import (
    TechniqueResult,
    build_model_frame,
    build_prediction_table,
    build_preprocessor,
    get_feature_names,
    regression_metrics,
    save_table,
    split_data,
)
from premodeling_modeling.plots import (
    plot_actual_vs_predicted,
    plot_coefficients,
    plot_residual_histogram,
    plot_residuals,
)


TECHNIQUE = "log_log_regression"


def run_log_log_regression(
    df: pd.DataFrame,
    output_dir: str | Path,
    target: str = "storage_cost",
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    test_size: float = 0.25,
    random_state: int = 42,
) -> TechniqueResult:
    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    table_dir = output_dir / "tables"
    model_dir = output_dir / "models"

    if numeric_features is None:
        numeric_features = [
            "size_mb",
            "days_stored",
            "days_since_last_access",
            "transfer_duration_sec",
            "transfer_speed_mbps",
            "queue_pressure",
            "congestion_factor",
        ]

    if categorical_features is None:
        categorical_features = [
            "file_type",
            "storage_tier",
            "time_slot",
            "scenario_name",
        ]

    log_features = [
        col for col in numeric_features
        if col not in {"movement_storage", "has_error"}
    ]

    X, y, numeric_features, categorical_features = build_model_frame(
        df=df,
        target=target,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        log_target=True,
        log_numeric_features=log_features,
    )

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = regression_metrics(y_test, y_pred, technique=TECHNIQUE, target=f"log1p_{target}")

    feature_names = get_feature_names(model.named_steps["preprocessor"])
    raw_model = model.named_steps["model"]

    coefficients = pd.DataFrame({
        "feature": feature_names,
        "coefficient": raw_model.coef_,
    })
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients["intercept"] = raw_model.intercept_
    coefficients["interpretation"] = "En variables log-transformadas, el coeficiente aproxima elasticidad."

    predictions = build_prediction_table(y_test, y_pred, target=f"log1p_{target}")

    save_table(metrics, table_dir / "metrics.csv")
    save_table(coefficients, table_dir / "coefficients_elasticities.csv")
    save_table(predictions, table_dir / "predictions.csv")

    plot_paths = [
        plot_actual_vs_predicted(y_test, y_pred, plot_dir, "actual_vs_predicted_log_scale.png"),
        plot_residuals(y_test, y_pred, plot_dir, "residuals_vs_prediction_log_scale.png"),
        plot_residual_histogram(y_test, y_pred, plot_dir, "residual_histogram_log_scale.png"),
    ]

    coef_plot = plot_coefficients(coefficients, plot_dir, "elasticities.png")
    if coef_plot:
        plot_paths.append(coef_plot)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"{TECHNIQUE}_{target}.joblib"
    joblib.dump(model, model_path)

    return TechniqueResult(
        technique=TECHNIQUE,
        target=target,
        output_dir=output_dir,
        metrics=metrics,
        coefficients=coefficients,
        predictions=predictions,
        model_path=model_path,
        plots=plot_paths,
    )
