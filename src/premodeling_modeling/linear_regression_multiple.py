from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from premodeling_modeling.common import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
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


TECHNIQUE = "linear_regression_multiple"


def run_linear_regression_multiple(
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
            col for col in DEFAULT_NUMERIC_FEATURES
            if col not in {target}
        ]

    if categorical_features is None:
        categorical_features = DEFAULT_CATEGORICAL_FEATURES.copy()

    X, y, numeric_features, categorical_features = build_model_frame(
        df=df,
        target=target,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
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

    metrics = regression_metrics(y_test, y_pred, technique=TECHNIQUE, target=target)

    feature_names = get_feature_names(model.named_steps["preprocessor"])
    raw_model = model.named_steps["model"]

    coefficients = pd.DataFrame({
        "feature": feature_names,
        "coefficient": raw_model.coef_,
    })
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients["intercept"] = raw_model.intercept_

    predictions = build_prediction_table(y_test, y_pred, target=target)

    save_table(metrics, table_dir / "metrics.csv")
    save_table(coefficients, table_dir / "coefficients.csv")
    save_table(predictions, table_dir / "predictions.csv")

    plot_paths = [
        plot_actual_vs_predicted(y_test, y_pred, plot_dir, "actual_vs_predicted.png"),
        plot_residuals(y_test, y_pred, plot_dir, "residuals_vs_prediction.png"),
        plot_residual_histogram(y_test, y_pred, plot_dir, "residual_histogram.png"),
    ]

    coef_plot = plot_coefficients(coefficients, plot_dir, "coefficients.png")
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
