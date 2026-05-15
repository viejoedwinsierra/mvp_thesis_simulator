from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

from premodeling_modeling.common import (
    DEFAULT_CATEGORICAL_FEATURES,
    DEFAULT_NUMERIC_FEATURES,
    TechniqueResult,
    build_model_frame,
    build_preprocessor,
    classification_metrics,
    get_feature_names,
    save_table,
    split_data,
)
from premodeling_modeling.plots import plot_coefficients, plot_confusion_matrix


TECHNIQUE = "logistic_regression_error"


def run_logistic_regression_error(
    df: pd.DataFrame,
    output_dir: str | Path,
    target: str = "has_error",
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
            if col not in {target, "transfer_duration_sec"}
        ]

    if categorical_features is None:
        categorical_features = DEFAULT_CATEGORICAL_FEATURES.copy()

    X, y, numeric_features, categorical_features = build_model_frame(
        df=df,
        target=target,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    if y.nunique() < 2:
        raise ValueError("Regresion logistica requiere dos clases en has_error.")

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )

    preprocessor = build_preprocessor(
        numeric_features,
        categorical_features,
        scale_numeric=True,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = classification_metrics(
        y_test,
        y_pred,
        y_score,
        technique=TECHNIQUE,
        target=target,
    )

    feature_names = get_feature_names(model.named_steps["preprocessor"])
    raw_model = model.named_steps["model"]

    coefficients = pd.DataFrame({
        "feature": feature_names,
        "coefficient": raw_model.coef_[0],
    })
    coefficients["abs_coefficient"] = coefficients["coefficient"].abs()
    coefficients["interpretation"] = "Coeficiente positivo aumenta log-odds de error."

    predictions = pd.DataFrame({
        "target": target,
        "y_true": y_test.values,
        "y_pred": y_pred,
        "probability_error": y_score,
    })

    cm = pd.DataFrame(
        confusion_matrix(y_test, y_pred),
        index=["real_0", "real_1"],
        columns=["pred_0", "pred_1"],
    )

    save_table(metrics, table_dir / "metrics.csv")
    save_table(coefficients, table_dir / "coefficients_log_odds.csv")
    save_table(predictions, table_dir / "predictions.csv")
    save_table(cm.reset_index(names="class"), table_dir / "confusion_matrix.csv")

    plot_paths = [
        plot_confusion_matrix(cm, plot_dir, "confusion_matrix.png"),
    ]

    coef_plot = plot_coefficients(coefficients, plot_dir, "coefficients_log_odds.png")
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
