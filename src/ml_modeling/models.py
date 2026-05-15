from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import SGDClassifier, SGDRegressor
from sklearn.pipeline import Pipeline

from ml_modeling.common import (
    MLModelResult,
    build_model_frame,
    build_preprocessor,
    classification_metrics,
    extract_feature_importance,
    persist_model,
    prediction_table,
    regression_metrics,
    save_table,
    split_data,
)
from ml_modeling.plots import (
    plot_actual_vs_predicted,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_residual_histogram,
    plot_residuals,
)


REGRESSION_MODELS = {
    "random_forest_regressor": RandomForestRegressor(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    ),
    "gradient_boosting_regressor": GradientBoostingRegressor(
        random_state=42,
        n_estimators=180,
        learning_rate=0.05,
        max_depth=3,
    ),
    "hist_gradient_boosting_regressor": HistGradientBoostingRegressor(
        random_state=42,
        max_iter=220,
        learning_rate=0.05,
    ),
    "sgd_regressor": SGDRegressor(
        random_state=42,
        max_iter=3000,
        tol=1e-3,
    ),
}


CLASSIFICATION_MODELS = {
    "random_forest_classifier": RandomForestClassifier(
        n_estimators=250,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
        class_weight="balanced",
    ),
    "gradient_boosting_classifier": GradientBoostingClassifier(
        random_state=42,
        n_estimators=180,
        learning_rate=0.05,
        max_depth=3,
    ),
    "hist_gradient_boosting_classifier": HistGradientBoostingClassifier(
        random_state=42,
        max_iter=220,
        learning_rate=0.05,
    ),
    "sgd_classifier": SGDClassifier(
        random_state=42,
        loss="log_loss",
        max_iter=3000,
        tol=1e-3,
        class_weight="balanced",
    ),
}


def run_regression_model(
    df: pd.DataFrame,
    output_dir: str | Path,
    technique: str,
    target: str = "storage_cost",
    test_size: float = 0.25,
    random_state: int = 42,
) -> MLModelResult:
    if technique not in REGRESSION_MODELS:
        raise ValueError(f"Modelo de regresion no soportado: {technique}")

    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    table_dir = output_dir / "tables"

    scale_numeric = technique == "sgd_regressor"

    X, y, numeric_features, categorical_features = build_model_frame(
        df=df,
        target=target,
    )

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        task_type="regression",
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=scale_numeric,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", REGRESSION_MODELS[technique]),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = regression_metrics(
        y_true=y_test,
        y_pred=y_pred,
        technique=technique,
        target=target,
    )

    feature_importance = extract_feature_importance(
        model=model,
        preprocessor=model.named_steps["preprocessor"],
        technique=technique,
    )

    predictions = prediction_table(y_test, y_pred, target=target)

    save_table(metrics, table_dir / "metrics.csv")
    save_table(feature_importance, table_dir / "feature_importance.csv")
    save_table(predictions, table_dir / "predictions.csv")

    plots = [
        plot_actual_vs_predicted(y_test, y_pred, plot_dir),
        plot_residuals(y_test, y_pred, plot_dir),
        plot_residual_histogram(y_test, y_pred, plot_dir),
    ]

    importance_plot = plot_feature_importance(feature_importance, plot_dir)
    if importance_plot:
        plots.append(importance_plot)

    model_path = persist_model(model, output_dir, technique, target)

    return MLModelResult(
        family="machine_learning",
        technique=technique,
        target=target,
        task_type="regression",
        output_dir=output_dir,
        metrics=metrics,
        feature_importance=feature_importance,
        predictions=predictions,
        model_path=model_path,
        plots=plots,
    )


def run_classification_model(
    df: pd.DataFrame,
    output_dir: str | Path,
    technique: str,
    target: str = "has_error",
    test_size: float = 0.25,
    random_state: int = 42,
) -> MLModelResult:
    if technique not in CLASSIFICATION_MODELS:
        raise ValueError(f"Modelo de clasificacion no soportado: {technique}")

    output_dir = Path(output_dir)
    plot_dir = output_dir / "plots"
    table_dir = output_dir / "tables"

    scale_numeric = technique == "sgd_classifier"

    X, y, numeric_features, categorical_features = build_model_frame(
        df=df,
        target=target,
        extra_exclusions={"case_group"},
    )

    y = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)

    if y.nunique() < 2:
        raise ValueError(f"{target} requiere al menos dos clases.")

    X_train, X_test, y_train, y_test = split_data(
        X,
        y,
        task_type="classification",
        test_size=test_size,
        random_state=random_state,
    )

    preprocessor = build_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=scale_numeric,
    )

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", CLASSIFICATION_MODELS[technique]),
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    if hasattr(model.named_steps["model"], "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model.named_steps["model"], "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    metrics = classification_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_score=y_score,
        technique=technique,
        target=target,
    )

    feature_importance = extract_feature_importance(
        model=model,
        preprocessor=model.named_steps["preprocessor"],
        technique=technique,
    )

    predictions = prediction_table(y_test, y_pred, target=target, y_score=y_score)

    save_table(metrics, table_dir / "metrics.csv")
    save_table(feature_importance, table_dir / "feature_importance.csv")
    save_table(predictions, table_dir / "predictions.csv")

    plots = [
        plot_confusion_matrix(y_test, y_pred, plot_dir),
    ]

    importance_plot = plot_feature_importance(feature_importance, plot_dir)
    if importance_plot:
        plots.append(importance_plot)

    model_path = persist_model(model, output_dir, technique, target)

    return MLModelResult(
        family="machine_learning",
        technique=technique,
        target=target,
        task_type="classification",
        output_dir=output_dir,
        metrics=metrics,
        feature_importance=feature_importance,
        predictions=predictions,
        model_path=model_path,
        plots=plots,
    )
