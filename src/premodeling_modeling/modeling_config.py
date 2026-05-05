from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    target: str
    model_type: str
    description: str


@dataclass(frozen=True)
class ModelingConfig:
    """Configuración de modelos estadísticos finales.

    Se enfoca en modelos interpretables, no en machine learning complejo.

    Mantiene compatibilidad:
    - conserva los targets y modelos actuales;
    - conserva test_size, random_state y max_features_for_statsmodels;
    - agrega configuración opcional para persistencia sin cambiar el flujo
      cuando no se usa.
    """

    model_specs: dict[str, list[ModelSpec]] = field(default_factory=lambda: {
        "transfer_duration_sec": [
            ModelSpec(
                target="transfer_duration_sec",
                model_type="ols_log_linear",
                description="Regresión lineal sobre target transformado log1p.",
            ),
            ModelSpec(
                target="transfer_duration_sec",
                model_type="glm_gamma_log",
                description="GLM Gamma con link log para variable positiva.",
            ),
        ],
        "storage_cost": [
            ModelSpec(
                target="storage_cost",
                model_type="ols_log_linear",
                description="Regresión log-lineal para costo transformado.",
            ),
            ModelSpec(
                target="storage_cost",
                model_type="glm_gamma_log",
                description="GLM Gamma con link log para costo positivo.",
            ),
        ],
        "has_error": [
            ModelSpec(
                target="has_error",
                model_type="logistic_regression",
                description="Regresión logística binaria interpretable.",
            ),
        ],
    })

    # Configuración actual de entrenamiento.
    test_size: float = 0.25
    random_state: int = 42
    max_features_for_statsmodels: int = 40

    # Configuración opcional de persistencia.
    # Por defecto queda desactivada para no cambiar el comportamiento actual.
    save_models: bool = False
    model_output_dir: str = "output/models/statistical"
    registry_filename: str = "model_registry.json"

    # Metadatos de trazabilidad para registry.
    registry_version: str = "1.0"
    persist_training_metrics: bool = True
    persist_model_metadata: bool = True
