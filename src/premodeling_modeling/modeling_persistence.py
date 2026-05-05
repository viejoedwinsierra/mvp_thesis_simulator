from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd

try:
    from statsmodels.iolib.smpickle import load_pickle
except Exception:  # pragma: no cover
    load_pickle = None

from premodeling_modeling.statistical_models import fit_statistical_models


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .lower()
    )


def train_and_save_statistical_models(
    df: pd.DataFrame,
    output_dir: str | Path = "output/models/statistical",
    model_results: Dict[str, Dict[str, Any]] | None = None,
) -> dict[str, Path]:
    """Entrena (si es necesario) y guarda modelos estadísticos.

    Mejora clave:
    - Permite reutilizar `model_results` ya entrenados.
    - Evita reentrenamiento duplicado.
    - Mantiene compatibilidad total con la versión anterior.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 🔥 Evita reentrenamiento duplicado
    if model_results is None:
        model_results = fit_statistical_models(df)

    exported: dict[str, Path] = {}
    registry: list[dict[str, Any]] = []

    for target, payload in model_results.items():
        target_dir = output_dir / _safe_name(target)
        target_dir.mkdir(parents=True, exist_ok=True)

        metadata = payload.get("metadata", {})
        split = payload.get("split", {})

        for model_type, model_payload in payload.get("models", {}).items():
            if "result" not in model_payload:
                continue

            result = model_payload["result"]

            model_name = f"{_safe_name(target)}__{_safe_name(model_type)}"
            model_path = target_dir / f"{model_name}.pkl"
            metadata_path = target_dir / f"{model_name}.json"

            # Guardado del modelo
            result.save(str(model_path), remove_data=False)

            exog_names = list(getattr(result.model, "exog_names", []))

            model_metadata = {
                "target": target,
                "model_type": model_type,
                "model_name": model_name,
                "model_path": str(model_path),
                "metadata": metadata,
                "split": split,
                "exog_names": exog_names,
                "metrics_training_run": model_payload.get("metrics", {}),
                "model_file_exists": True,
                "metadata_path": str(metadata_path),
            }

            metadata_path.write_text(
                json.dumps(model_metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            exported[model_name] = model_path
            registry.append(model_metadata)

    # 🔥 Registry enriquecido
    registry_payload = {
        "registry_version": "1.0",
        "n_models": len(registry),
        "models": registry,
    }

    registry_path = output_dir / "model_registry.json"
    registry_path.write_text(
        json.dumps(registry_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    exported["registry"] = registry_path
    return exported


def load_saved_model(model_path: str | Path):
    """Carga un modelo guardado de statsmodels."""
    if load_pickle is None:
        raise ImportError("No se pudo importar statsmodels.iolib.smpickle.load_pickle")

    return load_pickle(str(model_path))


def load_model_registry(
    model_dir: str | Path = "output/models/statistical",
) -> dict:
    registry_path = Path(model_dir) / "model_registry.json"

    if not registry_path.exists():
        raise FileNotFoundError(
            f"No existe el registro de modelos: {registry_path}. "
            "Primero ejecuta entrenamiento guardado."
        )

    return json.loads(registry_path.read_text(encoding="utf-8"))
