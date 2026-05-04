import copy
import json
import random
from datetime import date, timedelta
from pathlib import Path


# ============================================================
# RUTAS DEL PROYECTO
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent

CONFIG_DIR = BASE_DIR / "config"
OUTPUT_DIR = BASE_DIR / "config"

BASE_CONFIG_PATH = CONFIG_DIR / "simulation_config.json"


# ============================================================
# CONFIGURACIÓN DE GENERACIÓN
# ============================================================
START_DATE = date(2026, 4, 1)
DAYS_TO_GENERATE = 30
GLOBAL_SEED = 42


# ============================================================
# UTILIDADES
# ============================================================
def clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def random_multiplier(
    rng: random.Random,
    min_factor: float,
    max_factor: float,
) -> float:
    return rng.uniform(min_factor, max_factor)


# ============================================================
# VARIACIÓN ALEATORIA CONTROLADA
# ============================================================
def vary_config(config: dict, current_date: date, rng: random.Random) -> dict:
    config = copy.deepcopy(config)
    date_str = current_date.strftime("%Y-%m-%d")

    # =========================
    # Simulation
    # =========================
    config["simulation"]["simulation_date"] = date_str
    config["simulation"]["seed"] = rng.randint(1, 999999)

    base_files = config["simulation"]["max_valid_files_per_day"]
    config["simulation"]["max_valid_files_per_day"] = int(
        base_files * random_multiplier(rng, 0.85, 1.20)
    )

    # =========================
    # Arrival process
    # =========================
    if "arrival_process" in config:
        config["arrival_process"]["hourly_noise"] = round(
            clamp(
                config["arrival_process"].get("hourly_noise", 0.10)
                * random_multiplier(rng, 0.70, 1.50),
                0.02,
                0.30,
            ),
            4,
        )

    # =========================
    # Capacity
    # =========================
    if "capacity" in config and config["capacity"].get("enabled", False):
        base_capacity = config["capacity"]["files_per_hour"]

        config["capacity"]["files_per_hour"] = int(
            base_capacity * random_multiplier(rng, 0.80, 1.25)
        )

        config["capacity"]["duration_penalty_factor"] = round(
            clamp(
                config["capacity"].get("duration_penalty_factor", 0.25)
                * random_multiplier(rng, 0.75, 1.40),
                0.05,
                0.60,
            ),
            4,
        )

        config["capacity"]["error_penalty_factor"] = round(
            clamp(
                config["capacity"].get("error_penalty_factor", 0.50)
                * random_multiplier(rng, 0.75, 1.50),
                0.10,
                1.20,
            ),
            4,
        )

    # =========================
    # Outliers
    # =========================
    if "outliers" in config and config["outliers"].get("enabled", False):
        config["outliers"]["probability"] = round(
            clamp(
                config["outliers"].get("probability", 0.005)
                * random_multiplier(rng, 0.50, 2.00),
                0.001,
                0.02,
            ),
            5,
        )

        config["outliers"]["size_multiplier_min"] = round(
            random_multiplier(rng, 1.5, 3.0),
            2,
        )

        config["outliers"]["size_multiplier_max"] = round(
            random_multiplier(rng, 4.0, 8.0),
            2,
        )

        if (
            config["outliers"]["size_multiplier_max"]
            <= config["outliers"]["size_multiplier_min"]
        ):
            config["outliers"]["size_multiplier_max"] = (
                config["outliers"]["size_multiplier_min"] + 2
            )

    # =========================
    # File size distributions
    # =========================
    size_distributions = config["file_types"]["size_distribution_mb"]

    for _, params in size_distributions.items():
        params["mean"] = round(
            clamp(
                params["mean"] * random_multiplier(rng, 0.85, 1.20),
                params["min"],
                params["max"],
            ),
            4,
        )

        params["sigma"] = round(
            clamp(
                params["sigma"] * random_multiplier(rng, 0.90, 1.20),
                0.20,
                2.00,
            ),
            4,
        )

    # =========================
    # Transfer speed
    # =========================
    speed = config["transfer"]["speed_mbps"]

    speed["mean"] = round(
        clamp(
            speed["mean"] * random_multiplier(rng, 0.80, 1.25),
            speed["min"],
            speed["max"],
        ),
        4,
    )

    speed["sigma"] = round(
        clamp(
            speed["sigma"] * random_multiplier(rng, 0.85, 1.30),
            0.20,
            2.00,
        ),
        4,
    )

    # =========================
    # Error model
    # =========================
    errors = config["errors"]

    errors["base_error_probability"] = round(
        clamp(
            errors["base_error_probability"]
            * random_multiplier(rng, 0.60, 1.80),
            0.001,
            0.08,
        ),
        5,
    )

    multipliers = errors.get("multipliers", {})

    for key, value in multipliers.items():
        if isinstance(value, dict) and "multiplier" in value:
            value["multiplier"] = round(
                clamp(
                    value["multiplier"]
                    * random_multiplier(rng, 0.80, 1.30),
                    1.0,
                    6.0,
                ),
                4,
            )

        elif isinstance(value, (int, float)):
            multipliers[key] = round(
                clamp(
                    value * random_multiplier(rng, 0.80, 1.30),
                    1.0,
                    5.0,
                ),
                4,
            )

    return config


# ============================================================
# GENERACIÓN DE ARCHIVOS
# ============================================================
def generate_configs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Base config: {BASE_CONFIG_PATH}")
    print(f"Output dir : {OUTPUT_DIR}")

    if not BASE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"No existe el archivo base de configuración: {BASE_CONFIG_PATH}"
        )

    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as file:
        base_config = json.load(file)

    rng = random.Random(GLOBAL_SEED)

    for index in range(DAYS_TO_GENERATE):
        current_date = START_DATE + timedelta(days=index)

        config = vary_config(
            config=base_config,
            current_date=current_date,
            rng=rng,
        )

        output_file = OUTPUT_DIR / f"simulation_config_{current_date}.json"

        with output_file.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, ensure_ascii=False)

        print(f"Creado: {output_file}")

    print("Proceso finalizado.")


if __name__ == "__main__":
    generate_configs()