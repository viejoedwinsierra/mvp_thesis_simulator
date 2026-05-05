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
# CONFIGURACION DE GENERACION
# ============================================================
START_DATE = date(2026, 4, 1)
DAYS_TO_GENERATE = 30
GLOBAL_SEED = 42


SCENARIOS = [
    "normal",
    "high_load",
    "low_capacity",
    "network_degraded",
    "large_files",
    "high_error",
]

SCENARIO_WEIGHTS = [0.34, 0.18, 0.16, 0.14, 0.10, 0.08]


# ============================================================
# TARGETS DE CALIBRACION
# ============================================================
MAX_REASONABLE_EFFECTIVE_SIZE_MB = 3500
DEFAULT_PEAK_LOAD_SHARE = 0.50
DEFAULT_PEAK_SLOT_HOURS = 6


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


def pick_scenario(rng: random.Random) -> str:
    return rng.choices(
        population=SCENARIOS,
        weights=SCENARIO_WEIGHTS,
        k=1,
    )[0]


def get_scenario_description(scenario: str) -> str:
    return {
        "normal": "Operational baseline day with moderate variation.",
        "high_load": "Higher daily volume with moderate capacity pressure.",
        "low_capacity": "Reduced hourly capacity to force controlled congestion.",
        "network_degraded": "Lower transfer speed and higher transfer variability.",
        "large_files": "Larger file sizes with controlled long-tail behavior.",
        "high_error": "Higher base error probability without extreme congestion.",
    }.get(scenario, "Synthetic scenario.")


def estimate_peak_hour_load(max_valid_files_per_day: int) -> float:
    return (
        max_valid_files_per_day
        * DEFAULT_PEAK_LOAD_SHARE
        / DEFAULT_PEAK_SLOT_HOURS
    )


def compute_capacity_from_peak_load(
    max_valid_files_per_day: int,
    scenario: str,
    rng: random.Random,
) -> int:
    estimated_peak_hour = estimate_peak_hour_load(max_valid_files_per_day)

    capacity_factor_by_scenario = {
        "normal": (1.05, 1.30),
        "high_load": (0.80, 1.00),
        "low_capacity": (0.50, 0.70),
        "network_degraded": (0.75, 0.95),
        "large_files": (0.90, 1.15),
        "high_error": (0.95, 1.20),
    }

    min_factor, max_factor = capacity_factor_by_scenario.get(
        scenario,
        (0.90, 1.15),
    )

    return max(
        int(estimated_peak_hour * rng.uniform(min_factor, max_factor)),
        1,
    )


def apply_scenario_metadata(config: dict, scenario: str) -> None:
    config["scenario"] = {
        "name": scenario,
        "description": get_scenario_description(scenario),
    }


# ============================================================
# VARIACION DE BLOQUES
# ============================================================
def vary_simulation(
    config: dict,
    current_date: date,
    scenario: str,
    rng: random.Random,
) -> None:
    config["simulation"]["simulation_date"] = current_date.strftime("%Y-%m-%d")
    config["simulation"]["seed"] = rng.randint(1, 999999)

    base_files = config["simulation"]["max_valid_files_per_day"]

    load_ranges = {
        "normal": (0.92, 1.08),
        "high_load": (1.18, 1.38),
        "low_capacity": (0.98, 1.18),
        "network_degraded": (0.92, 1.10),
        "large_files": (0.85, 1.05),
        "high_error": (0.95, 1.12),
    }

    min_factor, max_factor = load_ranges.get(scenario, (0.90, 1.15))

    config["simulation"]["max_valid_files_per_day"] = int(
        base_files * random_multiplier(rng, min_factor, max_factor)
    )


def vary_arrival_process(config: dict, scenario: str, rng: random.Random) -> None:
    if "arrival_process" not in config:
        return

    base_noise = config["arrival_process"].get("hourly_noise", 0.12)

    noise_ranges = {
        "normal": (0.75, 1.05),
        "high_load": (1.05, 1.30),
        "low_capacity": (1.00, 1.25),
        "network_degraded": (0.95, 1.20),
        "large_files": (0.80, 1.10),
        "high_error": (0.85, 1.15),
    }

    min_factor, max_factor = noise_ranges.get(scenario, (0.80, 1.20))

    config["arrival_process"]["strategy"] = "poisson"
    config["arrival_process"]["hourly_noise"] = round(
        clamp(
            base_noise * random_multiplier(rng, min_factor, max_factor),
            0.05,
            0.22,
        ),
        4,
    )


def vary_capacity(config: dict, scenario: str, rng: random.Random) -> None:
    if "capacity" not in config:
        return

    capacity = config["capacity"]
    capacity["enabled"] = True

    max_valid_files_per_day = config["simulation"]["max_valid_files_per_day"]

    capacity["files_per_hour"] = compute_capacity_from_peak_load(
        max_valid_files_per_day=max_valid_files_per_day,
        scenario=scenario,
        rng=rng,
    )

    if scenario == "low_capacity":
        duration_range = (1.10, 1.35)
        error_range = (1.05, 1.25)
    elif scenario == "network_degraded":
        duration_range = (1.05, 1.25)
        error_range = (1.00, 1.20)
    elif scenario == "high_load":
        duration_range = (0.95, 1.20)
        error_range = (0.95, 1.20)
    elif scenario == "high_error":
        duration_range = (0.85, 1.10)
        error_range = (1.05, 1.25)
    else:
        duration_range = (0.85, 1.10)
        error_range = (0.85, 1.10)

    capacity["duration_penalty_factor"] = round(
        clamp(
            capacity.get("duration_penalty_factor", 0.55)
            * random_multiplier(rng, *duration_range),
            0.25,
            0.85,
        ),
        4,
    )

    capacity["error_penalty_factor"] = round(
        clamp(
            capacity.get("error_penalty_factor", 0.35)
            * random_multiplier(rng, *error_range),
            0.15,
            0.75,
        ),
        4,
    )


def vary_outliers(config: dict, scenario: str, rng: random.Random) -> None:
    if "outliers" not in config:
        return

    outliers = config["outliers"]
    outliers["enabled"] = True

    if scenario == "large_files":
        probability_factor = random_multiplier(rng, 1.15, 1.75)
        min_mult = random_multiplier(rng, 2.0, 3.0)
        max_mult = random_multiplier(rng, 5.0, 8.5)
    elif scenario == "high_load":
        probability_factor = random_multiplier(rng, 0.90, 1.35)
        min_mult = random_multiplier(rng, 1.8, 2.8)
        max_mult = random_multiplier(rng, 4.0, 7.0)
    else:
        probability_factor = random_multiplier(rng, 0.60, 1.20)
        min_mult = random_multiplier(rng, 1.5, 2.4)
        max_mult = random_multiplier(rng, 3.5, 6.5)

    outliers["probability"] = round(
        clamp(
            outliers.get("probability", 0.02) * probability_factor,
            0.005,
            0.035,
        ),
        5,
    )

    outliers["size_multiplier_min"] = round(min_mult, 2)
    outliers["size_multiplier_max"] = round(max(max_mult, min_mult + 0.75), 2)
    outliers["max_size_mb"] = int(outliers.get("max_size_mb", 1200))


def vary_file_sizes(config: dict, scenario: str, rng: random.Random) -> None:
    size_distributions = config["file_types"]["size_distribution_mb"]

    max_sigma_by_type = {
        "json": 0.90,
        "jpg": 0.85,
        "pdf": 1.00,
        "mp4": 1.05,
    }

    for file_type, params in size_distributions.items():
        if scenario == "large_files":
            mean_range = (1.05, 1.22)
            sigma_range = (0.95, 1.12)
        elif scenario == "high_load":
            mean_range = (0.95, 1.10)
            sigma_range = (0.90, 1.08)
        else:
            mean_range = (0.90, 1.10)
            sigma_range = (0.88, 1.08)

        params["mean"] = round(
            clamp(
                params["mean"] * random_multiplier(rng, *mean_range),
                params["min"],
                params["max"],
            ),
            4,
        )

        params["sigma"] = round(
            clamp(
                params["sigma"] * random_multiplier(rng, *sigma_range),
                0.20,
                max_sigma_by_type.get(file_type, 1.0),
            ),
            4,
        )

        if file_type == "mp4":
            if scenario == "large_files":
                params["max"] = int(rng.choice([600, 650, 700]))
            else:
                params["max"] = int(rng.choice([450, 500, 550]))


def calibrate_outlier_effective_size(config: dict) -> None:
    outliers = config.get("outliers")
    if not outliers or not outliers.get("enabled", False):
        return

    max_multiplier = outliers.get("size_multiplier_max", 1.0)
    size_distributions = config["file_types"]["size_distribution_mb"]

    for _, params in size_distributions.items():
        effective_max = params["max"] * max_multiplier

        if effective_max <= MAX_REASONABLE_EFFECTIVE_SIZE_MB:
            continue

        allowed_multiplier = MAX_REASONABLE_EFFECTIVE_SIZE_MB / params["max"]

        outliers["size_multiplier_max"] = round(
            clamp(
                allowed_multiplier,
                outliers["size_multiplier_min"] + 0.5,
                outliers["size_multiplier_max"],
            ),
            2,
        )


def vary_transfer(config: dict, scenario: str, rng: random.Random) -> None:
    speed = config["transfer"]["speed_mbps"]

    if scenario == "network_degraded":
        mean_range = (0.55, 0.78)
        sigma_range = (1.05, 1.25)
    elif scenario == "low_capacity":
        mean_range = (0.75, 0.98)
        sigma_range = (0.98, 1.15)
    elif scenario == "large_files":
        mean_range = (0.80, 1.05)
        sigma_range = (0.95, 1.12)
    else:
        mean_range = (0.85, 1.12)
        sigma_range = (0.90, 1.10)

    speed["mean"] = round(
        clamp(
            speed["mean"] * random_multiplier(rng, *mean_range),
            speed["min"],
            speed["max"],
        ),
        4,
    )

    speed["sigma"] = round(
        clamp(
            speed["sigma"] * random_multiplier(rng, *sigma_range),
            0.45,
            0.95,
        ),
        4,
    )

    speed["max"] = int(clamp(speed.get("max", 300), 150, 350))


def vary_errors(config: dict, scenario: str, rng: random.Random) -> None:
    errors = config["errors"]
    multipliers = errors.get("multipliers", {})

    if scenario == "high_error":
        base_range = (1.25, 1.75)
    elif scenario == "network_degraded":
        base_range = (1.00, 1.30)
    elif scenario == "low_capacity":
        base_range = (0.95, 1.25)
    else:
        base_range = (0.75, 1.15)

    errors["base_error_probability"] = round(
        clamp(
            errors["base_error_probability"] * random_multiplier(rng, *base_range),
            0.004,
            0.040,
        ),
        5,
    )

    if "large_file" in multipliers and isinstance(multipliers["large_file"], dict):
        multipliers["large_file"]["multiplier"] = round(
            random_multiplier(rng, 1.8, 2.8),
            4,
        )

    if "low_speed" in multipliers and isinstance(multipliers["low_speed"], dict):
        multipliers["low_speed"]["multiplier"] = round(
            random_multiplier(rng, 2.0, 3.2),
            4,
        )

    if "long_transfer" in multipliers and isinstance(multipliers["long_transfer"], dict):
        if scenario in ("network_degraded", "large_files", "low_capacity"):
            threshold_range = (60, 120)
            multiplier_range = (1.7, 2.6)
        else:
            threshold_range = (90, 160)
            multiplier_range = (1.3, 2.0)

        multipliers["long_transfer"]["threshold_sec"] = round(
            random_multiplier(rng, *threshold_range),
            2,
        )

        multipliers["long_transfer"]["multiplier"] = round(
            random_multiplier(rng, *multiplier_range),
            4,
        )

    if "archive_tier" in multipliers:
        multipliers["archive_tier"] = round(random_multiplier(rng, 1.1, 1.6), 4)

    if "peak_hour" in multipliers:
        multipliers["peak_hour"] = round(random_multiplier(rng, 1.2, 1.6), 4)

    if "movement_storage" in multipliers:
        multipliers["movement_storage"] = round(random_multiplier(rng, 1.05, 1.45), 4)

    if "congestion" in multipliers:
        if scenario == "low_capacity":
            multipliers["congestion"] = round(random_multiplier(rng, 1.7, 2.4), 4)
        elif scenario == "high_load":
            multipliers["congestion"] = round(random_multiplier(rng, 1.5, 2.2), 4)
        elif scenario == "network_degraded":
            multipliers["congestion"] = round(random_multiplier(rng, 1.4, 2.1), 4)
        else:
            multipliers["congestion"] = round(random_multiplier(rng, 1.2, 1.8), 4)


def add_validation_targets(config: dict) -> None:
    max_files = config["simulation"]["max_valid_files_per_day"]
    capacity = config.get("capacity", {}).get("files_per_hour", 0)
    estimated_peak = estimate_peak_hour_load(max_files)

    expected_queue_pressure = None
    if capacity > 0:
        expected_queue_pressure = estimated_peak / capacity

    config["validation_targets"] = {
        "estimated_peak_hour_load": round(estimated_peak, 4),
        "expected_peak_queue_pressure": (
            round(expected_queue_pressure, 4)
            if expected_queue_pressure is not None
            else None
        ),
        "notes": (
            "Use these values to compare generated hourly_arrival_count, "
            "queue_pressure and congestion_factor in descriptive analysis."
        ),
    }


def vary_config(config: dict, current_date: date, rng: random.Random) -> dict:
    config = copy.deepcopy(config)
    scenario = pick_scenario(rng)

    apply_scenario_metadata(config, scenario)

    vary_simulation(config, current_date, scenario, rng)
    vary_arrival_process(config, scenario, rng)
    vary_capacity(config, scenario, rng)
    vary_outliers(config, scenario, rng)
    vary_file_sizes(config, scenario, rng)
    calibrate_outlier_effective_size(config)
    vary_transfer(config, scenario, rng)
    vary_errors(config, scenario, rng)
    add_validation_targets(config)

    return config


# ============================================================
# GENERACION DE ARCHIVOS
# ============================================================
def generate_configs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Base config: {BASE_CONFIG_PATH}")
    print(f"Output dir : {OUTPUT_DIR}")

    if not BASE_CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"No existe el archivo base de configuracion: {BASE_CONFIG_PATH}"
        )

    with BASE_CONFIG_PATH.open("r", encoding="utf-8") as file:
        base_config = json.load(file)

    rng = random.Random(GLOBAL_SEED)

    for index in range(DAYS_TO_GENERATE):
        current_date = START_DATE + timedelta(days=index)
        config = vary_config(base_config, current_date, rng)

        output_file = OUTPUT_DIR / f"simulation_config_{current_date}.json"

        with output_file.open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2, ensure_ascii=False)

        targets = config.get("validation_targets", {})

        print(
            f"Creado: {output_file.name} | "
            f"scenario={config['scenario']['name']} | "
            f"files={config['simulation']['max_valid_files_per_day']} | "
            f"capacity={config.get('capacity', {}).get('files_per_hour')} | "
            f"expected_qp={targets.get('expected_peak_queue_pressure')} | "
            f"error_base={config['errors']['base_error_probability']}"
        )

    print("Proceso finalizado.")


if __name__ == "__main__":
    generate_configs()
