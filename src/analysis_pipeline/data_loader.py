from pathlib import Path
import pandas as pd


DATASET_FILE_PATTERN = "blob_inventory*.csv"


def _is_valid_scenario_dataset_file(file: Path, dataset_root: Path) -> bool:
    """
    Acepta solamente:
      output/dataset/<escenario>/blob_inventory_<escenario>_<fecha>.csv

    Rechaza:
      output/dataset/<escenario>/dataset/blob_inventory.csv
      cualquier blob_inventory dentro de subcarpetas internas.
    """
    try:
        relative = file.relative_to(dataset_root)
    except ValueError:
        return False

    if len(relative.parts) != 2:
        return False

    if relative.parts[1] == "blob_inventory.csv":
        return False

    return file.name.startswith("blob_inventory_") and file.suffix.lower() == ".csv"


def load_datasets_from_folder(dataset_dir: str | Path = "output/dataset") -> pd.DataFrame:
    """
    Carga datasets finales por escenario.

    Regla estricta:
      solo lee CSV directos dentro de output/dataset/<escenario>/.
    """
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe dataset_dir: {dataset_path}")

    files = [
        file
        for file in sorted(dataset_path.rglob(DATASET_FILE_PATTERN))
        if _is_valid_scenario_dataset_file(file, dataset_path)
    ]

    files = list(dict.fromkeys(files))

    if not files:
        raise FileNotFoundError(
            "No se encontraron datasets finales.\n\n"
            "Estructura esperada:\n"
            "output/dataset/<escenario>/blob_inventory_<escenario>_<fecha>.csv\n\n"
            "No se leerán archivos internos como:\n"
            "output/dataset/<escenario>/dataset/blob_inventory.csv"
        )

    frames = []

    for file in files:
        df_part = pd.read_csv(file)
        df_part["source_file"] = file.name
        df_part["source_path"] = str(file)

        relative = file.relative_to(dataset_path)
        scenario_name = relative.parts[0]

        df_part["scenario_name"] = scenario_name
        frames.append(df_part)

    print("\n=== ARCHIVOS DATASET DETECTADOS ===")
    for file in files:
        print(file.resolve())

    return pd.concat(frames, ignore_index=True)


def load_datasets_by_scenario(dataset_dir: str | Path = "output/dataset") -> dict[str, pd.DataFrame]:
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        raise FileNotFoundError(f"No existe dataset_dir: {dataset_path}")

    result = {}

    for scenario_dir in sorted(p for p in dataset_path.iterdir() if p.is_dir()):
        files = [
            file
            for file in sorted(scenario_dir.glob(DATASET_FILE_PATTERN))
            if file.is_file()
            and file.name != "blob_inventory.csv"
            and file.name.startswith("blob_inventory_")
        ]

        if not files:
            continue

        frames = []

        for file in files:
            df_part = pd.read_csv(file)
            df_part["source_file"] = file.name
            df_part["source_path"] = str(file)
            df_part["scenario_name"] = scenario_dir.name
            frames.append(df_part)

        result[scenario_dir.name] = pd.concat(frames, ignore_index=True)

    if not result:
        raise FileNotFoundError(
            f"No se encontraron escenarios con datasets finales en {dataset_path}"
        )

    return result
