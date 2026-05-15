from pathlib import Path
import pandas as pd


DATASET_FILE_PATTERN = "blob_inventory*.csv"


def load_datasets_from_folder(dataset_dir: str | Path = "output/dataset") -> pd.DataFrame:
    """
    Carga datasets blob_inventory*.csv.

    Cambio importante:
    - Ahora busca recursivamente dentro de output/dataset/<escenario>/.
    - Agrega scenario_name cuando el CSV esta dentro de una carpeta de escenario.
    """
    dataset_path = Path(dataset_dir)

    candidate_dirs = [
        dataset_path,
        Path("output/datasets"),
        Path("output"),
        Path("dataset"),
        Path("datasets"),
        Path("data"),
    ]

    files = []

    for folder in candidate_dirs:
        if folder.exists():
            files.extend(sorted(folder.rglob(DATASET_FILE_PATTERN)))

    files = list(dict.fromkeys(files))

    if not files:
        searched = "\n".join(str(p.resolve()) for p in candidate_dirs)
        raise FileNotFoundError(
            "No se encontraron archivos blob_inventory*.csv.\n\n"
            "Carpetas revisadas:\n"
            f"{searched}\n\n"
            "Solución: guarda archivos con nombre tipo "
            "blob_inventory_<escenario>_<fecha>.csv en output/dataset/<escenario>."
        )

    frames = []

    for file in files:
        df_part = pd.read_csv(file)
        df_part["source_file"] = file.name
        df_part["source_path"] = str(file)

        # Si esta dentro de output/dataset/<escenario>, tomar el nombre de carpeta.
        try:
            relative = file.relative_to(dataset_path)
            scenario_name = relative.parts[0] if len(relative.parts) > 1 else "root_dataset"
        except ValueError:
            scenario_name = file.parent.name

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
        files = sorted(scenario_dir.glob(DATASET_FILE_PATTERN))

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
            f"No se encontraron escenarios con {DATASET_FILE_PATTERN} en {dataset_path}"
        )

    return result
