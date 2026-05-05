from pathlib import Path
import pandas as pd


def load_datasets_from_folder(dataset_dir: str = "output/dataset") -> pd.DataFrame:
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
            files.extend(sorted(folder.glob("blob_inventory*.csv")))

    files = list(dict.fromkeys(files))

    if not files:
        searched = "\n".join(str(p.resolve()) for p in candidate_dirs)
        raise FileNotFoundError(
            "No se encontraron archivos blob_inventory*.csv.\n\n"
            "Carpetas revisadas:\n"
            f"{searched}\n\n"
            "Solución: guarda al menos un archivo con nombre tipo "
            "blob_inventory.csv o blob_inventory_run_001.csv en output/dataset."
        )

    frames = []

    for file in files:
        df_part = pd.read_csv(file)
        df_part["source_file"] = file.name
        frames.append(df_part)

    print("\n=== ARCHIVOS DATASET DETECTADOS ===")
    for file in files:
        print(file.resolve())

    return pd.concat(frames, ignore_index=True)