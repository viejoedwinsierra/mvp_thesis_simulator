    from __future__ import annotations

import os
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd


def safe_read_json(json_path: Path) -> tuple[Optional[Dict[str, Any]], bool]:
    try:
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f), True
    except Exception:
        return None, False


def file_hash_sha256(file_path: Path) -> Optional[str]:
    if not file_path.exists() or not file_path.is_file():
        return None
    sha = hashlib.sha256()
    try:
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()
    except Exception:
        return None


def extract_preview_200(metadata: Dict[str, Any]) -> str:
    """
    Intenta construir una vista resumida de 200 caracteres desde el JSON.
    """
    if not metadata:
        return ""

    candidate_fields = [
        "content_preview_200",
        "content_preview",
        "content_hash",
        "case_type",
        "document_id",
        "document_instance_id",
        "logical_creation_datetime",
    ]

    parts = []
    for field in candidate_fields:
        value = metadata.get(field)
        if value is not None:
            parts.append(f"{field}={value}")

    if "content_payload" in metadata and isinstance(metadata["content_payload"], dict):
        for k, v in metadata["content_payload"].items():
            parts.append(f"{k}={v}")

    text = " | ".join(parts)
    return text[:200]


def extract_hour(logical_creation_datetime: Optional[str]) -> Optional[int]:
    if not logical_creation_datetime:
        return None
    try:
        # espera algo como 2025-01-15T08:10:11
        return int(logical_creation_datetime[11:13])
    except Exception:
        return None


def build_record(
    root_path: Path,
    stem: str,
    pdf_path: Optional[Path],
    json_path: Optional[Path],
) -> Dict[str, Any]:
    pdf_exists = pdf_path is not None and pdf_path.exists()
    json_exists = json_path is not None and json_path.exists()

    metadata = None
    json_valid = False
    if json_exists:
        metadata, json_valid = safe_read_json(json_path)

    if pdf_exists and json_exists:
        pairing_status = "PAIR_OK" if json_valid else "PAIR_JSON_INVALID"
    elif json_exists and not pdf_exists:
        pairing_status = "JSON_WITHOUT_PDF"
    elif pdf_exists and not json_exists:
        pairing_status = "PDF_WITHOUT_JSON"
    else:
        pairing_status = "NOT_FOUND"

    simulation_date = None
    logical_creation_datetime = None
    case_group = None
    case_type = None
    is_error = None
    document_id = None
    document_instance_id = None
    parent_document_instance_id = None
    content_hash = None
    metadata_file_hash = None

    if metadata and json_valid:
        simulation_date = metadata.get("simulation_date")
        logical_creation_datetime = metadata.get("logical_creation_datetime")
        case_group = metadata.get("case_group")
        case_type = metadata.get("case_type")
        is_error = metadata.get("is_error")
        document_id = metadata.get("logical_document_id") or metadata.get("document_id")
        document_instance_id = metadata.get("document_instance_id")
        parent_document_instance_id = metadata.get("parent_document_instance_id")
        content_hash = metadata.get("content_hash")
        metadata_file_hash = metadata.get("file_hash")

    content_preview_200 = extract_preview_200(metadata or {})
    hour = extract_hour(logical_creation_datetime)

    return {
        "source_path": str(root_path),
        "file_stem": stem,
        "pdf_exists": pdf_exists,
        "json_exists": json_exists,
        "pairing_status": pairing_status,
        "simulation_date": simulation_date,
        "logical_creation_datetime": logical_creation_datetime,
        "hour": hour,
        "case_group": case_group,
        "case_type": case_type,
        "is_error": is_error,
        "document_id": document_id,
        "document_instance_id": document_instance_id,
        "parent_document_instance_id": parent_document_instance_id,
        "content_hash": content_hash,
        "metadata_file_hash": metadata_file_hash,
        "pdf_file_hash": file_hash_sha256(pdf_path) if pdf_exists else None,
        "content_preview_200": content_preview_200,
        "json_valid": json_valid,
        "pdf_path": str(pdf_path) if pdf_exists else None,
        "json_path": str(json_path) if json_exists else None,
    }


def scan_local_inventory(root_dir: str) -> pd.DataFrame:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"No existe la ruta: {root_dir}")

    grouped: Dict[str, Dict[str, Path]] = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix not in {".pdf", ".json"}:
            continue

        stem_key = str(path.with_suffix(""))
        grouped.setdefault(stem_key, {})
        grouped[stem_key][suffix] = path

    records: List[Dict[str, Any]] = []
    for stem_key, files in grouped.items():
        pdf_path = files.get(".pdf")
        json_path = files.get(".json")
        stem = Path(stem_key).name
        records.append(build_record(root, stem, pdf_path, json_path))

    df = pd.DataFrame(records)

    # Normalización útil para estadística
    if not df.empty:
        df["pdf_exists"] = df["pdf_exists"].astype(int)
        df["json_exists"] = df["json_exists"].astype(int)
        df["json_valid"] = df["json_valid"].astype(int)

        def normalize_error(v):
            if v is True:
                return 1
            if v is False:
                return 0
            if isinstance(v, (int, float)):
                return int(bool(v))
            return None

        df["is_error"] = df["is_error"].apply(normalize_error)

    return df


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Inventario local de PDFs y JSON para análisis estadístico")
    parser.add_argument("--input", required=True, help="Ruta local a analizar")
    parser.add_argument("--output-csv", required=True, help="Ruta del CSV de salida")
    args = parser.parse_args()

    df = scan_local_inventory(args.input)
    df.to_csv(args.output_csv, index=False, encoding="utf-8")

    print(f"CSV generado: {args.output_csv}")
    print(f"Total registros: {len(df)}")

    if not df.empty:
        print("\nResumen rápido:")
        print(df["pairing_status"].value_counts(dropna=False).to_string())
        if "case_type" in df.columns:
            print("\nTop case_type:")
            print(df["case_type"].value_counts(dropna=False).head(10).to_string())


if __name__ == "__main__":
    main()