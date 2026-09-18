"""Download AlphaFold DB structures for protein identifiers in a metadata table."""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

API_TEMPLATE = "https://alphafold.ebi.ac.uk/api/prediction/{protein_id}"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--id-column", default="Entry")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.05)
    return parser.parse_args()


def request_json(session, url, timeout, retries):
    last_error = None
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 404:
                return None, "not_found"
            response.raise_for_status()
            return response.json(), "ok"
        except requests.RequestException as error:
            last_error = str(error)
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    return None, f"request_error: {last_error}"


def download_file(session, url, path, timeout, retries):
    last_error = None
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            temporary = path.with_suffix(path.suffix + ".part")
            temporary.write_bytes(response.content)
            temporary.replace(path)
            return "downloaded"
        except requests.RequestException as error:
            last_error = str(error)
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    return f"download_error: {last_error}"


def main():
    args = parse_args()
    table = pd.read_csv(args.csv)
    if args.id_column not in table:
        raise ValueError(f"Missing ID column {args.id_column!r} in {args.csv}")
    protein_ids = table[args.id_column].dropna().astype(str).drop_duplicates().tolist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame()
    if args.manifest.exists():
        existing = pd.read_csv(args.manifest)
    existing_by_id = {
        str(row["protein_id"]): row.to_dict()
        for _, row in existing.iterrows()
        if "protein_id" in existing
    }

    records = []
    with requests.Session() as session:
        session.headers.update({"User-Agent": "GaCA-reproducibility/1.0"})
        for protein_id in tqdm(protein_ids, desc="AlphaFold structures"):
            output_path = args.output_dir / f"{protein_id}.pdb"
            if output_path.exists() and output_path.stat().st_size > 0:
                old = existing_by_id.get(protein_id, {})
                records.append(
                    {
                        "protein_id": protein_id,
                        "status": "exists",
                        "confidence": old.get("confidence"),
                        "version": old.get("version"),
                        "pdb_url": old.get("pdb_url"),
                        "file": str(output_path),
                    }
                )
                continue

            payload, status = request_json(
                session,
                API_TEMPLATE.format(protein_id=protein_id),
                args.timeout,
                args.retries,
            )
            record = {"protein_id": protein_id, "status": status}
            if status == "ok" and isinstance(payload, list) and payload:
                entry = payload[0]
                pdb_url = entry.get("pdbUrl")
                record.update(
                    {
                        "confidence": entry.get("globalMetricValue"),
                        "version": entry.get("latestVersion"),
                        "pdb_url": pdb_url,
                    }
                )
                if pdb_url:
                    record["status"] = download_file(
                        session, pdb_url, output_path, args.timeout, args.retries
                    )
                    record["file"] = str(output_path)
                else:
                    record["status"] = "missing_pdb_url"
            records.append(record)
            pd.DataFrame(records).to_csv(args.manifest, index=False)
            if args.delay:
                time.sleep(args.delay)

    manifest = pd.DataFrame(records)
    manifest.to_csv(args.manifest, index=False)
    print(manifest["status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()