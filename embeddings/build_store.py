import argparse
import gc
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_ROOT = REPOSITORY_ROOT / "data" / "embeddings"
DEFAULT_SPECS = {
    "seq_train": (EMBEDDING_ROOT / "seq_train.pkl", 1280),
    "seq_lt30": (EMBEDDING_ROOT / "seq_lt30.pkl", 1280),
    "seq_30_50": (EMBEDDING_ROOT / "seq_30_50.pkl", 1280),
    "str_train": (EMBEDDING_ROOT / "str_train.pkl", 512),
    "str_lt30": (EMBEDDING_ROOT / "str_lt30.pkl", 512),
    "str_30_50": (EMBEDDING_ROOT / "str_30_50.pkl", 512),
}


def as_matrix(value, expected_dim):
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if (array.ndim not in (2, 3) or array.shape[-1] != expected_dim
            or (array.ndim == 3 and array.shape[1] != 1) or array.size == 0):
        raise ValueError(
            f"Feature with shape {array.shape} cannot be reshaped to (-1, {expected_dim})"
        )
    return array.reshape(-1, expected_dim)


def build_store(name, source, expected_dim, output_root, overwrite=False):
    source = Path(source)
    destination = output_root / name
    values_path = destination / "values.npy"
    index_path = destination / "index.csv"
    metadata_path = destination / "metadata.json"

    if values_path.exists() and index_path.exists() and not overwrite:
        if not metadata_path.exists() or not source.exists():
            raise ValueError(f"{name}: cannot verify existing store; supply its source and rebuild with --overwrite")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        signature = (str(source), source.stat().st_size, source.stat().st_mtime_ns, expected_dim)
        saved = tuple(metadata.get(key) for key in (
            "source", "source_size_bytes", "source_mtime_ns", "feature_dim"
        ))
        if signature != saved:
            raise ValueError(f"{name}: source changed; rebuild the store with --overwrite")
        print(f"[skip] {name}: store already exists at {destination}", flush=True)
        return

    destination.mkdir(parents=True, exist_ok=True)
    print(f"[load] {name}: {source}", flush=True)
    with source.open("rb") as handle:
        features = pickle.load(handle)

    entries = []
    total_rows = 0
    source_dtypes = set()
    seen_ids = set()
    for protein_id, value in features.items():
        if str(protein_id) in seen_ids:
            raise ValueError(f"{name}: duplicate canonical protein ID {protein_id}")
        seen_ids.add(str(protein_id))
        matrix = as_matrix(value, expected_dim)
        if not np.isfinite(matrix).all() or np.any(np.abs(matrix) > np.finfo(np.float16).max):
            raise ValueError(f"{name}: invalid or float16-overflowing features for {protein_id}")
        entries.append((protein_id, str(protein_id), total_rows, matrix.shape[0]))
        total_rows += matrix.shape[0]
        source_dtypes.add(str(matrix.dtype))

    temporary_values = destination / "values.tmp.npy"
    mmap = np.lib.format.open_memmap(
        temporary_values,
        mode="w+",
        dtype=np.float16,
        shape=(total_rows, expected_dim),
    )
    for row_number, (source_key, protein_id, offset, length) in enumerate(entries, start=1):
        matrix = as_matrix(features[source_key], expected_dim)
        mmap[offset : offset + length] = matrix.astype(np.float16, copy=False)
        if row_number % 1000 == 0 or row_number == len(entries):
            print(
                f"[write] {name}: {row_number}/{len(entries)} proteins",
                flush=True,
            )
    mmap.flush()
    del mmap
    os.replace(temporary_values, values_path)

    pd.DataFrame(
        [(protein_id, offset, length) for _, protein_id, offset, length in entries],
        columns=["protein_id", "offset", "length"],
    ).to_csv(index_path, index=False)
    metadata = {
        "name": name,
        "source": str(source),
        "source_size_bytes": source.stat().st_size,
        "source_mtime_ns": source.stat().st_mtime_ns,
        "protein_count": len(entries),
        "total_rows": total_rows,
        "feature_dim": expected_dim,
        "stored_dtype": "float16",
        "source_dtypes": sorted(source_dtypes),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"[done] {name}: {len(entries)} proteins, {total_rows} rows",
        flush=True,
    )

    del features
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default=REPOSITORY_ROOT / "data" / "embeddings" / "feature_store",
    )
    parser.add_argument(
        "--stores",
        nargs="+",
        choices=sorted(DEFAULT_SPECS),
        default=list(DEFAULT_SPECS),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for name in args.stores:
        source, expected_dim = DEFAULT_SPECS[name]
        build_store(
            name,
            source,
            expected_dim,
            output_root,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
