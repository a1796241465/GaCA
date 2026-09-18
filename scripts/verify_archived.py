"""Lightweight validation of archived GaCA experiment artifacts."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch

from common.protocol import EXPECTED_COUNTS, validate_reference_tables, validate_split
from evaluation.paired_statistics import load_joined
from gaca.model import GaCA


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ACCURACY = {"lt30": 178 / 243, "30_50": 405 / 477}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prediction_accuracy(path, expected_samples):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"protein_id", "true_ec", "predicted_ec"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} lacks required prediction columns")
    ids = [row["protein_id"] for row in rows]
    if len(rows) != expected_samples or len(set(ids)) != len(ids):
        raise ValueError(f"{path} has an unexpected sample count or duplicate IDs")
    return sum(row["true_ec"] == row["predicted_ec"] for row in rows) / len(rows)


def verify(root=ROOT):
    metadata = root / "metadata"
    references = root / "reference_results"
    checkpoint = root / "checkpoints" / "gaca_73.25.pth"

    tables = {
        split: pd.read_csv(metadata / f"effective_{split}.csv", dtype=str)
        for split in ("train", "lt30", "30_50")
    }
    validate_reference_tables(tables)
    split = pd.read_csv(metadata / "fixed_fit_validation_split.csv", dtype=str)
    validate_split(split, tables["train"]["Entry"])
    split_counts = split["split"].value_counts().to_dict()
    if split_counts != {
        "fit": EXPECTED_COUNTS["fit"],
        "validation": EXPECTED_COUNTS["validation"],
    }:
        raise ValueError(f"Unexpected fixed-split counts: {split_counts}")

    expected_hash = (root / "checkpoints" / "SHA256SUMS").read_text(
        encoding="utf-8"
    ).split()[0]
    observed_hash = sha256(checkpoint)
    if observed_hash != expected_hash:
        raise ValueError("Archived checkpoint SHA256 does not match SHA256SUMS")
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model = GaCA(num_classes=3811, hidden_dim=512, dropout=0.5)
    model.load_state_dict(state, strict=True)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count != 4_156_645:
        raise ValueError(f"Unexpected GaCA parameter count: {parameter_count}")

    metrics = json.loads(
        (references / "neural" / "gaca" / "metrics.json").read_text(encoding="utf-8")
    )
    accuracies = {}
    for dataset, expected_samples in (("lt30", 243), ("30_50", 477)):
        path = references / "neural" / "gaca" / f"predictions_{dataset}.csv"
        accuracy = prediction_accuracy(path, expected_samples)
        archived = float(metrics["test_results"][dataset]["level_4_accuracy"])
        if abs(accuracy - archived) > 1e-12 or abs(accuracy - REFERENCE_ACCURACY[dataset]) > 1e-12:
            raise ValueError(f"Archived {dataset} predictions and metrics disagree")
        accuracies[dataset] = accuracy

    train_counts = tables["train"]["EC number"].value_counts().to_dict()
    for dataset in ("lt30", "30_50"):
        joined = load_joined(
            dataset,
            references / "neural" / "gaca",
            references / "retrieval" / "blastp",
            references / "retrieval" / "foldseek",
            train_counts,
        )
        if len(joined) != EXPECTED_COUNTS[dataset]:
            raise ValueError(f"Unexpected paired row count for {dataset}")

    return {
        "scope": "archived experiment artifacts; not a fresh end-to-end reproduction",
        "checkpoint_sha256": observed_hash,
        "trainable_parameters": parameter_count,
        "fixed_split": split_counts,
        "level_4_accuracy": accuracies,
        "paired_inputs": {"lt30": 243, "30_50": 477},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Validate archived GaCA artifacts without training or downloading data."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("This verifies the archived experiment artifacts; it is not a fresh end-to-end reproduction.")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
