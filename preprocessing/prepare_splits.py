"""Prepare the CARE-derived metadata tables used by the GaCA experiments."""

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"Entry", "Sequence", "EC number"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--care-task-dir", type=Path, required=True)
    parser.add_argument("--train-structures", type=Path, required=True)
    parser.add_argument("--lt30-structures", type=Path, required=True)
    parser.add_argument("--30-50-structures", dest="mid_structures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--max-length", type=int, default=600)
    parser.add_argument("--max-per-ec", type=int, default=25)
    parser.add_argument("--min-confidence", type=float, default=70.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_split(path):
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=["Entry", "Sequence", "EC number"]).copy()
    frame["Entry"] = frame["Entry"].astype(str)
    frame["Sequence"] = frame["Sequence"].astype(str)
    frame["EC number"] = frame["EC number"].astype(str)
    if "Length" not in frame:
        frame["Length"] = frame["Sequence"].str.len()
    return frame


def sample_training(frame, max_length, max_per_ec, seed):
    frame = frame.loc[frame["Length"] <= max_length].copy()
    groups = []
    for ec_label in frame["EC number"].drop_duplicates():
        group = frame.loc[frame["EC number"] == ec_label]
        if len(group) > max_per_ec:
            group = group.sample(n=max_per_ec, random_state=seed)
        groups.append(group)
    sampled = pd.concat(groups, ignore_index=True)
    return sampled.drop_duplicates(subset="Entry", keep="first").reset_index(drop=True)


def available_structure_ids(structure_dir):
    return {path.stem for path in structure_dir.glob("*.pdb")}


def confidence_ids(manifest_path, threshold):
    manifest = pd.read_csv(manifest_path)
    id_column = next(
        (name for name in ("protein_id", "uniprot_id", "Entry") if name in manifest),
        None,
    )
    if id_column is None or "confidence" not in manifest:
        raise ValueError(
            "The structure manifest must contain a protein ID column and confidence."
        )
    confidence = pd.to_numeric(manifest["confidence"], errors="coerce")
    return set(manifest.loc[confidence >= threshold, id_column].astype(str))


def retain_available(frame, structure_dir, allowed_ids=None):
    ids = available_structure_ids(structure_dir)
    if allowed_ids is not None:
        ids &= allowed_ids
    return frame.loc[frame["Entry"].isin(ids)].copy().reset_index(drop=True)


def main():
    args = parse_args()
    task = args.care_task_dir
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    train_source = read_split(task / "protein_train50.csv")
    train_sampled = sample_training(
        train_source, args.max_length, args.max_per_ec, args.seed
    )
    high_confidence = None
    if args.train_manifest:
        high_confidence = confidence_ids(args.train_manifest, args.min_confidence)
    train = retain_available(train_sampled, args.train_structures, high_confidence)

    lt30_source = read_split(task / "30_protein_test.csv")
    lt30 = retain_available(lt30_source, args.lt30_structures)
    mid_source = read_split(task / "30-50_protein_test.csv")
    mid = retain_available(mid_source, args.mid_structures)

    paths = {
        "train": output / "train_cleaned_with_structure.csv",
        "lt30": output / "test_30_cleaned_with_structure.csv",
        "30_50": output / "test_30_50_clean.csv",
    }
    train.to_csv(paths["train"], index=False)
    lt30.to_csv(paths["lt30"], index=False)
    mid.to_csv(paths["30_50"], index=False)

    summary = {
        "settings": {
            "max_length": args.max_length,
            "max_per_level4_ec": args.max_per_ec,
            "seed": args.seed,
            "minimum_train_structure_confidence": (
                args.min_confidence if args.train_manifest else None
            ),
        },
        "counts": {
            "care_train50_rows": len(train_source),
            "sampled_training_rows": len(train_sampled),
            "training_rows_with_structure": len(train),
            "lt30_rows_with_structure": len(lt30),
            "30_50_rows_with_structure": len(mid),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    (output / "preprocessing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()