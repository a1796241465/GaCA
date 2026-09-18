"""Create the deterministic CARE candidate tables used for structure retrieval."""

import argparse
import json
from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"Entry", "Sequence", "EC number"}


def read_split(path):
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=sorted(REQUIRED_COLUMNS)).copy()
    frame["Entry"] = frame["Entry"].astype(str)
    frame["Sequence"] = frame["Sequence"].astype(str)
    frame["EC number"] = frame["EC number"].astype(str)
    frame["Length"] = frame.get("Length", frame["Sequence"].str.len())
    return frame


def sample_training(frame, max_length, max_per_ec, seed):
    frame = frame.loc[frame["Length"] <= max_length].copy()
    sampled = []
    for label in frame["EC number"].drop_duplicates():
        group = frame.loc[frame["EC number"] == label]
        if len(group) > max_per_ec:
            group = group.sample(n=max_per_ec, random_state=seed)
        sampled.append(group)
    return (
        pd.concat(sampled, ignore_index=True)
        .drop_duplicates(subset="Entry", keep="first")
        .reset_index(drop=True)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--care-task-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=600)
    parser.add_argument("--max-per-ec", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources = {
        "train": args.care_task_dir / "protein_train50.csv",
        "lt30": args.care_task_dir / "30_protein_test.csv",
        "30_50": args.care_task_dir / "30-50_protein_test.csv",
    }
    tables = {
        "train": sample_training(
            read_split(sources["train"]), args.max_length, args.max_per_ec, args.seed
        ),
        "lt30": read_split(sources["lt30"]),
        "30_50": read_split(sources["30_50"]),
    }
    names = {
        "train": "train_candidates.csv",
        "lt30": "lt30_candidates.csv",
        "30_50": "30_50_candidates.csv",
    }
    for split, table in tables.items():
        table.to_csv(args.output_dir / names[split], index=False)
    record = {
        "max_length": args.max_length,
        "max_per_level4_ec": args.max_per_ec,
        "seed": args.seed,
        "counts": {split: len(table) for split, table in tables.items()},
    }
    (args.output_dir / "candidate_summary.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()