"""Export the exact sequence/structure-complete data used by all methods."""

import argparse
import json
from pathlib import Path

import pandas as pd
from common.protocol import validate_frame, validate_reference_tables

STORE_NAMES = {
    "train": ("seq_train", "str_train"),
    "lt30": ("seq_lt30", "str_lt30"),
    "30_50": ("seq_30_50", "str_30_50"),
}
CSV_NAMES = {
    "train": "train_cleaned_with_structure.csv",
    "lt30": "test_30_cleaned_with_structure.csv",
    "30_50": "test_30_50_clean.csv",
}
EXPECTED_COUNTS = {"train": 13671, "lt30": 243, "30_50": 477}


def store_ids(path):
    index = pd.read_csv(path / "index.csv", dtype={"protein_id": str})
    return set(index["protein_id"].astype(str))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-count-check", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    available = {
        split: store_ids(args.feature_root / seq_name)
        & store_ids(args.feature_root / str_name)
        for split, (seq_name, str_name) in STORE_NAMES.items()
    }
    tables = {}
    for split, filename in CSV_NAMES.items():
        table = pd.read_csv(args.data_root / filename, dtype={"Entry": str})
        validate_frame(table, args.data_root / filename)
        table["Entry"] = table["Entry"].astype(str)
        table["EC number"] = table["EC number"].astype(str)
        tables[split] = table.loc[table["Entry"].isin(available[split])].copy()

    training_labels = set(tables["train"]["EC number"])
    for split in ("lt30", "30_50"):
        tables[split] = tables[split].loc[
            tables[split]["EC number"].isin(training_labels)
        ].copy()

    counts = {split: len(table) for split, table in tables.items()}
    if not args.skip_count_check and counts != EXPECTED_COUNTS:
        raise RuntimeError(f"Expected {EXPECTED_COUNTS}, obtained {counts}")
    if not args.skip_count_check:
        validate_reference_tables(tables)
    for split, table in tables.items():
        table.to_csv(args.output_dir / CSV_NAMES[split], index=False)
    record = {
        "definition": (
            "sequence- and structure-complete proteins; test labels must occur "
            "in the effective training label space"
        ),
        "counts": counts,
        "level4_training_labels": len(training_labels),
    }
    (args.output_dir / "effective_split_summary.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
