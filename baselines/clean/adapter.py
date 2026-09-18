"""Prepare CARE/GaCA tables for CLEAN and evaluate CLEAN max-separation output."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef


def prefix(label, level):
    return ".".join(str(label).split(".")[:level])


def write_clean_input(table, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    table.loc[:, ["Entry", "EC number", "Sequence"]].to_csv(
        output, sep="\t", index=False
    )


def prepare(data_root, output_dir):
    train = pd.read_csv(data_root / "train_cleaned_with_structure.csv")
    known = set(train["EC number"].astype(str))
    write_clean_input(train, output_dir / "gaca_train.csv")
    for split, filename in (
        ("lt30", "test_30_cleaned_with_structure.csv"),
        ("30_50", "test_30_50_clean.csv"),
    ):
        test = pd.read_csv(data_root / filename)
        test = test.loc[test["EC number"].astype(str).isin(known)].copy()
        write_clean_input(test, output_dir / f"gaca_test_{split}.csv")
        test.loc[:, ["Entry", "EC number"]].rename(
            columns={"Entry": "protein_id", "EC number": "true_ec"}
        ).to_csv(output_dir / f"gaca_test_{split}_labels.csv", index=False)


def parse_maxsep(path):
    predictions = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split(",")
            if len(fields) < 2:
                continue
            prediction = fields[1]
            predictions[fields[0]] = prediction.replace("EC:", "").split("/")[0]
    return predictions


def evaluate(labels_path, result_path, output_prefix):
    labels = pd.read_csv(labels_path)
    calls = parse_maxsep(result_path)
    labels["predicted_ec"] = labels["protein_id"].astype(str).map(calls).fillna("")
    true = labels["true_ec"].astype(str)
    predicted = labels["predicted_ec"].astype(str)
    metrics = {"samples": len(labels)}
    for level in range(1, 5):
        metrics[f"level_{level}_accuracy"] = float(
            (true.map(lambda x: prefix(x, level)) == predicted.map(lambda x: prefix(x, level))).mean()
        )
    reference_classes = sorted(set(true))
    metrics["macro_f1_reference_classes"] = float(
        f1_score(
            true,
            predicted,
            labels=reference_classes,
            average="macro",
            zero_division=0,
        )
    )
    metrics["mcc"] = float(matthews_corrcoef(true, predicted))
    labels.to_csv(output_prefix.with_suffix(".predictions.csv"), index=False)
    output_prefix.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--data-root", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    evaluate_parser = subparsers.add_parser("evaluate")
    evaluate_parser.add_argument("--labels", type=Path, required=True)
    evaluate_parser.add_argument("--result", type=Path, required=True)
    evaluate_parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.data_root, args.output_dir)
    else:
        evaluate(args.labels, args.result, args.output_prefix)


if __name__ == "__main__":
    main()