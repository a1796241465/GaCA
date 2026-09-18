"""Evaluate top-1 BLASTp or Foldseek annotation transfer on the GaCA splits."""

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef

FORMATS = {
    "blastp": ["query", "target", "pident", "evalue", "bits", "length", "qcovs"],
    "foldseek": [
        "query", "target", "fident", "evalue", "bits",
        "alntmscore", "qtmscore", "ttmscore",
    ],
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=sorted(FORMATS), required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--test-csv", type=Path, required=True)
    parser.add_argument("--hits", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    return parser.parse_args()


def normalize_identifier(value):
    base = str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]
    for suffix in (".pdb.gz", ".cif.gz", ".mmcif.gz", ".pdb", ".cif", ".mmcif"):
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.startswith("AF-") and "-F" in base:
        base = base[3:].split("-F", 1)[0]
    return base


def prefix(label, level):
    return ".".join(str(label).split(".")[:level])


def metrics(predictions):
    true = predictions["true_ec"].astype(str)
    predicted = predictions["predicted_ec"].fillna("").astype(str)
    result = {"samples": len(predictions)}
    for level in range(1, 5):
        result[f"level_{level}_accuracy"] = float(
            (true.map(lambda x: prefix(x, level)) == predicted.map(lambda x: prefix(x, level))).mean()
        )
    labels = sorted(set(true))
    result["macro_f1_reference_classes"] = float(
        f1_score(true, predicted, labels=labels, average="macro", zero_division=0)
    )
    result["mcc"] = float(matthews_corrcoef(true, predicted))
    result["reference_class_count"] = len(labels)
    return result


def main():
    args = parse_args()
    train = pd.read_csv(args.train_csv).dropna(subset=["Entry", "EC number"])
    train["Entry"] = train["Entry"].astype(str)
    train["EC number"] = train["EC number"].astype(str)
    label_by_target = dict(zip(train["Entry"], train["EC number"]))

    test = pd.read_csv(args.test_csv).dropna(subset=["Entry", "EC number"]).copy()
    test["Entry"] = test["Entry"].astype(str)
    test["EC number"] = test["EC number"].astype(str)
    test = test.loc[test["EC number"].isin(set(label_by_target.values()))].copy()

    hits = pd.read_csv(args.hits, sep="\t", names=FORMATS[args.method], comment="#")
    hits["query_id"] = hits["query"].map(normalize_identifier)
    hits["target_id"] = hits["target"].map(normalize_identifier)
    hits["bits"] = pd.to_numeric(hits["bits"], errors="coerce")
    hits["evalue"] = pd.to_numeric(hits["evalue"], errors="coerce")
    extra_sort = ["alntmscore"] if args.method == "foldseek" else []
    for column in extra_sort:
        hits[column] = pd.to_numeric(hits[column], errors="coerce")
    hits = hits.sort_values(
        ["query_id", "bits", "evalue", *extra_sort],
        ascending=[True, False, True, *([False] * len(extra_sort))],
        na_position="last",
    )
    top_hits = hits.drop_duplicates("query_id", keep="first").set_index("query_id")

    rows = []
    for _, row in test.iterrows():
        protein_id = row["Entry"]
        if protein_id in top_hits.index:
            hit = top_hits.loc[protein_id]
            target_id = str(hit["target_id"])
            predicted_ec = label_by_target.get(target_id, "")
            identity_column = "pident" if args.method == "blastp" else "fident"
            hit_identity = pd.to_numeric(hit[identity_column], errors="coerce")
            bit_score = pd.to_numeric(hit["bits"], errors="coerce")
            evalue = pd.to_numeric(hit["evalue"], errors="coerce")
        else:
            target_id = predicted_ec = ""
            hit_identity = bit_score = evalue = float("nan")
        rows.append(
            {
                "protein_id": protein_id,
                "true_ec": row["EC number"],
                "target_id": target_id,
                "predicted_ec": predicted_ec,
                "hit_identity": hit_identity,
                "bit_score": bit_score,
                "evalue": evalue,
            }
        )
    predictions = pd.DataFrame(rows)
    result = {
        "method": args.method,
        "dataset": args.dataset,
        "top_hit_order": "bits descending, evalue ascending"
        + (", alignment TM-score descending" if args.method == "foldseek" else ""),
        "test_results": metrics(predictions),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / f"predictions_{args.dataset}.csv", index=False)
    (args.output_dir / f"metrics_{args.dataset}.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()