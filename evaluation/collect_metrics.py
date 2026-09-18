import argparse
import json
from pathlib import Path

import pandas as pd


NEURAL_MODELS = {
    "gaca": "GaCA",
    "seq_only": "ESM-2 sequence only",
    "str_only": "ESM-IF1 structure only",
    "mean_vector_gate": "Mean-pooling vector gate",
    "attention_concat": "Attention concatenation",
    "mean_concat": "Mean-pooled concatenation",
    "cross_attention": "Cross-attention",
}
CLASSICAL_MODELS = {
    "knn": "KNN",
    "svm": "SVM",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
}


def metric_row(dataset, method, method_key, protocol, metrics):
    return {
        "dataset": dataset,
        "method": method,
        "method_key": method_key,
        "protocol": protocol,
        "samples": metrics["samples"],
        "level_1_accuracy": metrics["level_1_accuracy"],
        "level_2_accuracy": metrics["level_2_accuracy"],
        "level_3_accuracy": metrics["level_3_accuracy"],
        "level_4_accuracy": metrics["level_4_accuracy"],
        "macro_f1": metrics["macro_f1_reference_classes"],
        "mcc": metrics["mcc"],
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--blastp-root", type=Path, required=True)
    parser.add_argument("--foldseek-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for key, name in NEURAL_MODELS.items():
        payload = read_json(args.neural_root / key / "metrics.json")
        protocol = "tune_12303_validate_1368_refit_13671"
        if key == "cross_attention":
            protocol += "_dim512_heads4"
        for dataset, metrics in payload["test_results"].items():
            rows.append(metric_row(dataset, name, key, protocol, metrics))

    for key, name in CLASSICAL_MODELS.items():
        payload = read_json(args.baseline_root / key / "metrics.json")
        protocol = (
            "tune_12303_validate_1368_refit_13671_l2norm"
            if key == "lightgbm"
            else "fit_all_13671_fixed_hyperparameters"
        )
        for dataset, metrics in payload["test_results"].items():
            rows.append(metric_row(dataset, name, key, protocol, metrics))

    for key, name, root in (
        ("blastp", "BLASTp", args.blastp_root),
        ("foldseek", "Foldseek", args.foldseek_root),
    ):
        for dataset in ("lt30", "30_50"):
            payload = read_json(root / f"metrics_{dataset}.json")
            rows.append(
                metric_row(
                    dataset,
                    name,
                    key,
                    "top-1 transfer from the effective 13,671-protein training library",
                    payload["test_results"],
                )
            )

    output = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"saved {len(output)} rows to {args.output}")


if __name__ == "__main__":
    main()
