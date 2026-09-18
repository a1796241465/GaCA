"""Export probabilities and vector-wise gates from a trained GaCA checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


from gaca import train as runner


def load_state_dict(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in checkpoint:
                return checkpoint[key]
    return checkpoint


def verify_predictions(actual, expected_path):
    expected = pd.read_csv(expected_path, dtype={"protein_id": str})
    columns = ["protein_id", "true_ec", "predicted_ec"]
    actual_core = actual[columns].astype(str).reset_index(drop=True)
    expected_core = expected[columns].astype(str).reset_index(drop=True)
    if not actual_core.equals(expected_core):
        merged = actual_core.merge(
            expected_core,
            on="protein_id",
            how="outer",
            suffixes=("_actual", "_expected"),
            indicator=True,
        )
        mismatched = merged.loc[
            (merged["_merge"] != "both")
            | (merged["true_ec_actual"] != merged["true_ec_expected"])
            | (merged["predicted_ec_actual"] != merged["predicted_ec_expected"])
        ]
        raise RuntimeError(
            f"Inference predictions disagree with {expected_path}: "
            f"{len(mismatched)} mismatched rows"
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-predictions-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runner.CSV_PATHS = {
        "train": args.data_root / "train_cleaned_with_structure.csv",
        "lt30": args.data_root / "test_30_cleaned_with_structure.csv",
        "30_50": args.data_root / "test_30_50_clean.csv",
    }

    runner.seed_everything(42)
    train_dataset, test_datasets = runner.load_datasets(args.feature_root)
    runner.validate_final_protocol(train_dataset, test_datasets)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = runner.GaCA(
        num_classes=len(train_dataset.label_encoder.classes_),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(load_state_dict(args.checkpoint, device), strict=True)
    model.eval()

    summary = {
        "checkpoint": str(args.checkpoint.resolve()),
        "device": str(device),
        "classes": len(train_dataset.label_encoder.classes_),
        "subsets": {},
    }
    for split, dataset in test_datasets.items():
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=device.type == "cuda",
            collate_fn=runner.collate_batch,
        )
        metrics, predictions, probabilities, gates = runner.evaluate_test(
            model, dataset, loader, device, bool(args.amp and device.type == "cuda")
        )
        if gates is None:
            raise RuntimeError("GaCA inference did not return gate vectors")
        if args.expected_predictions_dir:
            verify_predictions(
                predictions,
                args.expected_predictions_dir / f"predictions_{split}.csv",
            )

        np.savez_compressed(
            args.output_dir / f"probabilities_{split}.npz",
            probabilities=probabilities,
            true_indices=dataset.labels,
            protein_ids=np.asarray(dataset.ids, dtype=str),
        )
        np.save(args.output_dir / f"gate_vectors_{split}.npy", gates)
        predictions.to_csv(
            args.output_dir / f"inference_predictions_{split}.csv", index=False
        )
        summary["subsets"][split] = {
            **metrics,
            "probability_shape": list(probabilities.shape),
            "gate_shape": list(gates.shape),
        }
        print(
            f"[done] {split}: probabilities={probabilities.shape}, "
            f"gates={gates.shape}, level4={metrics['level_4_accuracy']:.6f}",
            flush=True,
        )

    (args.output_dir / "inference_artifacts_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
