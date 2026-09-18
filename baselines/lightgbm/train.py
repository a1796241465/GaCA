"""Leakage-free LightGBM baseline for the GaCA rerun."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPOSITORY_ROOT / "data" / "processed"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "outputs"
CSV_PATHS = {
    "train": DATA_ROOT / "train_cleaned_with_structure.csv",
    "lt30": DATA_ROOT / "test_30_cleaned_with_structure.csv",
    "30_50": DATA_ROOT / "test_30_50_clean.csv",
}
STORE_NAMES = {
    "train": ("seq_train", "str_train"),
    "lt30": ("seq_lt30", "str_lt30"),
    "30_50": ("seq_30_50", "str_30_50"),
}


class MemmapFeatureStore:
    def __init__(self, directory):
        directory = Path(directory)
        self.values = np.load(directory / "values.npy", mmap_mode="r")
        index = pd.read_csv(directory / "index.csv", dtype={"protein_id": str})
        self.locations = {
            str(row.protein_id): (int(row.offset), int(row.length))
            for row in index.itertuples(index=False)
        }

    def __contains__(self, protein_id):
        return str(protein_id) in self.locations

    def mean(self, protein_id):
        offset, length = self.locations[str(protein_id)]
        rows = np.asarray(
            self.values[offset : offset + length], dtype=np.float32
        )
        return rows.mean(axis=0, dtype=np.float32)


def load_rows(csv_path, seq_store, str_store, known_labels=None):
    frame = pd.read_csv(csv_path, dtype={"Entry": str})
    frame["Entry"] = frame["Entry"].astype(str)
    frame["EC number"] = frame["EC number"].astype(str)
    rows = frame[
        frame["Entry"].map(
            lambda protein_id: protein_id in seq_store
            and protein_id in str_store
        )
    ].copy()
    if known_labels is not None:
        rows = rows[rows["EC number"].isin(known_labels)].copy()
    return rows.reset_index(drop=True)


def build_features(rows, seq_store, str_store):
    features = np.empty((len(rows), 1792), dtype=np.float32)
    for index, protein_id in enumerate(rows["Entry"]):
        features[index, :1280] = seq_store.mean(protein_id)
        features[index, 1280:] = str_store.mean(protein_id)
        if (index + 1) % 1000 == 0 or index + 1 == len(rows):
            print(f"[features] {index + 1}/{len(rows)}", flush=True)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return features / norms


def hierarchical_flags(reference, prediction):
    reference_fields = reference.split(".")
    prediction_fields = prediction.split(".")
    return [
        reference_fields[:level] == prediction_fields[:level]
        for level in range(1, 5)
    ]


def evaluate(rows, predicted_indices, label_encoder):
    predicted_indices = np.asarray(predicted_indices, dtype=int)
    predictions = label_encoder.inverse_transform(predicted_indices)
    references = rows["EC number"].to_numpy(dtype=str)
    true_indices = label_encoder.transform(references)
    records = []
    for protein_id, reference, prediction in zip(
        rows["Entry"], references, predictions
    ):
        flags = hierarchical_flags(reference, prediction)
        records.append(
            {
                "protein_id": protein_id,
                "true_ec": reference,
                "predicted_ec": prediction,
                "level_1_correct": flags[0],
                "level_2_correct": flags[1],
                "level_3_correct": flags[2],
                "level_4_correct": flags[3],
            }
        )
    result = pd.DataFrame(records)
    reference_classes = np.unique(true_indices)
    metrics = {
        "samples": len(result),
        "level_1_accuracy": float(result["level_1_correct"].mean()),
        "level_2_accuracy": float(result["level_2_correct"].mean()),
        "level_3_accuracy": float(result["level_3_correct"].mean()),
        "level_4_accuracy": float(result["level_4_correct"].mean()),
        "macro_f1_reference_classes": float(
            f1_score(
                true_indices,
                predicted_indices,
                labels=reference_classes,
                average="macro",
                zero_division=0,
            )
        ),
        "mcc": float(matthews_corrcoef(true_indices, predicted_indices)),
        "reference_class_count": int(len(reference_classes)),
    }
    return metrics, result


def model_parameters(num_classes, n_estimators):
    return {
        "num_leaves": 31,
        "max_depth": 10,
        "learning_rate": 0.1,
        "n_estimators": int(n_estimators),
        "objective": "multiclass",
        "num_class": int(num_classes),
        "n_jobs": 8,
        "random_state": 42,
        "verbosity": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--split-file",
        type=Path,
        default=REPOSITORY_ROOT / "metadata" / "fixed_fit_validation_split.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=DEFAULT_OUTPUT / "feature_store",
    )
    parser.add_argument("--max-estimators", type=int, default=100)
    parser.add_argument("--early-stopping-rounds", type=int, default=10)
    return parser.parse_args()


def main():
    global CSV_PATHS
    args = parse_args()
    CSV_PATHS = {
        "train": args.data_root / "train_cleaned_with_structure.csv",
        "lt30": args.data_root / "test_30_cleaned_with_structure.csv",
        "30_50": args.data_root / "test_30_50_clean.csv",
    }
    output_dir = args.output_dir / "baselines" / "lightgbm"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    stores = {
        split: (
            MemmapFeatureStore(args.feature_root / names[0]),
            MemmapFeatureStore(args.feature_root / names[1]),
        )
        for split, names in STORE_NAMES.items()
    }
    train_rows = load_rows(CSV_PATHS["train"], *stores["train"])
    label_encoder = LabelEncoder().fit(train_rows["EC number"])
    known_labels = set(label_encoder.classes_)
    test_rows = {
        split: load_rows(CSV_PATHS[split], *stores[split], known_labels)
        for split in ("lt30", "30_50")
    }

    split_frame = pd.read_csv(
        args.split_file,
        dtype={"protein_id": str},
    )
    if split_frame["protein_id"].duplicated().any():
        raise RuntimeError("The fixed split contains duplicate protein identifiers")
    if set(split_frame["split"]) - {"fit", "validation"}:
        raise RuntimeError("The fixed split contains an unknown partition name")
    if set(split_frame["protein_id"]) != set(train_rows["Entry"]):
        raise RuntimeError("The fixed split does not exactly match the training proteins")
    split_by_id = dict(zip(split_frame["protein_id"], split_frame["split"]))
    assigned = train_rows["Entry"].map(split_by_id)
    if assigned.isna().any():
        raise RuntimeError("The fixed split omits effective training proteins")
    fitting_mask = assigned.eq("fit").to_numpy()
    validation_mask = assigned.eq("validation").to_numpy()
    if fitting_mask.sum() != 12303 or validation_mask.sum() != 1368:
        raise RuntimeError("Unexpected fixed split counts")
    fitting_labels = set(train_rows.loc[fitting_mask, "EC number"])
    if fitting_labels != known_labels:
        raise RuntimeError("Fitting data do not cover every training label")

    print("[features] training", flush=True)
    x_train = build_features(train_rows, *stores["train"])
    x_tests = {}
    for split in ("lt30", "30_50"):
        print(f"[features] {split}", flush=True)
        x_tests[split] = build_features(test_rows[split], *stores[split])

    y_train = label_encoder.transform(train_rows["EC number"])
    tune_parameters = model_parameters(
        len(label_encoder.classes_), args.max_estimators
    )
    tuning_model = lgb.LGBMClassifier(**tune_parameters)
    print("[tune] fitting with validation-only early stopping", flush=True)
    tuning_model.fit(
        x_train[fitting_mask],
        y_train[fitting_mask],
        eval_set=[(x_train[validation_mask], y_train[validation_mask])],
        eval_metric="multi_logloss",
        callbacks=[
            lgb.early_stopping(args.early_stopping_rounds, verbose=True),
            lgb.log_evaluation(period=10),
        ],
    )
    best_iteration = int(
        tuning_model.best_iteration_ or args.max_estimators
    )
    print(f"[tune] best_iteration={best_iteration}", flush=True)

    final_parameters = model_parameters(
        len(label_encoder.classes_), best_iteration
    )
    final_model = lgb.LGBMClassifier(**final_parameters)
    print("[refit] fitting all 13,671 effective training proteins", flush=True)
    final_model.fit(x_train, y_train)
    final_model.booster_.save_model(str(output_dir / "model.txt"))

    test_metrics = {}
    for split in ("lt30", "30_50"):
        predicted = final_model.predict(x_tests[split])
        metrics, predictions = evaluate(
            test_rows[split], predicted, label_encoder
        )
        test_metrics[split] = metrics
        predictions.to_csv(
            output_dir / f"predictions_{split}.csv", index=False
        )
        print(
            f"[test] {split}: "
            f"L1={metrics['level_1_accuracy']:.5f} "
            f"L2={metrics['level_2_accuracy']:.5f} "
            f"L3={metrics['level_3_accuracy']:.5f} "
            f"L4={metrics['level_4_accuracy']:.5f}",
            flush=True,
        )

    result = {
        "model": "LightGBM",
        "feature_definition": (
            "row-wise L2-normalised concatenation of mean-pooled ESM-2 "
            "and ESM-IF1 residue embeddings"
        ),
        "fitting_samples": int(fitting_mask.sum()),
        "validation_samples": int(validation_mask.sum()),
        "full_refit_samples": int(len(train_rows)),
        "classes": int(len(label_encoder.classes_)),
        "tuning_parameters": tune_parameters,
        "best_iteration": best_iteration,
        "final_parameters": final_parameters,
        "test_results": test_metrics,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
            "lightgbm": lgb.__version__,
        },
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "class_index": np.arange(len(label_encoder.classes_)),
            "ec_label": label_encoder.classes_,
        }
    ).to_csv(output_dir / "label_mapping.csv", index=False)


if __name__ == "__main__":
    main()
