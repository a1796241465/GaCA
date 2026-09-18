"""Reproducible KNN, linear SVM, and Random Forest baselines for GaCA."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, matthews_corrcoef
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC


MODEL_NAMES = ("knn", "svm", "random_forest")
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


def l2_normalise(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return features / norms


def build_sequence_features(rows, sequence_store):
    features = np.empty((len(rows), 1280), dtype=np.float32)
    for index, protein_id in enumerate(rows["Entry"]):
        features[index] = sequence_store.mean(protein_id)
    return l2_normalise(features)


def build_concatenated_features(rows, seq_store, str_store):
    features = np.empty((len(rows), 1792), dtype=np.float32)
    for index, protein_id in enumerate(rows["Entry"]):
        features[index, :1280] = seq_store.mean(protein_id)
        features[index, 1280:] = str_store.mean(protein_id)
    return l2_normalise(features)


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


def model_and_parameters(name):
    if name == "knn":
        parameters = {"n_neighbors": 1, "metric": "cosine", "n_jobs": -1}
        return KNeighborsClassifier(**parameters), parameters
    if name == "svm":
        parameters = {"C": 1.0, "max_iter": 1000, "random_state": 42, "dual": True}
        return LinearSVC(**parameters), parameters
    if name == "random_forest":
        parameters = {
            "n_estimators": 100,
            "n_jobs": -1,
            "random_state": 42,
        }
        return RandomForestClassifier(**parameters), parameters
    raise KeyError(name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = args.output_dir / "baselines"
    output_root.mkdir(parents=True, exist_ok=True)
    csv_paths = {
        "train": args.data_root / "train_cleaned_with_structure.csv",
        "lt30": args.data_root / "test_30_cleaned_with_structure.csv",
        "30_50": args.data_root / "test_30_50_clean.csv",
    }
    stores = {
        split: (
            MemmapFeatureStore(args.feature_root / names[0]),
            MemmapFeatureStore(args.feature_root / names[1]),
        )
        for split, names in STORE_NAMES.items()
    }
    train_rows = load_rows(csv_paths["train"], *stores["train"])
    label_encoder = LabelEncoder().fit(train_rows["EC number"])
    known_labels = set(label_encoder.classes_)
    test_rows = {
        split: load_rows(csv_paths[split], *stores[split], known_labels)
        for split in ("lt30", "30_50")
    }
    y_train = label_encoder.transform(train_rows["EC number"])

    sequence_features = {}
    concatenated_features = {}
    if "knn" in args.models:
        sequence_features["train"] = build_sequence_features(
            train_rows, stores["train"][0]
        )
        for split in ("lt30", "30_50"):
            sequence_features[split] = build_sequence_features(
                test_rows[split], stores[split][0]
            )
    if any(name in args.models for name in ("svm", "random_forest")):
        concatenated_features["train"] = build_concatenated_features(
            train_rows, *stores["train"]
        )
        for split in ("lt30", "30_50"):
            concatenated_features[split] = build_concatenated_features(
                test_rows[split], *stores[split]
            )

    aggregate = {}
    for name in args.models:
        started = time.time()
        model, parameters = model_and_parameters(name)
        train_features = (
            sequence_features["train"]
            if name == "knn"
            else concatenated_features["train"]
        )
        model.fit(train_features, y_train)
        model_dir = output_root / name
        model_dir.mkdir(parents=True, exist_ok=True)
        test_metrics = {}
        for split in ("lt30", "30_50"):
            features = (
                sequence_features[split]
                if name == "knn"
                else concatenated_features[split]
            )
            predicted = model.predict(features)
            metrics, predictions = evaluate(
                test_rows[split], predicted, label_encoder
            )
            test_metrics[split] = metrics
            predictions.to_csv(
                model_dir / f"predictions_{split}.csv", index=False
            )
            print(
                f"[test] {name} {split}: "
                f"L4={metrics['level_4_accuracy']:.5f}",
                flush=True,
            )
        result = {
            "model": name,
            "training_samples": int(len(train_rows)),
            "classes": int(len(label_encoder.classes_)),
            "parameters": parameters,
            "feature_definition": (
                "row-wise L2-normalised mean-pooled ESM-2 sequence embeddings"
                if name == "knn"
                else "row-wise L2-normalised concatenation of mean-pooled "
                "ESM-2 and ESM-IF1 residue embeddings"
            ),
            "test_results": test_metrics,
            "elapsed_seconds": time.time() - started,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "sklearn": sklearn.__version__,
            },
        }
        (model_dir / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        aggregate[name] = result
    (output_root / "classical_baselines_metrics.json").write_text(
        json.dumps(aggregate, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
