from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COUNTS = {
    "train": 13671,
    "lt30": 243,
    "30_50": 477,
    "classes": 3811,
    "fit": 12303,
    "validation": 1368,
}


def validate_frame(frame, source):
    source = Path(source)
    required = {"Entry", "EC number"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing columns: {sorted(missing)}")
    if frame["Entry"].isna().any() or frame["EC number"].isna().any():
        raise ValueError(f"{source} contains missing protein identifiers or EC labels")
    if frame["Entry"].astype(str).duplicated().any():
        raise ValueError(f"{source} contains duplicate protein identifiers")


def validate_split(split, protein_ids):
    required = {"protein_id", "split"}
    missing = required.difference(split.columns)
    if missing:
        raise ValueError(f"Fixed split is missing columns: {sorted(missing)}")
    if split["protein_id"].duplicated().any():
        raise ValueError("Fixed split contains duplicate protein identifiers")
    unknown = set(split["split"]) - {"fit", "validation"}
    if unknown:
        raise ValueError(f"Unknown fixed-split partitions: {sorted(unknown)}")
    expected_ids = set(map(str, protein_ids))
    observed_ids = set(split["protein_id"].astype(str))
    if observed_ids != expected_ids:
        raise ValueError(
            "Fixed split does not exactly match the effective training proteins"
        )


def validate_partitions(labels, fitting, validation):
    fitting = np.asarray(fitting, dtype=int)
    validation = np.asarray(validation, dtype=int)
    if len(fitting) != EXPECTED_COUNTS["fit"]:
        raise ValueError(f"Expected 12,303 fitting proteins, found {len(fitting)}")
    if len(validation) != EXPECTED_COUNTS["validation"]:
        raise ValueError(f"Expected 1,368 validation proteins, found {len(validation)}")
    if np.intersect1d(fitting, validation).size:
        raise ValueError("Fitting and validation partitions overlap")
    if len(fitting) + len(validation) != len(labels):
        raise ValueError("Fitting and validation partitions do not cover the training data")
    if set(np.asarray(labels)[fitting]) != set(np.asarray(labels)):
        raise ValueError("The fitting partition does not contain every training label")


def validate_final_protocol(train_dataset, test_datasets):
    observed = {
        "train": len(train_dataset),
        "lt30": len(test_datasets["lt30"]),
        "30_50": len(test_datasets["30_50"]),
        "classes": len(train_dataset.label_encoder.classes_),
    }
    expected = {key: EXPECTED_COUNTS[key] for key in observed}
    if observed != expected:
        raise ValueError(f"Expected final protocol {expected}, found {observed}")
    validate_reference_tables({
        "train": train_dataset.frame,
        **{name: dataset.frame for name, dataset in test_datasets.items()},
    })
    reference = pd.read_csv(
        Path(__file__).resolve().parents[1] / "metadata" / "label_mapping_reference.csv"
    ).sort_values("class_index")
    if not np.array_equal(train_dataset.label_encoder.classes_, reference["ec_label"]):
        raise ValueError("Class-index mapping differs from the archived 73.25% run")


def validate_reference_tables(tables):
    metadata = Path(__file__).resolve().parents[1] / "metadata"
    for name, frame in tables.items():
        validate_frame(frame, name)
        reference = pd.read_csv(metadata / f"effective_{name}.csv", dtype=str)
        actual = frame[["Entry", "EC number"]].astype(str).reset_index(drop=True)
        if not actual.equals(reference):
            raise ValueError(
                f"{name}: protein IDs, row order, or labels differ from the archived run"
            )
