"""Recoverable command orchestration for the audited GaCA workflow."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
HISTORICAL_SEED = 42
REFERENCE = {"lt30": 0.7325102880658436, "30_50": 0.8490566037735849}
Validator = Callable[[], None]


class PipelineError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    data_root: Path
    output_root: Path
    device: str = "auto"
    workers: int = 4
    seed: int = HISTORICAL_SEED
    resume: bool = False
    force: bool = False

    @property
    def state_root(self):
        return self.output_root / ".reproduce" / "receipts"


@dataclass
class Action:
    key: str
    command: list[str]
    inputs: tuple[Path, ...] = ()
    outputs: tuple[Path, ...] = ()
    validator: Validator = lambda: None
    recovery: str = "error"
    required_tools: tuple[str, ...] = ()


@dataclass
class Stage:
    name: str
    actions: list[Action] = field(default_factory=list)


def py(module, *arguments):
    return [PYTHON, "-m", module, *map(str, arguments)]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_signature(path):
    path = Path(path)
    if not path.exists():
        return {"kind": "missing"}
    if path.is_file():
        stat = path.stat()
        result = {
            "kind": "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        }
        if stat.st_size <= 8 * 1024 * 1024:
            result["sha256"] = sha256(path)
        return result
    entries = []
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        stat = item.stat()
        entries.append((str(item.relative_to(path)), stat.st_size, stat.st_mtime_ns))
    encoded = json.dumps(entries, separators=(",", ":")).encode("utf-8")
    return {
        "kind": "directory",
        "files": len(entries),
        "listing_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def signatures(paths):
    return {str(Path(path).resolve()): path_signature(path) for path in paths}


def require_files(paths):
    paths = list(paths)
    if not paths:
        raise PipelineError("Expected at least one output file")
    missing = [str(path) for path in paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise PipelineError("Missing or empty output files: " + ", ".join(missing))


def csv_rows(path, required=()):
    require_files([path])
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not set(required).issubset(reader.fieldnames or []):
            raise PipelineError(f"{path} lacks required columns: {sorted(required)}")
        return list(reader)


def validate_care(care_root):
    task = care_root / "splits" / "task1"
    require_files(
        [
            care_root / "gaca_source_record.json",
            task / "protein_train50.csv",
            task / "30_protein_test.csv",
            task / "30-50_protein_test.csv",
        ]
    )


def validate_candidates(root):
    summary_path = root / "candidate_summary.json"
    paths = [
        root / "train_candidates.csv",
        root / "lt30_candidates.csv",
        root / "30_50_candidates.csv",
    ]
    require_files([summary_path, *paths])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {"max_length": 600, "max_per_level4_ec": 25, "seed": 42}
    if any(summary.get(key) != value for key, value in expected.items()):
        raise PipelineError("Candidate summary does not match the historical protocol")
    for path in paths:
        rows = csv_rows(path, {"Entry", "Sequence", "EC number"})
        ids = [str(row["Entry"]) for row in rows]
        if not rows or len(ids) != len(set(ids)):
            raise PipelineError(f"{path} is empty or contains duplicate protein IDs")


def validate_structures(candidate_csv, structure_root, manifest):
    candidate_ids = {row["Entry"] for row in csv_rows(candidate_csv, {"Entry"})}
    records = csv_rows(manifest, {"protein_id", "status"})
    manifest_ids = [row["protein_id"] for row in records]
    if set(manifest_ids) != candidate_ids or len(manifest_ids) != len(set(manifest_ids)):
        raise PipelineError(f"{manifest} does not cover each candidate exactly once")
    for row in records:
        if row["status"] in {"downloaded", "exists"}:
            pdb = structure_root / f"{row['protein_id']}.pdb"
            if not pdb.is_file() or pdb.stat().st_size == 0:
                raise PipelineError(f"Manifest references a missing structure: {pdb}")


def validate_preprocessing(processed):
    summary_path = processed / "preprocessing_summary.json"
    paths = [
        processed / "train_cleaned_with_structure.csv",
        processed / "test_30_cleaned_with_structure.csv",
        processed / "test_30_50_clean.csv",
    ]
    require_files([summary_path, *paths])
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    expected = {"max_length": 600, "max_per_level4_ec": 25, "seed": 42}
    settings = summary.get("settings", {})
    if any(settings.get(key) != value for key, value in expected.items()):
        raise PipelineError("Preprocessing summary does not match the historical protocol")
    for path in paths:
        rows = csv_rows(path, {"Entry", "Sequence", "EC number"})
        ids = [row["Entry"] for row in rows]
        if not rows or len(ids) != len(set(ids)):
            raise PipelineError(f"{path} is empty or contains duplicate protein IDs")


def validate_alignment(output_pkl, output_csv, stats_path):
    require_files([output_pkl, output_csv, stats_path])
    rows = csv_rows(output_csv, {"Entry"})
    report = json.loads(stats_path.read_text(encoding="utf-8"))
    if report.get("accepted_rows") != len(rows):
        raise PipelineError(f"Alignment count mismatch in {stats_path}")
    expected = {"min_identity": 0.95, "min_length": 50, "max_length": 1000}
    if any(report.get(key) != value for key, value in expected.items()):
        raise PipelineError(f"Alignment settings in {stats_path} do not match the protocol")


def validate_embedding(path):
    require_files([path])


def validate_feature_stores(feature_root, embedding_root):
    dimensions = {
        "seq_train": 1280,
        "seq_lt30": 1280,
        "seq_30_50": 1280,
        "str_train": 512,
        "str_lt30": 512,
        "str_30_50": 512,
    }
    for name, feature_dim in dimensions.items():
        store = feature_root / name
        source = embedding_root / f"{name}.pkl"
        require_files([store / "values.npy", store / "index.csv", store / "metadata.json"])
        metadata = json.loads((store / "metadata.json").read_text(encoding="utf-8"))
        expected = (str(source), source.stat().st_size, source.stat().st_mtime_ns, feature_dim)
        actual = tuple(
            metadata.get(key)
            for key in ("source", "source_size_bytes", "source_mtime_ns", "feature_dim")
        )
        if actual != expected:
            raise PipelineError(f"Feature-store metadata is stale for {name}")


def validate_effective(effective_root):
    import pandas as pd
    from common.protocol import validate_reference_tables

    paths = {
        "train": effective_root / "train_cleaned_with_structure.csv",
        "lt30": effective_root / "test_30_cleaned_with_structure.csv",
        "30_50": effective_root / "test_30_50_clean.csv",
    }
    require_files([*paths.values(), effective_root / "effective_split_summary.json"])
    validate_reference_tables({name: pd.read_csv(path) for name, path in paths.items()})


def validate_predictions(path, expected_rows):
    rows = csv_rows(path, {"protein_id", "true_ec", "predicted_ec"})
    ids = [row["protein_id"] for row in rows]
    if len(rows) != expected_rows or len(ids) != len(set(ids)):
        raise PipelineError(f"Unexpected prediction rows in {path}")


def validate_neural_model(model_root, name, require_gaca_artifacts=False):
    directory = model_root / name
    required = [
        directory / "metrics.json",
        directory / "final_full_training_model.pth",
        directory / "tuning_history.csv",
        directory / "full_refit_history.csv",
        directory / "predictions_lt30.csv",
        directory / "predictions_30_50.csv",
    ]
    require_files(required)
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("full_refit_samples") != 13671 or metrics.get("batch_size") is None:
        raise PipelineError(f"Incomplete training metrics for {name}")
    for dataset, samples in (("lt30", 243), ("30_50", 477)):
        result = metrics.get("test_results", {}).get(dataset, {})
        if result.get("samples") != samples or "level_4_accuracy" not in result:
            raise PipelineError(f"Incomplete {dataset} metrics for {name}")
        validate_predictions(directory / f"predictions_{dataset}.csv", samples)
    if require_gaca_artifacts:
        if metrics.get("trainable_parameters") != 4_156_645:
            raise PipelineError("GaCA parameter count differs from the archived architecture")
        require_files(
            [
                directory / "probabilities_lt30.npz",
                directory / "probabilities_30_50.npz",
                directory / "gate_vectors_lt30.npy",
                directory / "gate_vectors_30_50.npy",
            ]
        )


def validate_classical(output_root, names):
    for name in names:
        directory = output_root / "baselines" / name
        require_files([directory / "metrics.json"])
        validate_predictions(directory / "predictions_lt30.csv", 243)
        validate_predictions(directory / "predictions_30_50.csv", 477)


def validate_retrieval(output_root, dataset=None):
    sample_counts = {"lt30": 243, "30_50": 477}
    datasets = sample_counts if dataset is None else (dataset,)
    found = False
    for name in datasets:
        metrics = output_root / f"metrics_{name}.json"
        predictions = output_root / f"predictions_{name}.csv"
        if dataset is None and not metrics.exists() and not predictions.exists():
            continue
        require_files([metrics, predictions])
        validate_predictions(predictions, sample_counts[name])
        found = True
    if not found:
        raise PipelineError(f"No retrieval results found in {output_root}")


def validate_figures(output_root):
    require_files(
        [
            output_root / "updated_figure_statistics.json",
            output_root / "Fig_Main_Results.pdf",
            output_root / "precision_recall_curves.pdf",
            output_root / "Fig_S2_Softmax_Scores.pdf",
            output_root / "Fig_Level1_Confusion.pdf",
            output_root / "Fig_Gating_Distributions.pdf",
        ]
    )


class PipelineRunner:
    def __init__(self, config, root=ROOT):
        self.config = config
        self.root = Path(root)

    def receipt_path(self, action):
        return self.config.state_root / f"{action.key}.json"

    def receipt(self, action):
        path = self.receipt_path(action)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise PipelineError(f"Invalid resume receipt {path}: {error}") from error

    def write_receipt(self, action):
        self.config.state_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": 1,
            "command": action.command,
            "inputs": signatures(action.inputs),
            "outputs": signatures(action.outputs),
        }
        self.receipt_path(action).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def receipt_matches(self, action, receipt):
        return (
            receipt.get("schema") == 1
            and receipt.get("command") == action.command
            and receipt.get("inputs") == signatures(action.inputs)
            and receipt.get("outputs") == signatures(action.outputs)
        )

    def output_exists(self, action):
        return any(path.exists() for path in action.outputs)

    def check_tools(self, action):
        missing = [tool for tool in action.required_tools if shutil.which(tool) is None]
        if missing:
            raise PipelineError("Required executable(s) not found on PATH: " + ", ".join(missing))

    def execute_process(self, command):
        subprocess.run(command, cwd=self.root, check=True)

    def execute_action(self, action):
        self.check_tools(action)
        receipt = None if self.config.force else self.receipt(action)

        if self.config.resume and receipt is not None:
            try:
                action.validator()
            except Exception as error:
                raise PipelineError(
                    f"Cannot resume {action.key}: recorded outputs are invalid ({error}). "
                    "Use --force to rerun this action."
                ) from error
            if not self.receipt_matches(action, receipt):
                raise PipelineError(
                    f"Cannot resume {action.key}: command, inputs, or outputs changed. "
                    "Use --force after reviewing the change."
                )
            print(f"    [resume] {action.key}")
            return False

        if self.config.resume and receipt is None and self.output_exists(action):
            if action.recovery == "accept":
                action.validator()
                self.write_receipt(action)
                print(f"    [accepted] {action.key}")
                return False
            if action.recovery != "rerun":
                raise PipelineError(
                    f"Cannot verify pre-existing outputs for {action.key}; "
                    "use --force to rebuild them."
                )
            print(f"    [recover] rerunning {action.key}; existing outputs will not be trusted")

        if not self.config.resume and not self.config.force and self.output_exists(action):
            if action.recovery == "accept":
                action.validator()
                self.write_receipt(action)
                print(f"    [accepted] {action.key}")
                return False
            raise PipelineError(
                f"Outputs already exist for {action.key}; use --resume or --force."
            )

        if self.config.force and action.recovery == "accept" and self.output_exists(action):
            action.validator()
            self.write_receipt(action)
            print(f"    [kept] {action.key} (raw source is not destructively replaced)")
            return False

        # CARE cloning requires an empty destination; its downloader creates it.
        if action.key != "care_download":
            for output in action.outputs:
                output.parent.mkdir(parents=True, exist_ok=True)
        command = action.command
        if self.config.force and action.key == "feature_stores":
            command = [*command, "--overwrite"]
        print("    $ " + shlex.join(command), flush=True)
        # A failed or interrupted rerun must not retain an earlier completion.
        self.receipt_path(action).unlink(missing_ok=True)
        self.execute_process(command)
        action.validator()
        self.write_receipt(action)
        return True

    def run(self, stages):
        print("GaCA Reproduction Pipeline")
        print()
        for index, stage in enumerate(stages, start=1):
            print(f"[{index}/{len(stages)}] {stage.name} ...", flush=True)
            ran = False
            try:
                for action in stage.actions:
                    ran = self.execute_action(action) or ran
            except Exception:
                print(f"[{index}/{len(stages)}] {stage.name} ... FAILED", flush=True)
                raise
            status = "OK" if ran else "SKIPPED"
            print(f"[{index}/{len(stages)}] {stage.name} ... {status}", flush=True)


def action(key, command, inputs=(), outputs=(), validator=lambda: None, recovery="error", tools=()):
    return Action(key, list(map(str, command)), tuple(map(Path, inputs)), tuple(map(Path, outputs)), validator, recovery, tuple(tools))


def build_gaca_stages(config):
    data = config.data_root
    outputs = config.output_root
    raw = data / "raw" / "care"
    task = raw / "splits" / "task1"
    candidates = data / "processed" / "candidates"
    processed = data / "processed"
    embeddings = data / "embeddings"
    stores = embeddings / "feature_store"
    effective = processed / "effective"
    structures = data / "structures"
    split_file = ROOT / "metadata" / "fixed_fit_validation_split.csv"
    device = "cuda" if config.device == "auto" else config.device

    care_files = [task / name for name in ("protein_train50.csv", "30_protein_test.csv", "30-50_protein_test.csv")]
    candidate_files = {
        "train": candidates / "train_candidates.csv",
        "lt30": candidates / "lt30_candidates.csv",
        "30_50": candidates / "30_50_candidates.csv",
    }
    structure_dirs = {name: structures / name for name in candidate_files}
    manifests = {name: structures / f"{name}_manifest.csv" for name in candidate_files}
    processed_files = {
        "train": processed / "train_cleaned_with_structure.csv",
        "lt30": processed / "test_30_cleaned_with_structure.csv",
        "30_50": processed / "test_30_50_clean.csv",
    }

    stages = [
        Stage("Preparing CARE data", [
            action(
                "care_download",
                py("data_download.download_care", "--output-dir", raw),
                outputs=[raw / "gaca_source_record.json", *care_files],
                validator=lambda: validate_care(raw),
                recovery="accept",
                tools=("git",),
            ),
            action(
                "candidate_selection",
                py(
                    "preprocessing.select_candidates", "--care-task-dir", task,
                    "--output-dir", candidates, "--max-length", "600",
                    "--max-per-ec", "25", "--seed", str(config.seed),
                ),
                inputs=care_files,
                outputs=[*candidate_files.values(), candidates / "candidate_summary.json"],
                validator=lambda: validate_candidates(candidates),
                recovery="rerun",
            ),
        ]),
        Stage("Downloading AlphaFold structures", [
            action(
                f"alphafold_{name}",
                py(
                    "data_download.alphafold", "--csv", candidate_files[name],
                    "--output-dir", structure_dirs[name], "--manifest", manifests[name],
                ),
                inputs=[candidate_files[name]],
                outputs=[manifests[name], structure_dirs[name]],
                validator=lambda n=name: validate_structures(candidate_files[n], structure_dirs[n], manifests[n]),
                recovery="rerun",
            )
            for name in ("train", "lt30", "30_50")
        ]),
        Stage("Preprocessing and validating sequences", [
            action(
                "prepare_splits",
                py(
                    "preprocessing.prepare_splits", "--care-task-dir", task,
                    "--train-structures", structure_dirs["train"],
                    "--lt30-structures", structure_dirs["lt30"],
                    "--30-50-structures", structure_dirs["30_50"],
                    "--train-manifest", manifests["train"], "--output-dir", processed,
                    "--max-length", "600", "--max-per-ec", "25",
                    "--min-confidence", "70", "--seed", str(config.seed),
                ),
                inputs=[*care_files, *manifests.values(), *structure_dirs.values()],
                outputs=[*processed_files.values(), processed / "preprocessing_summary.json"],
                validator=lambda: validate_preprocessing(processed),
                recovery="rerun",
            ),
            *[
                action(
                    f"align_{name}",
                    py(
                        "preprocessing.align_sequences", "--csv", processed_files[name],
                        "--structures", structure_dirs[name],
                        "--output-pkl", processed / f"{name}_validated_sequences.pkl",
                        "--output-csv", processed / f"{name}_alignment.csv",
                        "--stats", processed / f"{name}_alignment.json",
                        "--min-identity", "0.95", "--min-length", "50", "--max-length", "1000",
                    ),
                    inputs=[processed_files[name], structure_dirs[name]],
                    outputs=[processed / f"{name}_validated_sequences.pkl", processed / f"{name}_alignment.csv", processed / f"{name}_alignment.json"],
                    validator=lambda n=name: validate_alignment(processed / f"{n}_validated_sequences.pkl", processed / f"{n}_alignment.csv", processed / f"{n}_alignment.json"),
                    recovery="rerun",
                )
                for name in ("train", "lt30")
            ],
        ]),
        Stage("Extracting ESM-2 embeddings", [
            action(
                f"esm2_{name}",
                py(
                    "embeddings.extract_esm2", "--input",
                    processed / f"{name}_validated_sequences.pkl" if name != "30_50" else processed_files[name],
                    "--output", embeddings / f"seq_{name}.pkl", "--batch-size", "16",
                    "--max-length", "1024", "--device", device,
                ),
                inputs=[processed / f"{name}_validated_sequences.pkl" if name != "30_50" else processed_files[name]],
                outputs=[embeddings / f"seq_{name}.pkl"],
                validator=lambda n=name: validate_embedding(embeddings / f"seq_{n}.pkl"),
                recovery="rerun",
            )
            for name in ("train", "lt30", "30_50")
        ]),
        Stage("Extracting ESM-IF1 embeddings", [
            action(
                f"esmif1_{name}",
                py(
                    "embeddings.extract_esm_if1", "--structures", structure_dirs[name],
                    "--output", embeddings / f"str_{name}.pkl", "--device", device,
                ),
                inputs=[structure_dirs[name], manifests[name]],
                outputs=[embeddings / f"str_{name}.pkl"],
                validator=lambda n=name: validate_embedding(embeddings / f"str_{n}.pkl"),
                recovery="rerun",
            )
            for name in ("train", "lt30", "30_50")
        ]),
        Stage("Building feature stores", [
            action(
                "feature_stores",
                py(
                    "scripts.build_feature_stores", "--embedding-root", embeddings,
                    "--output-root", stores,
                ),
                inputs=[embeddings / f"{prefix}_{name}.pkl" for prefix in ("seq", "str") for name in ("train", "lt30", "30_50")],
                outputs=[stores / f"{prefix}_{name}" for prefix in ("seq", "str") for name in ("train", "lt30", "30_50")],
                validator=lambda: validate_feature_stores(stores, embeddings),
                recovery="rerun",
            )
        ]),
        Stage("Exporting the effective dataset", [
            action(
                "effective_export",
                py(
                    "preprocessing.export_effective", "--data-root", processed,
                    "--feature-root", stores, "--output-dir", effective,
                ),
                inputs=[*processed_files.values(), *[stores / f"{prefix}_{name}" for prefix in ("seq", "str") for name in ("train", "lt30", "30_50")]],
                outputs=[effective / "train_cleaned_with_structure.csv", effective / "test_30_cleaned_with_structure.csv", effective / "test_30_50_clean.csv", effective / "effective_split_summary.json"],
                validator=lambda: validate_effective(effective),
                recovery="rerun",
            )
        ]),
        Stage("Training and evaluating GaCA", [
            action(
                "gaca_train",
                py(
                    "gaca.train", "--data-root", effective, "--feature-root", stores,
                    "--split-file", split_file, "--output-dir", outputs / "neural",
                    "--models", "gaca", "--batch-size", "128", "--hidden-dim", "512",
                    "--dropout", "0.5", "--learning-rate", "0.0001",
                    "--weight-decay", "0.005", "--label-smoothing", "0.1",
                    "--validation-ratio", "0.1", "--max-epochs", "120",
                    "--patience", "30", "--workers", str(config.workers),
                    "--seed", str(config.seed), "--amp",
                ),
                inputs=[effective, stores, split_file],
                outputs=[outputs / "neural" / "gaca"],
                validator=lambda: validate_neural_model(outputs / "neural", "gaca", True),
                recovery="rerun",
            )
        ]),
    ]
    return stages


def build_full_only_stages(config):
    data = config.data_root
    outputs = config.output_root
    effective = data / "processed" / "effective"
    stores = data / "embeddings" / "feature_store"
    structures = data / "structures"
    split_file = ROOT / "metadata" / "fixed_fit_validation_split.csv"
    train_csv = effective / "train_cleaned_with_structure.csv"
    test_csv = {"lt30": effective / "test_30_cleaned_with_structure.csv", "30_50": effective / "test_30_50_clean.csv"}
    neural = outputs / "neural"
    common_train = [
        "--data-root", effective, "--feature-root", stores, "--split-file", split_file,
        "--output-dir", neural, "--max-epochs", "120", "--patience", "30",
        "--workers", str(config.workers), "--seed", str(config.seed), "--amp",
    ]
    ablation_names = ("seq_only", "str_only", "mean_vector_gate", "attention_concat", "mean_concat", "cross_attention")
    blast = outputs / "blastp"
    fold = outputs / "foldseek"

    stages = [
        Stage("Running neural ablations", [
            action(
                "ablations_standard",
                py("ablations.train", *common_train, "--models", "seq_only", "str_only", "mean_vector_gate", "attention_concat", "mean_concat", "--batch-size", "128"),
                inputs=[effective, stores, split_file],
                outputs=[neural / name for name in ablation_names[:-1]],
                validator=lambda: [validate_neural_model(neural, name) for name in ablation_names[:-1]],
                recovery="rerun",
            ),
            action(
                "ablation_cross_attention",
                py("ablations.train", *common_train, "--models", "cross_attention", "--cross-attention-batch-size", "8", "--cross-attention-dim", "512", "--cross-attention-heads", "4"),
                inputs=[effective, stores, split_file],
                outputs=[neural / "cross_attention"],
                validator=lambda: validate_neural_model(neural, "cross_attention"),
                recovery="rerun",
            ),
        ]),
        Stage("Running classifier baselines", [
            action(
                "classical_baselines",
                py("baselines.classical.train", "--data-root", effective, "--feature-root", stores, "--output-dir", outputs, "--models", "knn", "svm", "random_forest"),
                inputs=[effective, stores],
                outputs=[outputs / "baselines" / name for name in ("knn", "svm", "random_forest")],
                validator=lambda: validate_classical(outputs, ("knn", "svm", "random_forest")),
                recovery="rerun",
            ),
            action(
                "lightgbm",
                py("baselines.lightgbm.train", "--data-root", effective, "--feature-root", stores, "--split-file", split_file, "--output-dir", outputs, "--max-estimators", "500", "--early-stopping-rounds", "30"),
                inputs=[effective, stores, split_file],
                outputs=[outputs / "baselines" / "lightgbm"],
                validator=lambda: validate_classical(outputs, ("lightgbm",)),
                recovery="rerun",
            ),
        ]),
    ]

    blast_actions = []
    fasta = {"train": blast / "train.fasta", "lt30": blast / "lt30.fasta", "30_50": blast / "30_50.fasta"}
    for name, source in (("train", train_csv), *test_csv.items()):
        blast_actions.append(action(f"blast_fasta_{name}", py("baselines.blastp.make_fasta", "--csv", source, "--output", fasta[name]), inputs=[source], outputs=[fasta[name]], validator=lambda p=fasta[name]: require_files([p]), recovery="rerun"))
    db_prefix = blast / "train_db"
    blast_actions.append(action("blast_database", ["makeblastdb", "-in", fasta["train"], "-dbtype", "prot", "-out", db_prefix], inputs=[fasta["train"]], outputs=[blast / "train_db.pin", blast / "train_db.phr", blast / "train_db.psq"], validator=lambda: require_files([blast / "train_db.pin", blast / "train_db.phr", blast / "train_db.psq"]), recovery="rerun", tools=("makeblastdb",)))
    for name in ("lt30", "30_50"):
        hits = blast / f"hits_{name}.tsv"
        blast_actions.append(action(f"blast_search_{name}", ["blastp", "-query", fasta[name], "-db", db_prefix, "-num_threads", str(config.workers), "-max_target_seqs", "100", "-outfmt", "6 qseqid sseqid pident evalue bitscore length qcovs", "-out", hits], inputs=[fasta[name], *[blast / f"train_db.{suffix}" for suffix in ("pin", "phr", "psq")]], outputs=[hits], validator=lambda p=hits: require_files([p]), recovery="rerun", tools=("blastp",)))
        blast_actions.append(action(f"blast_evaluate_{name}", py("baselines.retrieval.evaluate", "--method", "blastp", "--train-csv", train_csv, "--test-csv", test_csv[name], "--hits", hits, "--output-dir", blast, "--dataset", name), inputs=[train_csv, test_csv[name], hits], outputs=[blast / f"predictions_{name}.csv", blast / f"metrics_{name}.json"], validator=lambda n=name: validate_retrieval(blast, n), recovery="rerun"))
    stages.append(Stage("Running BLASTp", blast_actions))

    fold_actions = []
    effective_structures = {name: structures / "effective" / name for name in ("train", "lt30", "30_50")}
    source_structures = {"train": structures / "train", "lt30": structures / "lt30", "30_50": structures / "30_50"}
    for name, source in (("train", train_csv), *test_csv.items()):
        fold_actions.append(action(f"foldseek_prepare_{name}", py("baselines.foldseek.prepare_structures", "--csv", source, "--source-dir", source_structures[name], "--output-dir", effective_structures[name]), inputs=[source, source_structures[name]], outputs=[effective_structures[name]], validator=lambda p=effective_structures[name]: require_files(list(p.glob("*.pdb"))), recovery="rerun"))
    for name in ("lt30", "30_50"):
        hits = fold / f"hits_{name}.tsv"
        fold_actions.append(action(f"foldseek_search_{name}", ["foldseek", "easy-search", effective_structures[name], effective_structures["train"], hits, fold / f"tmp_{name}", "--threads", str(config.workers), "--search-type", "0", "--compressed", "0", "--exact-tmscore", "0", "--format-output", "query,target,fident,evalue,bits,alntmscore,qtmscore,ttmscore"], inputs=[effective_structures[name], effective_structures["train"]], outputs=[hits], validator=lambda p=hits: require_files([p]), recovery="rerun", tools=("foldseek",)))
        fold_actions.append(action(f"foldseek_evaluate_{name}", py("baselines.retrieval.evaluate", "--method", "foldseek", "--train-csv", train_csv, "--test-csv", test_csv[name], "--hits", hits, "--output-dir", fold, "--dataset", name), inputs=[train_csv, test_csv[name], hits], outputs=[fold / f"predictions_{name}.csv", fold / f"metrics_{name}.json"], validator=lambda n=name: validate_retrieval(fold, n), recovery="rerun"))
    stages.append(Stage("Running Foldseek", fold_actions))

    final_metrics = outputs / "final_metrics.csv"
    paper_tables = outputs / "paper_tables"
    figures = outputs / "figures"
    stages.append(Stage("Aggregating metrics, statistics, and figures", [
        action("collect_metrics", py("evaluation.collect_metrics", "--neural-root", neural, "--baseline-root", outputs / "baselines", "--blastp-root", blast, "--foldseek-root", fold, "--output", final_metrics), inputs=[neural, outputs / "baselines", blast, fold], outputs=[final_metrics], validator=lambda: require_files([final_metrics]), recovery="rerun"),
        action("paired_statistics", py("evaluation.paired_statistics", "--gaca-predictions", neural / "gaca", "--blastp-predictions", blast, "--foldseek-predictions", fold, "--train-csv", train_csv, "--output-dir", paper_tables), inputs=[neural / "gaca", blast, fold, train_csv], outputs=[paper_tables / "updated_gaca_vs_blastp.csv", paper_tables / "updated_gaca_vs_foldseek.csv", paper_tables / "updated_long_tail.csv", paper_tables / "updated_table8_cases.csv", paper_tables / "updated_derived_results.json"], validator=lambda: require_files([paper_tables / "updated_gaca_vs_blastp.csv", paper_tables / "updated_gaca_vs_foldseek.csv", paper_tables / "updated_long_tail.csv", paper_tables / "updated_table8_cases.csv", paper_tables / "updated_derived_results.json"]), recovery="rerun"),
        action("figures", py("figures.plot", "--metrics", final_metrics, "--gaca-results", neural / "gaca", "--output-dir", figures), inputs=[final_metrics, neural / "gaca"], outputs=[figures], validator=lambda: validate_figures(figures), recovery="rerun"),
    ]))
    return stages


def build_verify_stages(config):
    verification = config.output_root / "verification" / "archived_verification.json"
    return [
        Stage("Validating archived experiment artifacts", [
            action(
                "verify_archived",
                py("scripts.verify_archived", "--output", verification),
                inputs=[ROOT / "metadata", ROOT / "checkpoints", ROOT / "reference_results"],
                outputs=[verification],
                validator=lambda: require_files([verification]),
                recovery="rerun",
            )
        ]),
        Stage("Running release guard tests", [
            action(
                "release_guard_tests",
                [PYTHON, "-m", "unittest", "discover", "-s", "tests", "-v"],
                inputs=[ROOT / "tests", ROOT / "common", ROOT / "embeddings", ROOT / "evaluation"],
                recovery="rerun",
            )
        ]),
    ]


def build_pipeline(mode, config):
    if mode == "verify":
        return build_verify_stages(config)
    stages = build_gaca_stages(config)
    if mode == "full":
        stages.extend(build_full_only_stages(config))
    return stages


def print_summary(config):
    metrics_path = config.output_root / "neural" / "gaca" / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["test_results"]
    observed = {name: float(metrics[name]["level_4_accuracy"]) for name in REFERENCE}
    print("\n" + "=" * 56)
    print("GaCA reproduction completed\n")
    print(f"<30%    : {observed['lt30'] * 100:.2f}%")
    print(f"30-50%  : {observed['30_50'] * 100:.2f}%")
    print("\nHistorical checkpoint reference:")
    print(f"<30%    : {REFERENCE['lt30'] * 100:.2f}%")
    print(f"30-50%  : {REFERENCE['30_50'] * 100:.2f}%")
    print(f"\nResults: {config.output_root.resolve()}")
    print("=" * 56)


def add_common_options(parser):
    parser.add_argument("--data-root", type=Path, default=ROOT / "data", help="raw, processed, structure, and embedding root")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs", help="training, evaluation, and receipt output root")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto", help="gaca/full require CUDA; cpu is accepted only by verify")
    parser.add_argument("--workers", type=int, default=4, help="data-loader and external-tool worker count")
    parser.add_argument("--seed", type=int, default=HISTORICAL_SEED, help="fixed at 42 by the historical protocol")
    controls = parser.add_mutually_exclusive_group()
    controls.add_argument("--resume", action="store_true", help="skip only stages with matching receipts and valid outputs")
    controls.add_argument("--force", action="store_true", help="rerun mutable stages without deleting the CARE checkout")


def build_parser():
    parser = argparse.ArgumentParser(description="GaCA historical-protocol reproduction")
    subparsers = parser.add_subparsers(dest="mode", required=True)
    for mode, help_text in (
        ("verify", "validate archived artifacts without downloading or training"),
        ("gaca", "run the minimal fresh GaCA experiment pipeline"),
        ("full", "run the integrated paper experiments except CLEAN"),
    ):
        child = subparsers.add_parser(mode, help=help_text)
        add_common_options(child)
    return parser


def normalize_config(args, parser):
    if args.seed != HISTORICAL_SEED:
        parser.error("the historical reproduction protocol fixes --seed at 42")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.mode in {"gaca", "full"} and args.device == "cpu":
        parser.error("fresh historical-protocol training requires CUDA; --device cpu is only suitable for verify")
    return Config(
        data_root=args.data_root.resolve(),
        output_root=args.output_root.resolve(),
        device=args.device,
        workers=args.workers,
        seed=args.seed,
        resume=args.resume,
        force=args.force,
    )


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    config = normalize_config(args, parser)
    if args.mode == "verify":
        print("This verifies the archived experiment artifacts; it is not a fresh end-to-end reproduction.\n")
    try:
        if args.mode in {"gaca", "full"}:
            import torch
            if not torch.cuda.is_available():
                raise PipelineError("CUDA is required by the frozen GaCA training protocol")
        PipelineRunner(config).run(build_pipeline(args.mode, config))
        if args.mode in {"gaca", "full"}:
            print_summary(config)
    except (PipelineError, subprocess.CalledProcessError, OSError, ValueError, ImportError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
