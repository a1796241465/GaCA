# GaCA

GaCA is a sequence-structure fusion model for Level-4 enzyme commission (EC)
prediction under low sequence identity. This repository contains the audited code,
fixed protocol metadata, checkpoint, and archived outputs for the run that obtained
73.25% accuracy on the CARE `<30%` subset and 84.91% on the CARE `30-50%` subset.

## Quick Start

Run the following commands from the repository root on Linux with an NVIDIA GPU.

### Installation

```bash
conda env create -f environment.yml
conda activate gaca
bash scripts/setup_dependencies.sh
```

The setup script must be run after activating the environment. It installs the
PyTorch/CUDA-specific `torch-scatter` wheel and checks the ESM-IF1 dependencies.
BLAST+ and Foldseek are additional `PATH` dependencies for the `full` mode.

### Main GaCA reproduction

```bash
python reproduce.py gaca
```

This command runs the CARE and AlphaFold preparation, preprocessing, ESM-2 and
ESM-IF1 extraction, guarded feature-store construction, strict effective-set export,
fixed-split GaCA training, full-data refit, and test evaluation. If a long run is
interrupted, resume it with:

```bash
python reproduce.py gaca --resume
```

### Full paper experiments

```bash
python reproduce.py full
```

This adds the neural ablations, KNN, SVM, Random Forest, LightGBM, BLASTp,
Foldseek, metric aggregation, paired statistics, and manuscript figures. CLEAN is
not included because it requires a separate upstream environment and model files.

### Archived verification

```bash
python reproduce.py verify
```

This verifies the archived experiment artifacts; it is not a fresh end-to-end
reproduction. It checks the archived checkpoint, fixed split, protocol metadata,
predictions, paired-statistics inputs, and release guard tests without downloading,
extracting embeddings, or training a model.

Use `python reproduce.py gaca --help` for path and execution options. The main
options are `--data-root`, `--output-root`, `--device`, `--workers`, `--seed`,
`--resume`, and `--force`. Scientific settings remain fixed to the audited historical
protocol; in particular, the wrapper rejects a seed other than 42. `--force` reruns
mutable stages but does not destructively replace an existing CARE checkout.

## Expected Results

| CARE subset | Evaluated proteins | Historical checkpoint accuracy |
|---|---:|---:|
| `<30%` | 243 | 73.25% |
| `30-50%` | 477 | 84.91% |

Fresh retraining can differ because of software, hardware, and stochastic training
effects. The command reports the newly observed values next to these archived
references; it does not require bitwise equality with checkpoint replay.

## Resume and Output Validation

The orchestrator records receipts under `outputs/.reproduce/receipts/`. A receipt
contains the exact command plus relevant input and output signatures. `--resume`
skips an action only when its receipt and output contract both validate. AlphaFold's
manifest-based recovery is reused for interrupted downloads. Feature stores are
rebuilt through the audited store implementation and validated against source path,
size, modification time, and feature dimension.

ESM extraction does not provide partial checkpoints. An interrupted extractor is
rerun without repeating completed earlier stages. GaCA training also has no safe
mid-epoch resume checkpoint; an interrupted training action restarts training while
retaining all previously completed data and embedding stages. Ambiguous cached
outputs are not silently accepted.

## Reproducibility Status

Archived inputs, checkpoint, split, label mapping, metrics, and paired statistics
have been independently verified. Checkpoint inference was rerun on the original
embeddings and reproduced 178/243 correct predictions (73.25%) on `<30%` and
405/477 (84.91%) on `30-50%`, with protein-level predictions identical to the
archived results.

A fresh CARE/AlphaFold download, fresh ESM-2/ESM-IF1 extraction, complete neural
and classical-model retraining, BLASTp/Foldseek searches, and cross-environment
identity of training trajectories have not been validated. Checkpoint replay is not
presented as a fresh end-to-end reproduction. See
[`docs/INDEPENDENT_AUDIT.md`](docs/INDEPENDENT_AUDIT.md) for the evidence and
remaining limitations.

The benchmark splits come from the
[CARE paper](https://proceedings.neurips.cc/paper_files/paper/2024/hash/05a7ad45d75a3082d7a3a70de8743140-Abstract-Datasets_and_Benchmarks_Track.html).
CARE revisions and AlphaFold DB structures may change. This release does not
distribute the original feature arrays or pin all structures by content checksum.

## Repository Structure

```text
reproduce.py          unified verify, gaca, and full entry point
scripts/              orchestration, setup, and archived verification helpers
data_download/        CARE and AlphaFold DB download utilities
preprocessing/        filtering, sequence-structure checks, and effective splits
embeddings/           ESM-2/ESM-IF1 extraction and feature-store construction
gaca/                 GaCA model, training, full-data refit, and inference
ablations/            single-modality and fusion ablations
baselines/            classical, LightGBM, BLASTp, Foldseek, and CLEAN code
evaluation/           metric aggregation and paired statistics
figures/              manuscript figure generation
metadata/             fixed split, protocol, and reference label mapping
checkpoints/          archived 73.25%/84.91% GaCA checkpoint
reference_results/    archived metrics, predictions, and figures
tests/                release guards and orchestration tests
legacy/               original research code retained for provenance
docs/                 audit evidence and the manual command-by-command pipeline
```

## Advanced / Manual Pipeline

Every underlying Step 1-10 command remains documented in
[`docs/MANUAL_PIPELINE.md`](docs/MANUAL_PIPELINE.md). Use the manual path for
debugging individual modules or inspecting exact input/output contracts.

The supplied environment targets the server runs (Python 3.9, PyTorch 2.2.2) but is
not an exact lock of the original GaCA environment. Local records report Python
3.12.4, PyTorch 2.4.0, and an RTX 4060 Laptop GPU. The archived
`gaca_environment_and_config.json` was overwritten by a cross-attention run and is
not a GaCA-specific launch record; this provenance limitation remains documented.
