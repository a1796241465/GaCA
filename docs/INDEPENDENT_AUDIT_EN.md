# Independent Scientific Reproducibility Audit of GaCA-73.25-release

Completion date: 2026-09-15. Audit target: `D:\EC\GaCA-73.25-release`.

## Executive Summary

**Direct evidence supports preservation and replay of the original results, but it has not yet been demonstrated that the same results can be recovered by the complete workflow from newly downloaded data through full retraining.**

This independent audit compared the original rerun directory, local CARE source CSV files, original feature stores, fixed split, logs, checkpoint, and per-protein predictions directly; it did not treat the previous summary as evidence of a pass. Full-test-set checkpoint inference was performed. No retraining, BLASTp/Foldseek search, or embedding re-extraction was performed.

The formal conclusions are presented in Sections A–E. Several preliminary alerts appeared in the raw probe output and are explained under `resolutions` in `audit_evidence/final_evidence.json`; the number of preliminary alerts must not be treated as the final number of FAIL findings.

## A. PASS: Verified by Evidence

### A1. Data and Split

| Check | Observed result |
|---|---|
| Effective training proteins | 13,671 |
| Fitting / validation | 12,303 / 1,368 |
| Training labels | 3,811; fitting covers all labels |
| Two evaluation subsets | 243 / 477 |
| Fixed-split CSV | Identical to the original file row by row |
| class_index / ec_label mapping | Identical to the original file row by row |
| Effective IDs, labels, and row order | Fully identical after recomputing the intersection from the original CSV files and feature indices |
| Duplicate effective IDs | None |
| Train/test and test/test ID intersections | None |
| Intersections of exact-sequence SHA256 values among the sets above | None |
| Fitting/validation exact-sequence SHA256 intersection | None |

The original CARE training CSV contains 28,316 rows and 134 duplicated-ID rows; 131 IDs have multiple EC labels. Direct indexing by ID expands duplicated rows and caused the preliminary comparison to disagree. After reconstruction using the actual `sample_training(..., 600, 25, 42)` sampling procedure and the rule that retains the first occurrence of each ID, the IDs, labels, sequences, and ordering of the original 13,727-row source table were all identical. Among the selected source IDs, 93 came from multi-label IDs. The single-label selection rule was inherited from the original experiment and was not changed in this audit.

For 21 training proteins and one 30–50% test protein, the fourth field contains an `n` prefix; these are label strings from the original source. The all-numeric EC regular-expression alert resulted from an overly strict audit probe rather than a newly introduced label mismatch. These labels remain unchanged and are still treated as four fields when hierarchical metrics are calculated.

The checks above do not cover all pairwise sequence identities, near-homolog leakage, or contamination of pretraining corpora. They therefore cannot be generalized into a finding that all forms of data leakage have been excluded.

### A2. Saved Embeddings and Sampled Structure Checks

| Store | Proteins | Original array shape |
|---|---:|---|
| seq_train | 13,671 | (4,788,944, 1280) |
| str_train | 13,727 | (4,808,447, 512) |
| seq_lt30 | 312 | (119,017, 1280) |
| str_lt30 | 314 | (120,375, 512) |
| seq_30_50 | 549 | (229,375, 1280) |
| str_30_50 | 549 | (229,375, 512) |

IDs are unique in all six indices, offsets are contiguous and valid, the sums of lengths equal the array row counts, and every dtype is float16. A sample of 1,000 rows from each array contained no non-finite values; every value in every array was not scanned. The release input-ID manifests match the original indices, and the structure-length manifests match.

Data loading queries both stores by protein ID and does not rely on matching row order between the two files. Independent pooling does not require the two modalities to have the same number of residues.

From each of the three original PDB directories, 30 structures were sampled at evenly spaced positions, for 90 total. The N/CA/C coordinates grouped by residue by the new extractor were elementwise identical to the old extractor output, and their lengths matched the original embedding indices. For sampled entries with sequence manifests, the reconstructed sequence hashes also matched. This check validated sampled coordinate and sequence inputs only; it did not rerun the ESM encoder.

### A3. Model and Checkpoint

All 29 state_dict tensors in the checkpoint are exactly identical to the original file one by one. Strict loading also covered the BatchNorm buffers rather than merely checking whether the checkpoint could load.

| Module | Trainable parameters |
|---|---:|
| seq_pool | 164,097 |
| str_pool | 65,793 |
| seq_proj | 656,896 |
| str_proj | 263,680 |
| gate | 787,456 |
| classifier | 2,218,723 |
| Total | **4,156,645** |

Each projection is an independent Linear + LayerNorm + Dropout block. The attention scorer is Linear → Tanh → Linear. The gate is Linear → ReLU → Linear → Sigmoid. Fusion is gate × structure + (1 − gate) × sequence. The classifier retains Linear, BatchNorm, GELU, Dropout, and Linear layers.

The release adds a `forward_with_gate` export interface without adding any trainable module. In synthetic tests in eval mode and train mode with a fixed random seed, the maximum absolute difference in logits between the original model and the release was 0.0. Every parameter received a gradient in the backward test. Changing masked padding values did not change eval output.

Checkpoint SHA256: `cdcd4da798a95322f712c6901ecf862a23e80180736d7f9fe7ac3251928f66ef`.

### A4. Training Configuration and Metrics

The GaCA defaults/README configuration is batch size 128, AdamW, learning rate 1e-4, weight decay 5e-3, label smoothing 0.1, hidden dimension 512, dropout 0.5, maximum 120 epochs, patience 30, seed 42, and CUDA AMP. There is no scheduler. These parameters are actually used by constructors and training loops; no principal GaCA training parameter was found to be merely declared without taking effect.

The training epoch, validation, DataLoader, split generation, and random-seed functions are AST-identical to the original code. The best epoch is updated only when validation accuracy strictly improves. The seed is then reset, a new model is initialized, and training runs on all 13,671 effective training proteins for the selected number of epochs. In the archived GaCA history, the first maximum validation accuracy occurs at epoch 75, the metrics record 75, and the full-refit history contains exactly 75 epochs. The selected epoch counts for the other six neural variants also agree with their histories.

When the fixed split file exists, `--validation-ratio` and a new `--seed` do not repartition that CSV; this is the intended fixed-split design. `--refit-only-epochs` skips tuning and is not part of the primary README command. The runtime environment JSON was overwritten in the past, so the static configuration and log checks above cannot be described as recovery of the complete original GaCA launch record.

Level-1/2/3 metrics are computed from prefixes, and Level-4 from the complete label. Filtering is based on modality availability and the training label space, not on whether test predictions are correct. The NPZ protein_ids, true_indices, and argmax values agree with the prediction CSV files. Probability rows sum correctly; gate shape is (N,512), and gate values are finite and within (0,1).

### A5. Complete Checkpoint Inference and Independent Statistical Recalculation

Using the release inference entry point, archived checkpoint, and original local feature stores, every test protein was inferred with batch size 128 and CUDA AMP:

| Subset | Correct / total | Level-4 accuracy | Per-protein predictions |
|---|---|---|---|
| <30% | 178 / 243 | 73.25% | Identical to the original results row by row |
| 30–50% | 405 / 477 | 84.91% | Identical to the original results row by row |

All 26 standardized prediction CSV files for the different methods use the same 243 / 477 IDs, have consistent ground-truth labels, and contain no duplicate counts. Independent recalculation of Level-1 through Level-4 accuracy, macro-F1, and MCC differed from the final aggregation by at most 1.11e-16 in absolute value.

The archived LightGBM mapping agrees with the fixed split. The code retrieves mean-pooled 1280+512 features by ID, concatenates them, and applies per-protein L2 normalization; it does not fit a scaler on the test set. The 500-tree limit and early stopping of 30 are passed to training. The archived best iteration is 349. LightGBM was not retrained.

| GaCA − baseline | Subset | Accuracy difference (percentage points) | 95% CI (percentage points) | Exact McNemar p |
|---|---|---:|---|---:|
| BLASTp | <30% | 6.58 | [2.47, 11.11] | 0.00522287935 |
| BLASTp | 30–50% | 3.77 | [1.26, 6.29] | 0.00509764330 |
| Foldseek | <30% | 3.29 | [-0.82, 7.41] | 0.18493334204 |
| Foldseek | 30–50% | 3.56 | [1.05, 6.08] | 0.00947530428 |

The independent implementation did not call the release's `paired_bootstrap`. It used the same paired resampling sequence to calculate correctness and macro-F1 independently, and calculated exact McNemar values from combinatorial counts. Across the eight rows in the two archived tables, the maximum differences in difference, CI, and p were below 1e-16. The macro-F1 bootstrap p-values were 0.0016, 0.0076, 0.0592, and 0.0132, all identical to the archived values.

The earlier statement that the “difference was 0.0” is valid at the display precision used in the paper, but is not proof of full experimental reproduction. This audit compared the archived tables under `reference_results/paper_tables`; it did not reopen the latest final.tex for cell-by-cell verification and did not modify the manuscript.

## B. WARNING: Not Empirically Verified

1. `environment.yml` was not installed from scratch. The local machine lacked fair-esm, so the CLI and inference behavior of both embedding extractors were not validated. `--help` succeeded for the other 17 Python modules in the README, which demonstrates only that their interfaces import successfully.
2. CARE data, AlphaFold structures, and encoder weights were not downloaded again, and embeddings were not re-extracted. The sample of 90 PDB files does not represent validation of all atomic coordinates.
3. The neural networks, LightGBM, and classical baselines were not fully retrained. Checkpoint replay does not demonstrate identical cross-platform training trajectories.
4. External BLASTp/Foldseek searches were not run. Static inspection found that the README input/output paths, FASTA/PDB and TSV column definitions, and statistical and plotting inputs connect correctly; external-tool versions and tied top-hit behavior remain unverified.
5. The choice of a single label from multi-label source records, all near-homolog relationships, and pretraining-corpus coverage are outside the verified scope.

## C. FAIL: Problems Found and Repairs

| Problem and impact | Repair file and approach | Effect on the original 73.25% |
|---|---|---|
| ESM-IF1 dependencies were omitted, so a clean environment might fail to import | `environment.yml`, `requirements.txt`, and `README.md`: added PyG, specified CUDA 12.1 for the recommended environment, and provided the matching scatter wheel and import-check command | Does not alter GaCA; installation and fresh embedding extraction remain untested |
| README conflated the server/original GaCA environments; the purported GaCA config was actually for cross-attention | Clarified README; `gaca/train.py` now saves per-model configuration for new runs; the old `PIPELINE_AUDIT.md` is marked as superseded by this report | Does not alter results; the missing historical configuration cannot be reconstructed without evidence |
| The SVM `dual` default changes across sklearn versions | `baselines/classical/train.py` explicitly sets `dual=True`, restoring the choice made under the original sklearn 1.4.2 | Does not affect GaCA; prevents a new SVM run from switching solvers, but SVM was not retrained |
| Checking only sample counts could not prevent substituted IDs, labels, or row order | `common/protocol.py`, `preprocessing/export_effective.py`, `gaca/train.py`, and `gaca/evaluate.py`: now check archived IDs, labels, row order, mapping, and split labels by default | The original inputs pass; incorrect inputs raise an error instead of silently changing samples |
| An old cache could be skipped after source embeddings changed; incorrectly shaped but evenly divisible arrays could be forcibly reshaped | `embeddings/build_store.py`: checks source path/size/mtime/dimension and rejects incorrect shapes, empty values, non-finite values, float16 overflow, and duplicate IDs after canonicalization | Valid original arrays are unchanged; stale caches require explicit rebuilding, but this is not full content-hash protection |
| A paired inner join could silently discard extra baseline IDs | `evaluation/paired_statistics.py`: first checks that IDs are nonempty, unique, and identical as sets | Current pairs are unchanged; incorrect inputs raise an error |
| AMP conditions in standalone evaluation differed from the training entry point, so CPU could also enable float16 | `gaca/evaluate.py`: restricts AMP to CUDA and validates the default protocol | Original CUDA inference is unchanged |

The new `tests/test_release_guards.py` passed 8/8 regression tests, and syntax checks passed for 40 Python files in the main tree. This audit did not change the GaCA architecture or training hyperparameters, nor did it change the legacy directory, weights, archived results, figures, or manuscript.

### Unrepaired Preprocessing Defect

`preprocessing/align_sequences.py::alignment_bounds` uses globalxx without gap penalties, and identity counts only columns in which neither side is a gap. The actual call `alignment_bounds('ACDEFG','AEDCFG')` returns `(0,6,0,6,1.0)`. This 1.0 does not reliably demonstrate that the two retained intervals are highly similar; the short example tests only the function and does not involve the minimum-length restriction in `main`.

This rule is inherited from the historical workflow. Changing alignment/scoring/coverage would change the filtering criteria and might change the effective set and the 73.25% result, so the audit did not silently replace the experimental rule. The authors must decide whether to retain the historical mode and document its limitation or separately validate a corrected filtering workflow. The sample of 90 original structures showed no sequence-hash differences, but that is insufficient to prove that all inputs are unaffected.

## D. REPRODUCIBILITY RISK

- **The original inputs are not completely frozen.** IDs, labels, sequence hashes/cropping ranges, and structure lengths are available, but the original feature arrays are not distributed and the content hashes and versions of all PDB files are not fixed. The availability and content of new downloads may differ. Stronger validation can detect errors but cannot reconstruct old inputs.
- **The environment and provenance are incomplete.** Original logs support the parameter count and batch size, and the history supports epoch 75, but the shared environment JSON was overwritten. The recommended Linux/PyTorch 2.2.2 environment differs from the locally recorded Windows/PyTorch 2.4.0 experiment environment. A fixed seed and deterministic cuDNN do not guarantee bitwise reproduction across versions.
- **LightGBM row subsampling was not enabled.** The observed default is `subsample_freq=0`, so `subsample=0.8` does not mean that training used 80% row sampling; the original experiment behaved the same way. This audit did not change it to a positive frequency because that would change the baseline algorithm.
- **The statistical interpretation of the bootstrap still requires caution.** Each macro-F1 replicate reselects the ground-truth classes present in that replicate, so the support set changes with resampling. The p-value is twice the smaller empirical-tail probability, not an exact randomization test. This audit confirmed that pairing and reported numbers can be reproduced; it did not validate the procedure's coverage/error rate or replace the test.
- **Recalculating retrieval results is not the same as rerunning retrieval.** External versions, database ordering, and handling of tied scores still require formal comparison.
- **Legacy and primary workflow are separated.** No old GaCA-GitHub absolute-path or private-environment-variable dependency was found in the primary workflow. Historical paths remain in legacy files and provenance JSON. External audit scripts read the original directories for evidentiary comparison; those directories are not runtime dependencies of the release.

## E. FINAL VERDICT

**Subject to the limitations documented in this report, the repository can be frozen as a historical research-code release; it cannot be labeled as an end-to-end reproduction verified from original downloads through full retraining.**

The core architecture, weights, fixed split, checkpoint inference, archived metrics, and paired statistics are supported by evidence. An unconditional pass cannot be given because fresh extraction/refit was not performed, the original inputs/environment are not completely frozen, and a known alignment-filtering defect remains—not because the archived 73.25% result was calculated incorrectly.

**Based on the available evidence, do I believe this code can reproduce the original 73.25% result? Mostly yes, but verification is still required.**

The decisive reasons are that the original model and weights match and all 720 predictions on the original embeddings were reproduced exactly; the fixed split and selection of 75 epochs are verifiable; but the complete chain of new downloads, extraction, and training from scratch has not been empirically demonstrated, so checkpoint replay cannot be treated as an unconditional guarantee.

## Evidence and Verification

- `audit_evidence/evidence.json`: initial item-by-item checks, including preliminary probe alerts.
- `audit_evidence/second_evidence.json`: sampled original-coordinate checks, model forward pass, and independent metric and statistical recalculation.
- `audit_evidence/final_evidence.json`: final source verification, alert resolutions, complete inference, CLI and syntax checks, and change list.
- `audit_evidence/changes.diff`: exact changes to existing files in this audit.
- `audit_evidence/inference_artifacts_summary.json`: summary of the complete checkpoint inference in this audit.
- Additional local evidence and pre-change backups: `C:\Users\a1796\gaca_independent_audit`.
  This is a historical local audit path retained for provenance and is not a runtime dependency of the released repository.

`changes.diff` was regenerated from the trusted pre-audit backups described above and the current files and passed `git apply --check`; it was not reconstructed manually from this report. Its completeness is limited to the existing files covered by the backups and the tests and audit reports added in this audit.

Lightweight regression command: `python -m unittest discover -s tests -v`; it does not train a model.

External sources were used only to check dependencies/default behavior and were not treated as new experimental evidence:

- [Official ESM-IF1 environment instructions](https://github.com/facebookresearch/esm/tree/main/examples/inverse_folding)
- [PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/2.5.3/install/installation.html) and [PyTorch 2.2 / CUDA 12.1 wheels](https://data.pyg.org/whl/torch-2.2.0+cu121.html)
- [sklearn 1.4.2 LinearSVC defaults and changes](https://scikit-learn.org/1.4/modules/generated/sklearn.svm.LinearSVC.html)
- [LightGBM 4.6.0 parameter documentation](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html)
