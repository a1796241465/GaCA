# Curated final results

This directory contains compact outputs from the final unified rerun: metrics, per-protein predictions, training histories, retrieval results, and paired/statistical tables. Large embedding matrices and model checkpoints are not duplicated here.

`final_metrics.csv` is the machine-readable summary. Accuracy, macro-F1, and MCC values are stored on the 0-1 scale. Macro-F1 is averaged over Level-4 classes represented by at least one reference protein in the corresponding test subset.

The retrieval files were generated using the same effective training proteins and evaluated test identifiers as GaCA. The included `retrieval/original_combined_*` files also contain columns from the earlier manuscript analysis; use the method-specific prediction files for new analysis.