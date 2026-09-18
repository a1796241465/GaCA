# Pipeline audit

This is the initial packaging review, not evidence of a fresh end-to-end reproduction. The independent audit in `INDEPENDENT_AUDIT.md` supersedes its verification claims.

The release was checked against the completed 73.25%/84.91% rerun protocol.

- The main runner accepts an explicit data root. The previous unified runner had a hard-coded repository-relative data path.
- The fixed fit/validation split is checked for duplicate, missing, extra, and unknown protein identifiers before training.
- Final protocol counts are checked before neural training unless `--skip-protocol-check` is explicitly supplied for a different dataset.
- ESM-IF1 backbone atoms are grouped by residue before N/CA/C coordinates are stacked. The previous extractor collected each atom type independently and truncated the arrays, which could mispair atoms when a residue lacked a backbone atom.
- LightGBM uses the fixed label-protected split for early stopping, is refit on all effective training proteins at the selected iteration, and uses the row-wise L2 normalization applied in the final rerun.
- Retrieval outputs retain top-hit identity and score fields, allowing paired tables to be rebuilt directly from new BLASTp and Foldseek runs.
- The README uses direct commands. Earlier wrapper scripts are retained only under `legacy/`.

The main source tree passes syntax compilation. Non-embedding CLI imports were checked; ESM-2 and ESM-IF1 imports were not verified locally because fair-esm was unavailable. Full GPU training and the external BLASTp/Foldseek searches were not repeated locally; their completed outputs are retained under `reference_results/`.
