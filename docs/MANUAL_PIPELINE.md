# Manual GaCA Pipeline

This document preserves the explicit commands behind `python reproduce.py gaca` and
`python reproduce.py full`. Run all commands from the repository root. The unified
entry point is recommended for routine use because it validates outputs and records
resume metadata.

## 1. Download CARE and select candidates

```bash
python -m data_download.download_care --output-dir data/raw/care

python -m preprocessing.select_candidates \
  --care-task-dir data/raw/care/splits/task1 \
  --output-dir data/processed/candidates \
  --max-length 600 --max-per-ec 25 --seed 42
```

## 2. Download AlphaFold DB structures

```bash
python -m data_download.alphafold \
  --csv data/processed/candidates/train_candidates.csv \
  --output-dir data/structures/train \
  --manifest data/structures/train_manifest.csv

python -m data_download.alphafold \
  --csv data/processed/candidates/lt30_candidates.csv \
  --output-dir data/structures/lt30 \
  --manifest data/structures/lt30_manifest.csv

python -m data_download.alphafold \
  --csv data/processed/candidates/30_50_candidates.csv \
  --output-dir data/structures/30_50 \
  --manifest data/structures/30_50_manifest.csv
```

## 3. Prepare and validate metadata

```bash
python -m preprocessing.prepare_splits \
  --care-task-dir data/raw/care/splits/task1 \
  --train-structures data/structures/train \
  --lt30-structures data/structures/lt30 \
  --30-50-structures data/structures/30_50 \
  --train-manifest data/structures/train_manifest.csv \
  --output-dir data/processed \
  --max-length 600 --max-per-ec 25 --min-confidence 70 --seed 42

python -m preprocessing.align_sequences \
  --csv data/processed/train_cleaned_with_structure.csv \
  --structures data/structures/train \
  --output-pkl data/processed/train_validated_sequences.pkl \
  --output-csv data/processed/train_alignment.csv \
  --stats data/processed/train_alignment.json

python -m preprocessing.align_sequences \
  --csv data/processed/test_30_cleaned_with_structure.csv \
  --structures data/structures/lt30 \
  --output-pkl data/processed/lt30_validated_sequences.pkl \
  --output-csv data/processed/lt30_alignment.csv \
  --stats data/processed/lt30_alignment.json
```

The completed rerun used the CARE sequence column directly for the `30-50%` ESM-2
input and the corresponding AlphaFold DB structures for ESM-IF1.

## 4. Extract residue embeddings

```bash
python -m embeddings.extract_esm2 \
  --input data/processed/train_validated_sequences.pkl \
  --output data/embeddings/seq_train.pkl --batch-size 16

python -m embeddings.extract_esm2 \
  --input data/processed/lt30_validated_sequences.pkl \
  --output data/embeddings/seq_lt30.pkl --batch-size 16

python -m embeddings.extract_esm2 \
  --input data/processed/test_30_50_clean.csv \
  --output data/embeddings/seq_30_50.pkl --batch-size 16

python -m embeddings.extract_esm_if1 \
  --structures data/structures/train \
  --output data/embeddings/str_train.pkl

python -m embeddings.extract_esm_if1 \
  --structures data/structures/lt30 \
  --output data/embeddings/str_lt30.pkl

python -m embeddings.extract_esm_if1 \
  --structures data/structures/30_50 \
  --output data/embeddings/str_30_50.pkl

python -m embeddings.build_store \
  --output-root data/embeddings/feature_store

python -m preprocessing.export_effective \
  --data-root data/processed \
  --feature-root data/embeddings/feature_store \
  --output-dir data/processed/effective
```

The final protocol check expects 13,671 training proteins, 3,811 Level-4 labels,
243 evaluated `<30%` proteins, and 477 evaluated `30-50%` proteins.

## 5. Train GaCA

```bash
python -m gaca.train \
  --data-root data/processed/effective \
  --feature-root data/embeddings/feature_store \
  --split-file metadata/fixed_fit_validation_split.csv \
  --output-dir outputs/neural \
  --models gaca \
  --batch-size 128 --max-epochs 120 --patience 30 \
  --workers 4 --seed 42 --amp
```

The command selects the epoch count on the fixed label-protected 12,303/1,368
fit/validation partition. It then initializes a new model and trains it on all
13,671 effective training proteins for the selected number of epochs.

## 6. Run the ablations

```bash
python -m ablations.train \
  --data-root data/processed/effective \
  --feature-root data/embeddings/feature_store \
  --split-file metadata/fixed_fit_validation_split.csv \
  --output-dir outputs/neural \
  --models seq_only str_only mean_vector_gate attention_concat mean_concat \
  --batch-size 128 --max-epochs 120 --patience 30 \
  --workers 4 --seed 42 --amp

python -m ablations.train \
  --data-root data/processed/effective \
  --feature-root data/embeddings/feature_store \
  --split-file metadata/fixed_fit_validation_split.csv \
  --output-dir outputs/neural \
  --models cross_attention \
  --cross-attention-batch-size 8 --cross-attention-dim 512 \
  --cross-attention-heads 4 --max-epochs 120 --patience 30 \
  --workers 4 --seed 42 --amp
```

## 7. Run classifier baselines

```bash
python -m baselines.classical.train \
  --data-root data/processed/effective \
  --feature-root data/embeddings/feature_store \
  --output-dir outputs \
  --models knn svm random_forest

python -m baselines.lightgbm.train \
  --data-root data/processed/effective \
  --feature-root data/embeddings/feature_store \
  --split-file metadata/fixed_fit_validation_split.csv \
  --output-dir outputs \
  --max-estimators 500 --early-stopping-rounds 30
```

The CLEAN adapter is retained under `baselines/clean/`. CLEAN requires its upstream
model files and environment and is not part of the unified `full` command.

## 8. Run BLASTp

```bash
mkdir -p outputs/blastp

python -m baselines.blastp.make_fasta --csv data/processed/effective/train_cleaned_with_structure.csv --output outputs/blastp/train.fasta
python -m baselines.blastp.make_fasta --csv data/processed/effective/test_30_cleaned_with_structure.csv --output outputs/blastp/lt30.fasta
python -m baselines.blastp.make_fasta --csv data/processed/effective/test_30_50_clean.csv --output outputs/blastp/30_50.fasta

makeblastdb -in outputs/blastp/train.fasta -dbtype prot -out outputs/blastp/train_db

blastp -query outputs/blastp/lt30.fasta -db outputs/blastp/train_db -num_threads 16 -max_target_seqs 100 -outfmt '6 qseqid sseqid pident evalue bitscore length qcovs' -out outputs/blastp/hits_lt30.tsv
blastp -query outputs/blastp/30_50.fasta -db outputs/blastp/train_db -num_threads 16 -max_target_seqs 100 -outfmt '6 qseqid sseqid pident evalue bitscore length qcovs' -out outputs/blastp/hits_30_50.tsv

python -m baselines.retrieval.evaluate --method blastp --train-csv data/processed/effective/train_cleaned_with_structure.csv --test-csv data/processed/effective/test_30_cleaned_with_structure.csv --hits outputs/blastp/hits_lt30.tsv --output-dir outputs/blastp --dataset lt30
python -m baselines.retrieval.evaluate --method blastp --train-csv data/processed/effective/train_cleaned_with_structure.csv --test-csv data/processed/effective/test_30_50_clean.csv --hits outputs/blastp/hits_30_50.tsv --output-dir outputs/blastp --dataset 30_50
```

## 9. Run Foldseek

```bash
mkdir -p outputs/foldseek

python -m baselines.foldseek.prepare_structures --csv data/processed/effective/train_cleaned_with_structure.csv --source-dir data/structures/train --output-dir data/structures/effective/train
python -m baselines.foldseek.prepare_structures --csv data/processed/effective/test_30_cleaned_with_structure.csv --source-dir data/structures/lt30 --output-dir data/structures/effective/lt30
python -m baselines.foldseek.prepare_structures --csv data/processed/effective/test_30_50_clean.csv --source-dir data/structures/30_50 --output-dir data/structures/effective/30_50

foldseek easy-search data/structures/effective/lt30 data/structures/effective/train outputs/foldseek/hits_lt30.tsv outputs/foldseek/tmp_lt30 --threads 16 --search-type 0 --compressed 0 --exact-tmscore 0 --format-output 'query,target,fident,evalue,bits,alntmscore,qtmscore,ttmscore'
foldseek easy-search data/structures/effective/30_50 data/structures/effective/train outputs/foldseek/hits_30_50.tsv outputs/foldseek/tmp_30_50 --threads 16 --search-type 0 --compressed 0 --exact-tmscore 0 --format-output 'query,target,fident,evalue,bits,alntmscore,qtmscore,ttmscore'

python -m baselines.retrieval.evaluate --method foldseek --train-csv data/processed/effective/train_cleaned_with_structure.csv --test-csv data/processed/effective/test_30_cleaned_with_structure.csv --hits outputs/foldseek/hits_lt30.tsv --output-dir outputs/foldseek --dataset lt30
python -m baselines.retrieval.evaluate --method foldseek --train-csv data/processed/effective/train_cleaned_with_structure.csv --test-csv data/processed/effective/test_30_50_clean.csv --hits outputs/foldseek/hits_30_50.tsv --output-dir outputs/foldseek --dataset 30_50
```

## 10. Aggregate results, paired tests, and figures

```bash
python -m evaluation.collect_metrics \
  --neural-root outputs/neural \
  --baseline-root outputs/baselines \
  --blastp-root outputs/blastp \
  --foldseek-root outputs/foldseek \
  --output outputs/final_metrics.csv

python -m evaluation.paired_statistics \
  --gaca-predictions outputs/neural/gaca \
  --blastp-predictions outputs/blastp \
  --foldseek-predictions outputs/foldseek \
  --train-csv data/processed/effective/train_cleaned_with_structure.csv \
  --output-dir outputs/paper_tables

python -m figures.plot \
  --metrics outputs/final_metrics.csv \
  --gaca-results outputs/neural/gaca \
  --output-dir outputs/figures
```
