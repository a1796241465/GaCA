
python - <<'PY'
from CLEAN.utils import csv_to_fasta, retrive_esm1b_embedding, compute_esm_distance
for name in ['gaca_train_strict', 'gaca_test_lt30_strict', 'gaca_test_30_50_strict']:
    csv_to_fasta(f'data/{name}.csv', f'data/{name}.fasta')
    retrive_esm1b_embedding(name)
compute_esm_distance('gaca_train_strict')
PY

python train-triplet.py --training_data gaca_train_strict --model_name gaca_train_strict_triplet --epoch 2000

python - <<'PY'
from CLEAN.infer import infer_maxsep
infer_maxsep('gaca_train_strict', 'gaca_test_lt30_strict', report_metrics=True, pretrained=False, model_name='gaca_train_strict_triplet')
infer_maxsep('gaca_train_strict', 'gaca_test_30_50_strict', report_metrics=True, pretrained=False, model_name='gaca_train_strict_triplet')
PY
