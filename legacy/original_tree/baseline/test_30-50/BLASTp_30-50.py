import pandas as pd
import os
import subprocess
from tqdm import tqdm



TRAIN_CSV = '/home/lihaotian/new_ec/train_cleaned_with_structure.csv'

TEST_CSV = '/home/lihaotian/new_ec/test_30_50_clean.csv'

TEMP_DIR = 'blast_temp_30_50'
os.makedirs(TEMP_DIR, exist_ok=True)


def dataframe_to_fasta(df, filename):
    with open(filename, 'w') as f:
        for idx, row in df.iterrows():
            entry = str(row['Entry']).strip()
            ec = str(row['EC number']).strip()
            seq = str(row['Sequence']).strip()
            f.write(f">{entry}|{ec}\n{seq}\n")


def run_blast():
    print(">>> 1. 准备数据...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)




    print(f"   Train samples: {len(train_df)}")
    print(f"   Test samples:  {len(test_df)}")

    train_fasta = os.path.join(TEMP_DIR, 'train.fasta')
    test_fasta = os.path.join(TEMP_DIR, 'test.fasta')
    db_name = os.path.join(TEMP_DIR, 'train_db')
    out_file = os.path.join(TEMP_DIR, 'results.txt')

    dataframe_to_fasta(train_df, train_fasta)
    dataframe_to_fasta(test_df, test_fasta)

    print(">>> 2. 建库 (makeblastdb)...")
    subprocess.run(f"makeblastdb -in {train_fasta} -dbtype prot -out {db_name}", shell=True, check=True)

    print(">>> 3. 比对 (blastp)...")

    cmd = f"blastp -query {test_fasta} -db {db_name} -out {out_file} -outfmt 6 -max_target_seqs 1 -num_threads 16 -evalue 100"
    subprocess.run(cmd, shell=True, check=True)

    print(">>> 4. 解析结果...")
    y_true_map = {str(row['Entry']): str(row['EC number']) for _, row in test_df.iterrows()}
    preds = {}

    if os.path.exists(out_file):
        with open(out_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                q_id = parts[0].split('|')[0]
                t_ec = parts[1].split('|')[1]
                if q_id not in preds: preds[q_id] = t_ec

    y_true, y_pred = [], []
    for q_id, true_ec in y_true_map.items():
        y_true.append(true_ec)
        y_pred.append(preds.get(q_id, "No_Hit"))


    c1, c2, c3, c4 = 0, 0, 0, 0
    total = len(y_true)
    for t, p in zip(y_true, y_pred):
        if p == "No_Hit": continue
        tp, pp = t.split('.'), p.split('.')
        if len(pp) >= 1 and tp[0] == pp[0]: c1 += 1
        if len(pp) >= 2 and tp[:2] == pp[:2]: c2 += 1
        if len(pp) >= 3 and tp[:3] == pp[:3]: c3 += 1
        if t == p: c4 += 1

    print("\n" + "=" * 40)
    print(f"📊 BLASTp Results (30-50 Set, N={total})")
    print("=" * 40)
    print(f"   Level 1 Acc: {c1 / total * 100:.2f}%")
    print(f"   Level 2 Acc: {c2 / total * 100:.2f}%")
    print(f"   Level 3 Acc: {c3 / total * 100:.2f}%")
    print(f"   Level 4 Acc: {c4 / total * 100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    run_blast()