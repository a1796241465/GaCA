import pandas as pd
import os
import subprocess
import pickle
from tqdm import tqdm



TRAIN_CSV = '/home/lihaotian/new_ec/train_cleaned_with_structure.csv'
TEST_CSV = '/home/lihaotian/new_ec/test_30_cleaned_with_structure.csv'


SEQ_TRAIN_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/1024_train_seq_embeddings_esm2_mean.pkl'
SEQ_TEST_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/1024_test_seq_embeddings_esm2_mean.pkl'

STR_TRAIN_PKL = '/home/lihaotian/new_ec/Structures_Eembeddings_use_ESM-IF/train_structure_embeddings_esm_if.pkl'
STR_TEST_PKL = '/home/lihaotian/new_ec/Structures_Eembeddings_use_ESM-IF/test_structure_embeddings_esm_if.pkl'

TEMP_DIR = 'blast_temp'
os.makedirs(TEMP_DIR, exist_ok=True)


def load_pkl_keys(path):
    print(f"Loading keys from {os.path.basename(path)}...")
    if not os.path.exists(path): raise FileNotFoundError(f"❌ 找不到 {path}")
    with open(path, 'rb') as f:
        data = pickle.load(f)
        return set(data.keys())


def filter_data(df, seq_keys, str_keys, train_ecs=None):






    valid_entries = []


    common_ids = set(df['Entry'].astype(str)) & seq_keys & str_keys
    print(f"   >>> 三方交集 (CSV & Seq & Str): {len(common_ids)}")

    df_filtered = df[df['Entry'].astype(str).isin(common_ids)].copy()


    if train_ecs is not None:
        before = len(df_filtered)
        df_filtered = df_filtered[df_filtered['EC number'].isin(train_ecs)]
        print(f"   >>> 剔除 Unseen EC: {before} -> {len(df_filtered)}")

    return df_filtered


def dataframe_to_fasta(df, filename):
    with open(filename, 'w') as f:
        for idx, row in df.iterrows():
            entry = str(row['Entry']).strip()
            ec = str(row['EC number']).strip()
            seq = str(row['Sequence']).strip()

            f.write(f">{entry}|{ec}\n{seq}\n")


def run_blast_strict():
    print(">>> 1. 加载 PKL 索引以对齐样本...")
    train_seq_keys = load_pkl_keys(SEQ_TRAIN_PKL)
    train_str_keys = load_pkl_keys(STR_TRAIN_PKL)
    test_seq_keys = load_pkl_keys(SEQ_TEST_PKL)
    test_str_keys = load_pkl_keys(STR_TEST_PKL)

    print("\n>>> 2. 筛选训练集 (Training Set Alignment)...")
    train_df = pd.read_csv(TRAIN_CSV)
    train_df_valid = filter_data(train_df, train_seq_keys, train_str_keys)
    train_ecs = set(train_df_valid['EC number'].astype(str))
    print(f"✅ 训练集最终样本数: {len(train_df_valid)} (GACMA应为 13671)")

    print("\n>>> 3. 筛选测试集 (Test Set Alignment)...")
    test_df = pd.read_csv(TEST_CSV)
    test_df_valid = filter_data(test_df, test_seq_keys, test_str_keys, train_ecs)
    print(f"✅ 测试集最终样本数: {len(test_df_valid)} (GACMA应为 243)")

    print("\n>>> 4. 生成 FASTA 并运行 BLAST...")
    train_fasta = os.path.join(TEMP_DIR, 'train_strict.fasta')
    test_fasta = os.path.join(TEMP_DIR, 'test_strict.fasta')
    db_name = os.path.join(TEMP_DIR, 'train_db_strict')
    out_file = os.path.join(TEMP_DIR, 'results_strict.txt')

    dataframe_to_fasta(train_df_valid, train_fasta)
    dataframe_to_fasta(test_df_valid, test_fasta)


    print("   Running makeblastdb...")
    subprocess.run(f"makeblastdb -in {train_fasta} -dbtype prot -out {db_name}", shell=True, check=True)


    print("   Running blastp...")

    cmd = f"blastp -query {test_fasta} -db {db_name} -out {out_file} -outfmt 6 -max_target_seqs 1 -num_threads 16 -evalue 100"
    subprocess.run(cmd, shell=True, check=True)

    print("\n>>> 5. 计算指标...")
    y_true_map = {str(row['Entry']): str(row['EC number']) for _, row in test_df_valid.iterrows()}
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

    print("=" * 40)
    print(f"📊 BLASTp Results (Samples: {total})")
    print(f"   Level 1 Acc: {c1 / total * 100:.2f}%")
    print(f"   Level 2 Acc: {c2 / total * 100:.2f}%")
    print(f"   Level 3 Acc: {c3 / total * 100:.2f}%")
    print(f"   Level 4 Acc: {c4 / total * 100:.2f}%")
    print("=" * 40)


if __name__ == "__main__":
    run_blast_strict()