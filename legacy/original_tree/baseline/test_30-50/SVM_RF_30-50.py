import numpy as np
import pickle
import pandas as pd
import os
import torch
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


TRAIN_CSV = '/home/lihaotian/new_ec/train_cleaned_with_structure.csv'
TEST_CSV = '/home/lihaotian/new_ec/test_30_50_clean.csv'


SEQ_TRAIN_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/1024_train_seq_embeddings_esm2_mean.pkl'
STR_TRAIN_PKL = '/home/lihaotian/new_ec/new_str/test_30-50/train_structure_embeddings_esm_if.pkl'


SEQ_TEST_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/30_50_seq_embeddings_esm2_mean.pkl'
STR_TEST_PKL = '/home/lihaotian/new_ec/new_str/test_30-50/30-50test_structure_embeddings_esm_if.pkl'


def load_pkl(path):
    print(f"Loading {os.path.basename(path)}...")
    with open(path, 'rb') as f: return pickle.load(f)


def normalize_features(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1e-6
    return X / norm


def prepare_data(csv_path, seq_path, str_path, is_train=True, le=None):
    df = pd.read_csv(csv_path)
    df['Entry'] = df['Entry'].astype(str)
    seq_dict = load_pkl(seq_path)
    str_dict = load_pkl(str_path)

    common_ids = set(df['Entry']) & set(seq_dict.keys()) & set(str_dict.keys())

    X_list, y_list = [], []
    for _, row in df.iterrows():
        pid = str(row['Entry'])
        if pid in common_ids:

            s = seq_dict[pid]
            if isinstance(s, torch.Tensor): s = s.cpu().numpy()
            if s.ndim > 1: s = s.mean(axis=0)


            t = str_dict[pid]
            if isinstance(t, torch.Tensor): t = t.cpu().numpy()
            if t.ndim > 1: t = t.mean(axis=0)


            feat = np.concatenate([s, t])
            X_list.append(feat)
            y_list.append(str(row['EC number']))

    X = np.array(X_list)
    X = normalize_features(X)
    y = np.array(y_list)

    if is_train:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        return X, y_enc, le
    else:

        known = set(le.classes_)
        mask = [label in known for label in y]
        X = X[mask]
        y_raw = [y[i] for i in range(len(y)) if mask[i]]
        y_enc = le.transform(y_raw)
        return X, y_enc, y_raw


def calc_hierarchical_metrics(y_true, y_pred):
    c1, c2, c3, c4 = 0, 0, 0, 0
    total = len(y_true)
    for t, p in zip(y_true, y_pred):
        tp, pp = t.split('.'), p.split('.')
        if len(pp) >= 1 and tp[0] == pp[0]: c1 += 1
        if len(pp) >= 2 and tp[:2] == pp[:2]: c2 += 1
        if len(pp) >= 3 and tp[:3] == pp[:3]: c3 += 1
        if t == p: c4 += 1
    return c1 / total, c2 / total, c3 / total, c4 / total


def run_ml():
    print(">>> 1. 准备训练数据 (Concat Seq+Str)...")
    X_train, y_train, le = prepare_data(TRAIN_CSV, SEQ_TRAIN_PKL, STR_TRAIN_PKL, is_train=True)

    print(">>> 2. 准备测试数据 (30-50)...")
    X_test, y_test, y_test_raw = prepare_data(TEST_CSV, SEQ_TEST_PKL, STR_TEST_PKL, is_train=False, le=le)

    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")




    print("\n🌲 Running Random Forest (n=100)...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=16, random_state=42)
    rf.fit(X_train, y_train)
    preds_rf = le.inverse_transform(rf.predict(X_test))

    l1, l2, l3, l4 = calc_hierarchical_metrics(y_test_raw, preds_rf)
    print(f"✅ RF Results: L1={l1 * 100:.1f}% | L4={l4 * 100:.2f}%")




    print("\n📐 Running SVM (Linear)...")
    svm = LinearSVC(C=1.0, max_iter=1000, random_state=42)
    svm.fit(X_train, y_train)
    preds_svm = le.inverse_transform(svm.predict(X_test))

    sl1, sl2, sl3, sl4 = calc_hierarchical_metrics(y_test_raw, preds_svm)
    print(f"✅ SVM Results: L1={sl1 * 100:.1f}% | L4={sl4 * 100:.2f}%")

    print("\n" + "=" * 50)
    print("📊 Final ML Baselines Report (30-50 Set)")
    print("=" * 50)
    print(f"{'Method':<15} | {'L1':<8} | {'L2':<8} | {'L3':<8} | {'L4':<8}")
    print("-" * 55)
    print(f"{'Random Forest':<15} | {l1 * 100:.2f}%   | {l2 * 100:.2f}%   | {l3 * 100:.2f}%   | {l4 * 100:.2f}%")
    print(f"{'SVM (Linear)':<15} | {sl1 * 100:.2f}%   | {sl2 * 100:.2f}%   | {sl3 * 100:.2f}%   | {sl4 * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    run_ml()