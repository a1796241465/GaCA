import numpy as np
import pickle
import pandas as pd
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score



SEQ_TRAIN_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/1024_train_seq_embeddings_esm2_mean.pkl'
SEQ_TEST_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/1024_test_seq_embeddings_esm2_mean.pkl'
TRAIN_CSV = '/home/lihaotian/new_ec/train_cleaned_with_structure.csv'
TEST_CSV = '/home/lihaotian/new_ec/test_30_cleaned_with_structure.csv'


def load_pkl(path):
    with open(path, 'rb') as f: return pickle.load(f)


def normalize_features(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1e-6
    return X / norm


def prepare_data(csv_path, seq_pkl, is_train=True, le=None):
    df = pd.read_csv(csv_path)
    df['Entry'] = df['Entry'].astype(str)
    seq_dict = load_pkl(seq_pkl)


    common_ids = set(df['Entry']) & set(seq_dict.keys())

    X, y = [], []
    for _, row in df.iterrows():
        pid = str(row['Entry'])
        if pid in common_ids:
            s = seq_dict[pid]

            if not isinstance(s, np.ndarray): s = s.cpu().numpy()
            if s.ndim > 1: s = s.mean(axis=0)

            X.append(s)
            y.append(str(row['EC number']))

    X = np.array(X)
    X = normalize_features(X)

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


def run_knn():
    print(">>> Loading Data for KNN...")
    X_train, y_train, le = prepare_data(TRAIN_CSV, SEQ_TRAIN_PKL, is_train=True)
    X_test, y_test, y_test_raw = prepare_data(TEST_CSV, SEQ_TEST_PKL, is_train=False, le=le)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")



    print(">>> Training KNN (k=1, metric=cosine)...")
    knn = KNeighborsClassifier(n_neighbors=1, metric='cosine', n_jobs=8)
    knn.fit(X_train, y_train)

    print(">>> Predicting...")
    preds = knn.predict(X_test)
    preds_str = le.inverse_transform(preds)


    c1, c2, c3, c4 = 0, 0, 0, 0
    total = len(y_test_raw)
    for t, p in zip(y_test_raw, preds_str):
        tp, pp = t.split('.'), p.split('.')
        if len(pp) >= 1 and tp[0] == pp[0]: c1 += 1
        if len(pp) >= 2 and tp[:2] == pp[:2]: c2 += 1
        if len(pp) >= 3 and tp[:3] == pp[:3]: c3 += 1
        if t == p: c4 += 1

    print("\n📊 KNN (ESM-2) Results:")
    print(f"   Level 1 Acc: {c1 / total * 100:.2f}%")
    print(f"   Level 2 Acc: {c2 / total * 100:.2f}%")
    print(f"   Level 3 Acc: {c3 / total * 100:.2f}%")
    print(f"   Level 4 Acc: {c4 / total * 100:.2f}%")


if __name__ == "__main__":
    run_knn()