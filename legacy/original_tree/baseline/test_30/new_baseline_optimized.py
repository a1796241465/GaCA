import numpy as np
import pickle
import pandas as pd
import os
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
from datetime import datetime


TRAIN_CSV = '/home/lihaotian/new_ec/train_cleaned_with_structure.csv'
TEST_CSV = '/home/lihaotian/new_ec/test_30_cleaned_with_structure.csv'

SEQ_TRAIN_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/1024_train_seq_embeddings_esm2_mean.pkl'
STR_TRAIN_PKL = '/home/lihaotian/new_ec/new_str/test_30/train_structure_embeddings_esm_if.pkl'

SEQ_TEST_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/1024_test_seq_embeddings_esm2_mean.pkl'
STR_TEST_PKL = '/home/lihaotian/new_ec/new_str/test_30/test_structure_embeddings_esm_if.pkl'



def log_time(msg):
    """打印带时间戳的日志"""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def load_pkl(path):
    log_time(f"Loading {os.path.basename(path)}...")
    with open(path, 'rb') as f:
        return pickle.load(f)


def prepare_data(csv_path, seq_path, str_path, is_train=True, le=None):
    """准备训练或测试数据"""
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
            t = str_dict[pid]


            if isinstance(s, torch.Tensor):
                s = s.cpu().numpy()
            if isinstance(t, torch.Tensor):
                t = t.cpu().numpy()


            if s.ndim > 1:
                s = s.mean(axis=0)
            if t.ndim > 1:
                t = t.mean(axis=0)

            feat = np.concatenate([s, t])
            X_list.append(feat)
            y_list.append(str(row['EC number']))

    X = np.array(X_list)
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
    """计算分层准确率(EC号的4个层级)"""
    c1 = c2 = c3 = c4 = 0
    total = len(y_true)
    for t, p in zip(y_true, y_pred):
        tp, pp = t.split('.'), p.split('.')
        if tp[0] == pp[0]:
            c1 += 1
        if len(tp) >= 2 and len(pp) >= 2 and tp[:2] == pp[:2]:
            c2 += 1
        if len(tp) >= 3 and len(pp) >= 3 and tp[:3] == pp[:3]:
            c3 += 1
        if t == p:
            c4 += 1
    return c1 / total, c2 / total, c3 / total, c4 / total



def run_all_baselines():
    log_time(">>> Loading Training Data...")
    X_train, y_train, le = prepare_data(TRAIN_CSV, SEQ_TRAIN_PKL, STR_TRAIN_PKL, is_train=True)

    log_time(">>> Loading Test Data...")
    X_test, y_test, y_test_raw = prepare_data(TEST_CSV, SEQ_TEST_PKL, STR_TEST_PKL, is_train=False, le=le)

    log_time(f"Train: {X_train.shape}, Test: {X_test.shape}")
    log_time(f"Number of classes: {len(le.classes_)}")
    print("\n====== Running Baseline Models ======\n")

    results = {}


    log_time("🔍 Training KNN (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(X_train, y_train)
    log_time("KNN: Predicting...")
    preds = le.inverse_transform(knn.predict(X_test))
    results["KNN(5)"] = calc_hierarchical_metrics(y_test_raw, preds)
    log_time("✅ KNN Done")


    log_time("\n🧠 Training MLP...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation='relu',
        solver='adam',
        batch_size=256,
        max_iter=30,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=5,
        verbose=False
    )
    mlp.fit(X_train, y_train)
    log_time("MLP: Predicting...")
    preds = le.inverse_transform(mlp.predict(X_test))
    results["MLP(512-256)"] = calc_hierarchical_metrics(y_test_raw, preds)
    log_time("✅ MLP Done")


    log_time("\n🌿 Training LightGBM (Optimized)...")
    

    lgbm = lgb.LGBMClassifier(
        num_leaves=31,
        max_depth=10,
        learning_rate=0.1,
        n_estimators=100,
        objective='multiclass',
        num_class=len(le.classes_),
        n_jobs=8,
        random_state=42,
        verbose=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1
    )
    
    log_time(f"Training LightGBM with {len(le.classes_)} classes...")
    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=False),
            lgb.log_evaluation(period=20)
        ]
    )
    
    log_time("LightGBM: Predicting...")
    preds = le.inverse_transform(lgbm.predict(X_test))
    results["LightGBM"] = calc_hierarchical_metrics(y_test_raw, preds)
    log_time("✅ LightGBM Done")


    print("\n" + "=" * 60)
    print("📊 Final Multi-Model EC Classification Results")
    print("=" * 60)
    print(f"{'Model':<18} | {'L1':<8} | {'L2':<8} | {'L3':<8} | {'L4':<8}")
    print("-" * 60)

    for name, (l1, l2, l3, l4) in results.items():
        print(f"{name:<18} | {l1 * 100:.2f}% | {l2 * 100:.2f}% | {l3 * 100:.2f}% | {l4 * 100:.2f}%")

    print("=" * 60)

    return results


if __name__ == "__main__":
    run_all_baselines()
