import pandas as pd
import pickle
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 当前运行设备: {device}")



TRAIN_PKL = '/home/lihaotian/new_ec/Sequences_Embedding_use_ESM-2/1024_train_seq_embeddings_esm2_mean.pkl'
TEST_PKL = '/home/lihaotian/new_ec/Sequences_Embedding_use_ESM-2/1024_test_seq_embeddings_esm2_mean.pkl'


TRAIN_CSV = '/home/lihaotian/new_ec/train_cleaned_with_structure.csv'
TEST_CSV = '/home/lihaotian/new_ec/CARE_datasets/splits/task1/30_protein_test.csv'
GACMA_TEST_PKL = '/home/lihaotian/new_ec/processed_multimodal_datasets/multimodal_test_30.pkl'




def perform_mean_pooling(emb):
    """
    解决内存溢出：对 (Seq_Len, Hidden) 取平均 -> (Hidden,)
    注意：新生成的 v2 特征已经是 (1280,) 的 1D 向量了。
    但保留这个函数是安全的，因为它会直接返回 1D 向量，不进行额外操作。
    """
    if isinstance(emb, torch.Tensor):
        if emb.dim() == 2: return torch.mean(emb, dim=0)
    elif isinstance(emb, np.ndarray):
        if len(emb.shape) == 2: return np.mean(emb, axis=0)
    return emb


def load_data_gpu(pkl_path, csv_path, filter_ids=None):
    print(f"\n[Loading] CSV: {csv_path}")
    if not os.path.exists(csv_path):
        print(f"❌ 错误: 找不到 CSV {csv_path}")
        return None, None

    df = pd.read_csv(csv_path)
    df['Entry'] = df['Entry'].astype(str)

    print(f"[Loading] PKL: {pkl_path}")
    if not os.path.exists(pkl_path):
        print(f"❌ 错误: 找不到 PKL {pkl_path}")
        return None, None

    with open(pkl_path, 'rb') as f:
        emb_data = pickle.load(f)


    emb_dict = {}
    if isinstance(emb_data, dict):

        emb_dict = emb_data
    elif isinstance(emb_data, list):

        if len(emb_data) > 0 and isinstance(emb_data[0], dict):
            key = 'Entry' if 'Entry' in emb_data[0] else 'protein_id'
            for item in emb_data:
                if key in item and 'embedding' in item:
                    emb_dict[item[key]] = item['embedding']

    X_list, y_list = [], []

    match_count = 0

    for idx, row in df.iterrows():
        pid = str(row['Entry'])
        if pid in emb_dict:
            if filter_ids is not None and pid not in filter_ids:
                continue

            match_count += 1
            emb = perform_mean_pooling(emb_dict[pid])

            if isinstance(emb, np.ndarray):
                emb = torch.from_numpy(emb)
            elif not isinstance(emb, torch.Tensor):
                emb = torch.tensor(emb)
            X_list.append(emb.float())
            y_list.append(str(row['EC number']))

    print(f"   > 成功匹配样本数: {match_count}")

    if len(X_list) == 0: return None, None
    return torch.stack(X_list), np.array(y_list)


def get_gacma_ids(pkl_path):
    if not os.path.exists(pkl_path): return None
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    return set([item['protein_id'] for item in data])


def calc_metrics(y_true, y_pred, model_name):
    """计算 L1, L2, L3, L4 全套指标"""
    acc_l1, acc_l2, acc_l3, acc_l4 = [], [], [], []
    for t, p in zip(y_true, y_pred):
        t_parts, p_parts = t.split('.'), p.split('.')
        acc_l1.append(1 if len(p_parts) >= 1 and t_parts[:1] == p_parts[:1] else 0)
        acc_l2.append(1 if len(p_parts) >= 2 and t_parts[:2] == p_parts[:2] else 0)
        acc_l3.append(1 if len(p_parts) >= 3 and t_parts[:3] == p_parts[:3] else 0)
        acc_l4.append(1 if t == p else 0)

    print(f">> {model_name} Results (N={len(y_true)}):")
    print(
        f"   L1: {np.mean(acc_l1) * 100:.2f}% | L2: {np.mean(acc_l2) * 100:.2f}% | L3: {np.mean(acc_l3) * 100:.2f}% | L4: {np.mean(acc_l4) * 100:.2f}%")




def run_knn_gpu(X_train, y_train, X_test, y_test):
    print(f"\n[GPU] Running KNN (Cosine, k=1)...")
    X_train_norm = torch.nn.functional.normalize(X_train, p=2, dim=1).to(device)
    X_test_norm = torch.nn.functional.normalize(X_test, p=2, dim=1).to(device)

    chunk_size = 100
    y_pred_all = []
    for i in range(0, len(X_test_norm), chunk_size):
        batch_test = X_test_norm[i: i + chunk_size]
        sim_matrix = torch.mm(batch_test, X_train_norm.t())
        _, indices = sim_matrix.topk(k=1, dim=1)
        indices = indices.cpu().numpy().flatten()
        y_pred_all.extend(y_train[indices])
    calc_metrics(y_test, y_pred_all, "ESM2 + KNN")


def run_logreg_gpu(X_train, y_train, X_test, y_test):
    print(f"\n[GPU] Running Logistic Regression...")
    unique_labels = sorted(list(set(y_train)))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for i, label in enumerate(unique_labels)}

    y_train_idx = torch.tensor([label_to_idx[y] for y in y_train]).long().to(device)
    X_train_gpu = X_train.to(device)
    X_test_gpu = X_test.to(device)


    model = nn.Linear(X_train.shape[1], len(unique_labels)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_train_gpu, y_train_idx)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model.train()
    for epoch in range(100):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_gpu)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()

    y_pred = [idx_to_label[idx] for idx in preds]
    calc_metrics(y_test, y_pred, "ESM2 + Logistic Regression")


def run_mlp_gpu(X_train, y_train, X_test, y_test):
    print(f"\n[GPU] Running MLP (2-Layer)...")
    unique_labels = sorted(list(set(y_train)))
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for i, label in enumerate(unique_labels)}

    y_train_idx = torch.tensor([label_to_idx[y] for y in y_train]).long().to(device)
    X_train_gpu = X_train.to(device)
    X_test_gpu = X_test.to(device)

    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, len(unique_labels))
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_train_gpu, y_train_idx)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model.train()
    for epoch in range(50):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        outputs = model(X_test_gpu)
        _, preds = torch.max(outputs, 1)
        preds = preds.cpu().numpy()

    y_pred = [idx_to_label[idx] for idx in preds]
    calc_metrics(y_test, y_pred, "ESM2 + MLP")




filter_ids = get_gacma_ids(GACMA_TEST_PKL)
if filter_ids: print(f"✅ 已加载 {len(filter_ids)} 个核心样本 ID (来自多模态文件)")

X_train, y_train = load_data_gpu(TRAIN_PKL, TRAIN_CSV)
X_test, y_test = load_data_gpu(TEST_PKL, TEST_CSV, filter_ids)

if X_train is not None:
    print(f"✅ 数据加载完毕: Train {X_train.shape}, Test {X_test.shape}")



    print("\n>>> 正在对齐测试集 (去除训练集中未见过的 EC)...")
    seen_ecs = set(y_train)
    mask = np.array([ec in seen_ecs for ec in y_test])

    X_test = X_test[torch.tensor(mask)]
    y_test = y_test[mask]

    print(f"✅ 最终评测样本数: {len(y_test)} (预期应为 243 左右)")
    print(f"   剔除样本数: {len(mask) - len(y_test)}")
    print("=" * 50)



    run_knn_gpu(X_train, y_train, X_test, y_test)


    run_logreg_gpu(X_train, y_train, X_test, y_test)


    run_mlp_gpu(X_train, y_train, X_test, y_test)


    print(f"\n[CPU] Running Random Forest (n=100)...")
    rf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
    rf.fit(X_train.numpy(), y_train)
    calc_metrics(y_test, rf.predict(X_test.numpy()), "ESM2 + Random Forest")