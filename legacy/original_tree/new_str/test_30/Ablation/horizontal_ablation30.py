import pandas as pd
import pickle
import numpy as np
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-3
EPOCHS = 100
PATIENCE = 30
LABEL_SMOOTHING = 0.1


SEQ_TRAIN_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/1024_train_seq_embeddings_esm2_mean.pkl'
SEQ_TEST_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/1024_test_seq_embeddings_esm2_mean.pkl'

STR_TRAIN_PKL = '/home/lihaotian/new_ec/new_str/test_30/train_structure_embeddings_esm_if.pkl'
STR_TEST_PKL = '/home/lihaotian/new_ec/new_str/test_30/test_structure_embeddings_esm_if.pkl'

TRAIN_CSV = '/home/lihaotian/new_ec/train_cleaned_with_structure.csv'
TEST_CSV = '/home/lihaotian/new_ec/test_30_cleaned_with_structure.csv'



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pkl(path):
    if not os.path.exists(path): raise FileNotFoundError(f"❌ 找不到文件: {path}")
    with open(path, 'rb') as f: return pickle.load(f)



def normalize_features(X):
    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1e-6
    return X / norm


def align_data(csv_path, seq_pkl, str_pkl, label_encoder=None, is_train=True):
    df = pd.read_csv(csv_path)
    df['Entry'] = df['Entry'].astype(str)
    seq_data = load_pkl(seq_pkl)
    str_data = load_pkl(str_pkl)

    common_ids = set(df['Entry']) & set(seq_data.keys()) & set(str_data.keys())
    print(f"📊 {os.path.basename(csv_path)}: 有效样本 {len(common_ids)}")

    X_seq_list, X_str_list, y_list = [], [], []
    for _, row in df.iterrows():
        pid = str(row['Entry'])
        if pid in common_ids:
            s_emb = seq_data[pid]
            if isinstance(s_emb, torch.Tensor): s_emb = s_emb.cpu().numpy()
            if s_emb.ndim == 2: s_emb = s_emb.mean(axis=0)

            t_emb = str_data[pid]
            if isinstance(t_emb, torch.Tensor): t_emb = t_emb.cpu().numpy()
            if t_emb.ndim == 2: t_emb = t_emb.mean(axis=0)

            X_seq_list.append(s_emb)
            X_str_list.append(t_emb)
            y_list.append(str(row['EC number']))

    X_seq = np.array(X_seq_list)
    X_str = np.array(X_str_list)
    y = np.array(y_list)


    X_seq = normalize_features(X_seq)
    X_str = normalize_features(X_str)

    if is_train:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        return X_seq, X_str, y_enc, le
    else:
        if label_encoder is None: raise ValueError("Test mode needs label_encoder")
        known_classes = set(label_encoder.classes_)
        mask = np.array([label in known_classes for label in y])

        X_seq = X_seq[mask]
        X_str = X_str[mask]
        y_clean_raw = y[mask]
        y_enc = label_encoder.transform(y_clean_raw)
        print(f"🛡️ 剔除 Unseen EC 后剩余: {len(y_enc)}")
        return X_seq, X_str, y_enc, y_clean_raw



def calc_hierarchical_metrics(y_true_str, y_pred_str):
    c1, c2, c3, c4 = 0, 0, 0, 0
    total = len(y_true_str)
    for t, p in zip(y_true_str, y_pred_str):
        tp, pp = t.split('.'), p.split('.')
        if len(pp) >= 1 and tp[0] == pp[0]: c1 += 1
        if len(pp) >= 2 and tp[:2] == pp[:2]: c2 += 1
        if len(pp) >= 3 and tp[:3] == pp[:3]: c3 += 1
        if t == p: c4 += 1
    return c1 / total, c2 / total, c3 / total, c4 / total


class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def train_and_eval(name, X_train, y_train, X_test, y_test_raw, le, num_classes):
    seed_everything(SEED)
    print(f"\n⚡ Running Experiment: [{name}] (Input Dim: {X_train.shape[1]})")


    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED, shuffle=True)


    X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    X_te_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    dl_tr = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)
    dl_val = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)

    model = SimpleMLP(X_train.shape[1], num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_val_acc = 0.0
    patience_cnt = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in dl_tr:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()


        model.eval()
        preds_val = []
        with torch.no_grad():
            for xb, yb in dl_val:
                out = model(xb)
                preds_val.append(out.argmax(1).cpu().numpy())
            val_acc = accuracy_score(y_val, np.concatenate(preds_val))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_cnt = 0
                best_state = model.state_dict()
            else:
                patience_cnt += 1
                if patience_cnt >= PATIENCE: break


    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        logits = model(X_te_t)
        preds_indices = torch.argmax(logits, dim=1).cpu().numpy()

    preds_str = le.inverse_transform(preds_indices)
    l1, l2, l3, l4 = calc_hierarchical_metrics(y_test_raw, preds_str)

    print(f"✅ [{name}] Final Results:")
    print(f"   Level 1 Acc: {l1 * 100:.2f}%")
    print(f"   Level 2 Acc: {l2 * 100:.2f}%")
    print(f"   Level 3 Acc: {l3 * 100:.2f}%")
    print(f"   Level 4 Acc: {l4 * 100:.2f}% (Total Match)")

    return l4




print(">>> Loading Data...")
X_seq_tr, X_str_tr, y_tr_enc, le = align_data(TRAIN_CSV, SEQ_TRAIN_PKL, STR_TRAIN_PKL, is_train=True)
X_seq_te, X_str_te, y_te_enc, y_te_raw = align_data(TEST_CSV, SEQ_TEST_PKL, STR_TEST_PKL, label_encoder=le,
                                                    is_train=False)

num_classes = len(le.classes_)
print(f"类别总数: {num_classes}")

print("\n" + "=" * 50)
print("🚀 开始横向消融实验 (Horizontal Ablation)")
print("=" * 50)


train_and_eval("Sequence Only", X_seq_tr, y_tr_enc, X_seq_te, y_te_raw, le, num_classes)


train_and_eval("Structure Only", X_str_tr, y_tr_enc, X_str_te, y_te_raw, le, num_classes)



X_concat_tr = np.hstack([X_seq_tr, X_str_tr])
X_concat_te = np.hstack([X_seq_te, X_str_te])
train_and_eval("Simple Concatenation", X_concat_tr, y_tr_enc, X_concat_te, y_te_raw, le, num_classes)