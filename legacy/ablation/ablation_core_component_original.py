import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
import random


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 5e-3
EPOCHS = 100
PATIENCE = 30
VAL_RATIO = 0.1
SEED = 42
LABEL_SMOOTHING = 0.1
DROPOUT = 0.5


SEQ_TRAIN_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/1024_train_seq_embeddings_esm2_mean.pkl'
SEQ_TEST_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/30_50_seq_embeddings_esm2_mean.pkl'

STR_TRAIN_PKL = '/home/lihaotian/new_ec/Structures_Eembeddings_use_ESM-IF/train_structure_embeddings_esm_if.pkl'
STR_TEST_PKL = '/home/lihaotian/new_ec/Structures_Eembeddings_use_ESM-IF/30-50test_structure_embeddings_esm_if.pkl'

TRAIN_CSV = '/home/lihaotian/new_ec/train_cleaned_with_structure.csv'
TEST_CSV = '/home/lihaotian/new_ec/test_30_50_clean.csv'



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_pkl(path):
    if not os.path.exists(path): raise FileNotFoundError(f"❌ 找不到 {path}")
    with open(path, 'rb') as f: return pickle.load(f)


def normalize_features(X):

    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1e-6
    return X / norm



class GACMADataset(Dataset):
    def __init__(self, csv_path, seq_path, str_path, is_train=True, le=None):
        df = pd.read_csv(csv_path)
        df['Entry'] = df['Entry'].astype(str)
        seq_dict = load_pkl(seq_path)
        str_dict = load_pkl(str_path)

        common_ids = set(df['Entry']) & set(seq_dict.keys()) & set(str_dict.keys())

        self.seq_data, self.str_data, self.labels = [], [], []
        for _, row in df.iterrows():
            pid = str(row['Entry'])
            if pid in common_ids:
                s = seq_dict[pid]
                t = str_dict[pid]
                if isinstance(s, torch.Tensor): s = s.cpu().numpy()
                if isinstance(t, torch.Tensor): t = t.cpu().numpy()
                if s.ndim > 1: s = s.mean(axis=0)
                if t.ndim > 1: t = t.mean(axis=0)
                self.seq_data.append(s)
                self.str_data.append(t)
                self.labels.append(str(row['EC number']))


        self.seq_data = normalize_features(np.array(self.seq_data))
        self.str_data = normalize_features(np.array(self.str_data))

        if is_train:
            self.le = LabelEncoder()
            self.encoded_labels = self.le.fit_transform(self.labels)
        else:
            self.le = le
            known_classes = set(le.classes_)
            mask = np.array([L in known_classes for L in self.labels])
            self.seq_data = self.seq_data[mask]
            self.str_data = self.str_data[mask]
            self.labels = [self.labels[i] for i in range(len(self.labels)) if mask[i]]
            self.encoded_labels = le.transform(self.labels)

    def __len__(self):
        return len(self.encoded_labels)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.seq_data[idx]), torch.FloatTensor(self.str_data[idx]),
                torch.LongTensor([self.encoded_labels[idx]]))






class Model_NoGate(nn.Module):
    def __init__(self, num_classes, seq_dim=1280, str_dim=512, hidden_dim=512):
        super().__init__()
        self.seq_proj = nn.Sequential(nn.Linear(seq_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(DROPOUT))
        self.str_proj = nn.Sequential(nn.Linear(str_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(DROPOUT))


        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, num_classes)
        )

    def forward(self, seq_emb, str_emb):
        h1 = self.seq_proj(seq_emb)
        h2 = self.str_proj(str_emb)
        combined = torch.cat([h1, h2], dim=1)
        f = self.fusion(combined)
        return self.classifier(f)




class Model_WithAttn(nn.Module):
    def __init__(self, num_classes, seq_dim=1280, str_dim=512, hidden_dim=512):
        super().__init__()
        self.seq_proj = nn.Sequential(nn.Linear(seq_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(DROPOUT))
        self.str_proj = nn.Sequential(nn.Linear(str_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(DROPOUT))


        self.W_Q_seq, self.W_K_str, self.W_V_str = nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim,
                                                                                                hidden_dim), nn.Linear(
            hidden_dim, hidden_dim)
        self.W_Q_str, self.W_K_seq, self.W_V_seq = nn.Linear(hidden_dim, hidden_dim), nn.Linear(hidden_dim,
                                                                                                hidden_dim), nn.Linear(
            hidden_dim, hidden_dim)


        self.gating_net = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
                                        nn.Sigmoid())

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(512, num_classes)
        )

    def global_attn(self, q, k, v):
        score = (q * k).sum(dim=-1, keepdim=True) / (512 ** 0.5)
        return v * torch.sigmoid(score)

    def forward(self, seq_emb, str_emb):
        h1 = self.seq_proj(seq_emb)
        h2 = self.str_proj(str_emb)

        h1_enh = self.global_attn(self.W_Q_seq(h1), self.W_K_str(h2), self.W_V_str(h2))
        h2_enh = self.global_attn(self.W_Q_str(h2), self.W_K_seq(h1), self.W_V_seq(h1))

        f1 = h1 + h1_enh
        f2 = h2 + h2_enh

        g = self.gating_net(torch.cat([f1, f2], dim=1))
        f = g * f2 + (1 - g) * f1
        return self.classifier(f)



class GACMA_Final(nn.Module):
    def __init__(self, num_classes, seq_dim=1280, str_dim=512, hidden_dim=512):
        super().__init__()
        self.seq_proj = nn.Sequential(nn.Linear(seq_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(DROPOUT))
        self.str_proj = nn.Sequential(nn.Linear(str_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(DROPOUT))

        self.gating_net = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
                                        nn.Sigmoid())

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(512, num_classes)
        )

    def forward(self, seq_emb, str_emb):
        h1 = self.seq_proj(seq_emb)
        h2 = self.str_proj(str_emb)
        combined = torch.cat([h1, h2], dim=1)
        g = self.gating_net(combined)
        f = g * h2 + (1 - g) * h1
        return self.classifier(f)



def run_experiment(model_class, name, train_dl, val_dl, test_dl, num_classes, le):
    seed_everything(SEED)
    print(f"\n⚡ Running Ablation: [{name}]")

    model = model_class(num_classes).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_acc = 0.0
    patience_cnt = 0
    best_state = None


    for epoch in range(EPOCHS):
        model.train()
        for seq, stru, label in train_dl:
            seq, stru, label = seq.to(device), stru.to(device), label.to(device).view(-1)
            optimizer.zero_grad()
            out = model(seq, stru)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()


        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for seq, stru, label in val_dl:
                seq, stru, label = seq.to(device), stru.to(device), label.to(device).view(-1)
                out = model(seq, stru)
                preds.extend(out.argmax(1).cpu().numpy())
                acts.extend(label.cpu().numpy())
        val_acc = accuracy_score(acts, preds)

        if val_acc > best_acc:
            best_acc = val_acc
            patience_cnt = 0
            best_state = model.state_dict()
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE: break


    model.load_state_dict(best_state)
    model.eval()
    preds_str, acts_str = [], []
    with torch.no_grad():
        for seq, stru, label in test_dl:
            seq, stru, label = seq.to(device), stru.to(device), label.to(device).view(-1)
            out = model(seq, stru)
            p_idx = out.argmax(1).cpu().numpy()
            l_idx = label.cpu().numpy()
            preds_str.extend(le.inverse_transform(p_idx))
            acts_str.extend(le.inverse_transform(l_idx))


    def calc_metrics(y_true, y_pred):
        c1, c2, c3, c4 = 0, 0, 0, 0
        total = len(y_true)
        for t, p in zip(y_true, y_pred):
            tp, pp = t.split('.'), p.split('.')
            if len(pp) >= 1 and tp[0] == pp[0]: c1 += 1
            if len(pp) >= 2 and tp[:2] == pp[:2]: c2 += 1
            if len(pp) >= 3 and tp[:3] == pp[:3]: c3 += 1
            if t == p: c4 += 1
        return c1 / total, c2 / total, c3 / total, c4 / total

    l1, l2, l3, l4 = calc_metrics(acts_str, preds_str)

    print(f"✅ [{name}] Final Results:")
    print(f"   Level 1 Acc: {l1 * 100:.2f}%")
    print(f"   Level 2 Acc: {l2 * 100:.2f}%")
    print(f"   Level 3 Acc: {l3 * 100:.2f}%")
    print(f"   Level 4 Acc: {l4 * 100:.2f}% (Total Match)")
    return l4


def run():
    print(">>> Loading Data...")
    full_train_ds = GACMADataset(TRAIN_CSV, SEQ_TRAIN_PKL, STR_TRAIN_PKL, is_train=True)
    le = full_train_ds.le
    train_idx, val_idx = train_test_split(range(len(full_train_ds)), test_size=VAL_RATIO, random_state=SEED,
                                          shuffle=True)

    train_dl = DataLoader(Subset(full_train_ds, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl = DataLoader(Subset(full_train_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dl = DataLoader(GACMADataset(TEST_CSV, SEQ_TEST_PKL, STR_TEST_PKL, is_train=False, le=le),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    num_classes = len(le.classes_)

    print("\n" + "=" * 50)
    print("🚀 开始纵向消融实验 (Internal Ablation)")
    print("=" * 50)


    run_experiment(Model_NoGate, "w/o Gating", train_dl, val_dl, test_dl, num_classes, le)


    run_experiment(Model_WithAttn, "w/ Cross-Attention", train_dl, val_dl, test_dl, num_classes, le)


    run_experiment(GACMA_Final, "GACMA Final (Ours)", train_dl, val_dl, test_dl, num_classes, le)


if __name__ == "__main__":
    run()