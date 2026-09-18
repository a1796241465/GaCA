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
HIDDEN_DIM = 512
DROPOUT = 0.5

EPOCHS = 120
PATIENCE = 30
VAL_RATIO = 0.1
SEED = 42
LABEL_SMOOTHING = 0.1


SEQ_TRAIN_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/1024_train_seq_embeddings_esm2_mean.pkl'
SEQ_TEST_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/1024_test_seq_embeddings_esm2_mean.pkl'

STR_TRAIN_PKL = '/home/lihaotian/new_ec/new_str/train_structure_embeddings_esm_if.pkl'
STR_TEST_PKL = '/home/lihaotian/new_ec/new_str/test_structure_embeddings_esm_if.pkl'

TRAIN_CSV = '/home/lihaotian/new_ec/train_cleaned_with_structure.csv'
TEST_CSV = '/home/lihaotian/new_ec/test_30_cleaned_with_structure.csv'

SAVE_MODEL_PATH = 'best_GACMA_final.pth'



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
        print(f"📊 {os.path.basename(csv_path)}: 有效样本 {len(common_ids)}")

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
            print(f"🛡️ 剔除 Unseen EC 后剩余: {len(self.encoded_labels)}")

    def __len__(self):
        return len(self.encoded_labels)

    def __getitem__(self, idx):
        return (torch.FloatTensor(self.seq_data[idx]), torch.FloatTensor(self.str_data[idx]),
                torch.LongTensor([self.encoded_labels[idx]]))



class GACMA_Final(nn.Module):
    def __init__(self, num_classes, seq_dim=1280, str_dim=512, hidden_dim=512, dropout=0.5):
        super().__init__()

        self.seq_proj = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)
        )
        self.str_proj = nn.Sequential(
            nn.Linear(str_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)
        )


        self.gating_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )


        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, seq_emb, str_emb):
        seq_h = self.seq_proj(seq_emb)
        str_h = self.str_proj(str_emb)

        combined = torch.cat([seq_h, str_h], dim=1)
        g = self.gating_net(combined)

        fusion_feat = g * str_h + (1 - g) * seq_h
        return self.classifier(fusion_feat)



def run():
    seed_everything(SEED)
    print(f"🚀 启动 GACMA (Winner Version) | BS={BATCH_SIZE} | LR={LEARNING_RATE}")

    full_train_ds = GACMADataset(TRAIN_CSV, SEQ_TRAIN_PKL, STR_TRAIN_PKL, is_train=True)
    le = full_train_ds.le

    train_idx, val_idx = train_test_split(range(len(full_train_ds)), test_size=VAL_RATIO, random_state=SEED,
                                          shuffle=True)
    train_dl = DataLoader(Subset(full_train_ds, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_dl = DataLoader(Subset(full_train_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_dl = DataLoader(GACMADataset(TEST_CSV, SEQ_TEST_PKL, STR_TEST_PKL, is_train=False, le=le),
                         batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = GACMA_Final(len(le.classes_), hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)

    best_val_acc = 0.0
    patience_cnt = 0

    print(">>> Training Started...")
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

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            torch.save(model.state_dict(), SAVE_MODEL_PATH)

        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early Stopping at Ep {epoch + 1}")
                break


    print("\n" + "=" * 50)
    print("🏆 GACMA Final Metrics (Winner)")
    model.load_state_dict(torch.load(SAVE_MODEL_PATH))
    model.eval()

    preds_str, acts_str = [], []
    with torch.no_grad():
        for seq, stru, label in tqdm(test_dl, desc="Testing"):
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
    print(f"\n📊 Accuracy Report:")
    print(f"   Level 1 (Main Class):   {l1 * 100:.2f}%")
    print(f"   Level 2 (Sub Class):    {l2 * 100:.2f}%")
    print(f"   Level 3 (Sub-Sub):      {l3 * 100:.2f}%")
    print(f"   Level 4 (Exact Match):  {l4 * 100:.2f}%")
    print("=" * 50)


if __name__ == "__main__":
    run()