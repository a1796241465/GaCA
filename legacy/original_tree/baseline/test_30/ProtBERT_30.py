import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import BertModel, BertTokenizer
from tqdm import tqdm
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
EPOCHS = 100
PATIENCE = 20


PROT_BERT_LOCAL_PATH = '/home/lihaotian/new_ec/baseline/prot_bert_local/'


TRAIN_CSV = '/home/lihaotian/new_ec/train_cleaned_with_structure.csv'
TEST_CSV = '/home/lihaotian/new_ec/test_30_cleaned_with_structure.csv'
SEQ_COL_NAME = 'Sequence'



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)



def calc_hierarchical_metrics(y_true_str, y_pred_str):
    c1, c2, c3, c4 = 0, 0, 0, 0
    total = len(y_true_str)

    for t, p in zip(y_true_str, y_pred_str):

        tp = str(t).split('.')
        pp = str(p).split('.')


        if len(tp) >= 1 and len(pp) >= 1 and tp[0] == pp[0]:
            c1 += 1

        if len(tp) >= 2 and len(pp) >= 2 and tp[:2] == pp[:2]:
            c2 += 1

        if len(tp) >= 3 and len(pp) >= 3 and tp[:3] == pp[:3]:
            c3 += 1

        if t == p:
            c4 += 1

    return c1 / total, c2 / total, c3 / total, c4 / total



print(f">>> 正在从本地加载 ProtBERT 模型: {PROT_BERT_LOCAL_PATH} ...")

if not os.path.exists(PROT_BERT_LOCAL_PATH):
    raise FileNotFoundError(f"❌ 找不到模型文件夹: {PROT_BERT_LOCAL_PATH}")

try:
    tokenizer = BertTokenizer.from_pretrained(PROT_BERT_LOCAL_PATH, do_lower_case=False)
    bert_model = BertModel.from_pretrained(PROT_BERT_LOCAL_PATH).to(device)
    bert_model.eval()
    print("✅ 模型加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    exit()


def extract_protbert_features(csv_path, is_train=True, label_encoder=None):
    df = pd.read_csv(csv_path)

    if SEQ_COL_NAME not in df.columns:
        raise ValueError(f"CSV中找不到列名 '{SEQ_COL_NAME}'")

    sequences = df[SEQ_COL_NAME].astype(str).tolist()
    labels = df['EC number'].astype(str).tolist()

    processed_seqs = [" ".join(list(seq)) for seq in sequences]

    features = []
    print(f"🔄 正在提取特征: {os.path.basename(csv_path)} ...")

    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(processed_seqs), batch_size)):
            batch_seqs = processed_seqs[i: i + batch_size]

            inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            outputs = bert_model(input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.mean(dim=1)
            features.append(embeddings.cpu().numpy())

    X = np.concatenate(features, axis=0)
    y = np.array(labels)

    norm = np.linalg.norm(X, axis=1, keepdims=True)
    norm[norm == 0] = 1e-6
    X = X / norm

    if is_train:
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        return X, y_enc, le
    else:
        known_classes = set(label_encoder.classes_)
        mask = np.array([label in known_classes for label in y])
        X = X[mask]
        y_clean_raw = y[mask]
        y_enc = label_encoder.transform(y_clean_raw)
        return X, y_enc, y_clean_raw



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



seed_everything(SEED)


X_train, y_train, le = extract_protbert_features(TRAIN_CSV, is_train=True)
X_test, y_test, y_test_raw = extract_protbert_features(TEST_CSV, is_train=False, label_encoder=le)

num_classes = len(le.classes_)
print(f"✅ 特征提取完毕! 训练集维度: {X_train.shape}, 类别数: {num_classes}")


X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED, shuffle=True)

X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
y_tr_t = torch.tensor(y_tr, dtype=torch.long).to(device)
X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
X_te_t = torch.tensor(X_test, dtype=torch.float32).to(device)

dl_tr = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=BATCH_SIZE, shuffle=True)
dl_val = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=BATCH_SIZE, shuffle=False)


model = SimpleMLP(input_dim=1024, num_classes=num_classes).to(device)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

best_val_acc = 0.0
patience_cnt = 0
best_state = None

print("🚀 开始训练 ProtBERT + MLP Baseline...")
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

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Val Acc = {val_acc:.4f}")


model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    logits = model(X_te_t)
    preds_indices = torch.argmax(logits, dim=1).cpu().numpy()

preds_str = le.inverse_transform(preds_indices)


l1, l2, l3, l4 = calc_hierarchical_metrics(y_test_raw, preds_str)

print("\n" + "=" * 40)
print(f"🏆 ProtBERT Baseline 最终结果 (Level 1-4):")
print(f"   Level 1 Acc: {l1 * 100:.2f}%")
print(f"   Level 2 Acc: {l2 * 100:.2f}%")
print(f"   Level 3 Acc: {l3 * 100:.2f}%")
print(f"   Level 4 Acc: {l4 * 100:.2f}%")
print("=" * 40)