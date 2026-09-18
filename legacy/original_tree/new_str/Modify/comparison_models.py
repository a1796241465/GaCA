"""
ProteinF3S & SES-Adapter 对比实验
针对审稿人 R2 要求: "lacks direct comparison against ProteinF3S or SES-Adapter"

ProteinF3S-style: Multi-scale bidirectional cross-attention fusion (sequence <-> structure)
SES-Adapter-style: Structure-conditioned bottleneck adapter layers
GaCA (Ours): Structure-Guided Gating with Attention Pooling

使用相同的 ESM-2/ESM-IF1 冻结特征, 公平对比融合架构。
"""
import os, pickle, json, random, copy
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, average_precision_score
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")




BATCH_SIZE = 128
LR = 1e-4
WD = 5e-3
HIDDEN_DIM = 512
DROPOUT = 0.5
EPOCHS = 120
PATIENCE = 30
VAL_RATIO = 0.1
SEED = 42
LABEL_SMOOTH = 0.1
SEQ_DIM = 1280
STR_DIM = 512

SEQ_TRAIN = "D:/EC/Sequences_Embeddings_use_ESM-2/Modify/1024_train_seq_embeddings_esm2.pkl"
SEQ_TEST  = "D:/EC/Sequences_Embeddings_use_ESM-2/Modify/1024_test_seq_embeddings_esm2.pkl"
SEQ_TEST_50 = "D:/EC/Sequences_Embeddings_use_ESM-2/Modify/30_50_seq_embeddings_esm2.pkl"

STR_TRAIN = "D:/EC/new_str/Modify/train_structure_embeddings_esm_if.pkl"
STR_TEST  = "D:/EC/new_str/Modify/test_structure_embeddings_esm_if.pkl"
STR_TEST_50 = "D:/EC/new_str/Modify/30-50test_structure_embeddings_esm_if.pkl"

TRAIN_CSV = "D:/EC/train_cleaned_with_structure.csv"
TEST_CSV  = "D:/EC/test_30_cleaned_with_structure.csv"
TEST_50_CSV = "D:/EC/test_30_50_clean.csv"

SAVE_DIR = "D:/EC/new_str/Modify/comparison_results"
os.makedirs(SAVE_DIR, exist_ok=True)





def seed_everything(s):
    random.seed(s); os.environ['PYTHONHASHSEED'] = str(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def load_pkl(p):
    with open(p, 'rb') as f: return pickle.load(f)

def pad_collate(batch):
    seqs = [b[0] for b in batch]; strs = [b[1] for b in batch]
    return pad_sequence(seqs, batch_first=True), pad_sequence(strs, batch_first=True), torch.cat([b[2] for b in batch])





class EnzymeDataset(Dataset):
    def __init__(self, csv_path, seq_path, str_path, is_train=True, le=None):
        df = pd.read_csv(csv_path); df['Entry'] = df['Entry'].astype(str)
        seq_d, str_d = load_pkl(seq_path), load_pkl(str_path)
        common = set(df['Entry']) & set(seq_d.keys()) & set(str_d.keys())
        self.ids, self.seq, self.str, self.labels = [], [], [], []
        for _, row in df.iterrows():
            pid = str(row['Entry'])
            if pid in common:
                self.ids.append(pid); self.seq.append(seq_d[pid])
                self.str.append(str_d[pid]); self.labels.append(str(row['EC number']))
        if is_train:
            self.le = LabelEncoder(); self.enc = self.le.fit_transform(self.labels)
        else:
            self.le = le; known = set(le.classes_)
            keep = [i for i, L in enumerate(self.labels) if L in known]
            self.ids = [self.ids[i] for i in keep]; self.seq = [self.seq[i] for i in keep]
            self.str = [self.str[i] for i in keep]; self.labels = [self.labels[i] for i in keep]
            self.enc = le.transform(self.labels)

    def __len__(self): return len(self.enc)
    def __getitem__(self, idx):
        s = torch.tensor(self.seq[idx], dtype=torch.float32)
        t = torch.tensor(self.str[idx], dtype=torch.float32)
        if s.dim() > 1: s = s.view(-1, SEQ_DIM)
        if t.dim() > 1: t = t.view(-1, STR_DIM)
        if s.dim() == 1: s = s.unsqueeze(0)
        if t.dim() == 1: t = t.unsqueeze(0)
        return s, t, torch.LongTensor([self.enc[idx]])





class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(dim, 128), nn.Tanh(), nn.Linear(128, 1))
    def forward(self, x):
        w = F.softmax(self.attn(x), dim=1)
        return torch.sum(w * x, dim=1)

class GaCA(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.seq_pool = AttentionPooling(SEQ_DIM)
        self.str_pool = AttentionPooling(STR_DIM)
        self.seq_proj = nn.Sequential(nn.Linear(SEQ_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Dropout(DROPOUT))
        self.str_proj = nn.Sequential(nn.Linear(STR_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Dropout(DROPOUT))
        self.gate = nn.Sequential(nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM), nn.ReLU(), nn.Linear(HIDDEN_DIM, HIDDEN_DIM), nn.Sigmoid())
        self.cls = nn.Sequential(nn.Linear(HIDDEN_DIM, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(512, num_classes))

    def forward(self, s, t):
        sp = self.seq_pool(s); tp = self.str_pool(t)
        sh = self.seq_proj(sp); th = self.str_proj(tp)
        g = self.gate(torch.cat([sh, th], dim=1))
        return self.cls(g * th + (1 - g) * sh)





class CrossAttentionFusion(nn.Module):
    """Seq <-> Str bidirectional cross-attention, ProteinF3S-style"""
    def __init__(self, dim, heads=4):
        super().__init__()
        self.seq_to_str = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=DROPOUT)
        self.str_to_seq = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=DROPOUT)
        self.seq_gate = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())
        self.str_gate = nn.Sequential(nn.Linear(dim*2, dim), nn.Sigmoid())
        self.norm1 = nn.LayerNorm(dim); self.norm2 = nn.LayerNorm(dim)

    def forward(self, seq, str_):

        seq_enh, _ = self.seq_to_str(query=seq, key=str_, value=str_)
        seq_g = self.seq_gate(torch.cat([seq, seq_enh], dim=-1))
        seq_out = self.norm1(seq + seq_g * seq_enh)

        str_enh, _ = self.str_to_seq(query=str_, key=seq, value=seq)
        str_g = self.str_gate(torch.cat([str_, str_enh], dim=-1))
        str_out = self.norm2(str_ + str_g * str_enh)
        return seq_out, str_out


class ProteinF3S_Style(nn.Module):
    """ProteinF3S-inspired: project -> cross-attention fusion -> global pool -> classify"""
    def __init__(self, num_classes):
        super().__init__()
        self.seq_proj = nn.Sequential(nn.Linear(SEQ_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Dropout(DROPOUT))
        self.str_proj = nn.Sequential(nn.Linear(STR_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Dropout(DROPOUT))
        self.cross_attn = CrossAttentionFusion(HIDDEN_DIM, heads=4)
        self.seq_pool = AttentionPooling(HIDDEN_DIM)
        self.str_pool = AttentionPooling(HIDDEN_DIM)

        self.fusion = nn.Sequential(
            nn.Linear(HIDDEN_DIM*2, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.GELU(), nn.Dropout(DROPOUT))
        self.cls = nn.Sequential(nn.Linear(HIDDEN_DIM, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(512, num_classes))

    def forward(self, s_raw, t_raw):
        s = self.seq_proj(s_raw); t = self.str_proj(t_raw)
        s_enh, t_enh = self.cross_attn(s, t)
        sp = self.seq_pool(s_enh); tp = self.str_pool(t_enh)
        fused = self.fusion(torch.cat([sp, tp], dim=1))
        return self.cls(fused)





class StructureAdapter(nn.Module):
    """SES-Adapter style: structure features condition a bottleneck adapter on sequence"""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.down = nn.Linear(dim, dim // reduction)
        self.up = nn.Linear(dim // reduction, dim)
        self.str_condition = nn.Sequential(nn.Linear(dim, dim//reduction), nn.GELU())
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, str_condition):

        cond = self.str_condition(str_condition)
        h = self.down(x)
        h = h * cond
        h = F.gelu(h)
        h = self.up(h)
        return self.norm(x + h)


class SESAdapter_Style(nn.Module):
    """SES-Adapter-inspired: structure-conditioned adapters on sequence encoder output"""
    def __init__(self, num_classes, num_adapters=3):
        super().__init__()
        self.seq_proj = nn.Sequential(nn.Linear(SEQ_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Dropout(DROPOUT))
        self.str_proj = nn.Sequential(nn.Linear(STR_DIM, HIDDEN_DIM), nn.LayerNorm(HIDDEN_DIM), nn.Dropout(DROPOUT))
        self.str_pool = AttentionPooling(STR_DIM)

        self.adapters = nn.ModuleList([StructureAdapter(HIDDEN_DIM) for _ in range(num_adapters)])
        self.seq_pool = AttentionPooling(HIDDEN_DIM)
        self.cls = nn.Sequential(nn.Linear(HIDDEN_DIM, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(512, num_classes))

    def forward(self, s_raw, t_raw):
        s = self.seq_proj(s_raw)
        t = self.str_proj(t_raw)
        t_global = self.str_pool(t_raw)
        t_cond = self.str_proj[0](t_global)

        for adapter in self.adapters:
            s = adapter(s, t_cond)
        sp = self.seq_pool(s)
        return self.cls(sp)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)





def compute_metrics(all_labels, all_preds, all_probs, le):
    """Full metrics suite"""
    acc_l1 = acc_l2 = acc_l3 = acc_l4 = 0
    total = len(all_labels)
    labels_str = le.inverse_transform(all_labels)
    preds_str = le.inverse_transform(all_preds)
    for t, p in zip(labels_str, preds_str):
        tp, pp = t.split('.'), p.split('.')
        if len(pp) >= 1 and tp[0] == pp[0]: acc_l1 += 1
        if len(pp) >= 2 and tp[:2] == pp[:2]: acc_l2 += 1
        if len(pp) >= 3 and tp[:3] == pp[:3]: acc_l3 += 1
        if t == p: acc_l4 += 1

    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    mcc = matthews_corrcoef(all_labels, all_preds)

    n_classes = len(le.classes_)
    labels_bin = label_binarize(all_labels, classes=range(n_classes))
    auprs = []
    for i in range(n_classes):
        if np.sum(labels_bin[:, i]) > 0:
            auprs.append(average_precision_score(labels_bin[:, i], np.array(all_probs)[:, i]))
    macro_aupr = np.mean(auprs) if auprs else 0.0

    return {
        "L1": acc_l1/total, "L2": acc_l2/total, "L3": acc_l3/total, "L4": acc_l4/total,
        "macro_f1": macro_f1, "micro_f1": micro_f1, "mcc": mcc, "macro_aupr": macro_aupr,
    }


def train_model(model_class, name, train_ds, val_idx, test_30_ds, test_50_ds, le, num_classes):
    """Train one model and return results"""
    seed_everything(SEED)
    model = model_class(num_classes).to(DEVICE)
    n_params = count_params(model)

    train_dl = DataLoader(Subset(train_ds, val_idx["train"]), batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    val_dl = DataLoader(Subset(train_ds, val_idx["val"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    best_val_acc = 0.0; patience_cnt = 0; best_state = None
    history = []

    log(f"  Training {name} ({n_params:,} params) ...")
    for epoch in range(EPOCHS):
        model.train()
        for s, t, lbl in train_dl:
            s, t, lbl = s.to(DEVICE), t.to(DEVICE), lbl.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(s, t), lbl)
            loss.backward(); optimizer.step()

        model.eval()
        preds, acts = [], []
        with torch.no_grad():
            for s, t, lbl in val_dl:
                s, t, lbl = s.to(DEVICE), t.to(DEVICE), lbl.to(DEVICE)
                out = model(s, t)
                preds.extend(out.argmax(1).cpu().numpy()); acts.extend(lbl.cpu().numpy())
        val_acc = accuracy_score(acts, preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc; patience_cnt = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                history.append(f"epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(SAVE_DIR, f"{name.replace(' ','_')}_best.pth"))


    results = {}
    for test_ds, ds_name in [(test_30_ds, "<30%"), (test_50_ds, "30-50%")]:
        if test_ds is None: continue
        dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for s, t, lbl in dl:
                s, t = s.to(DEVICE), t.to(DEVICE)
                out = model(s, t)
                probs = F.softmax(out, dim=1).cpu().numpy()
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_labels.extend(lbl.numpy())
                all_probs.extend(probs)
        results[ds_name] = compute_metrics(all_labels, all_preds, all_probs, le)

    return {"name": name, "params": n_params, "results": results, "history": history}


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")





def main():
    log("="*60)
    log("ProteinF3S & SES-Adapter Comparison Experiment")
    log("="*60)


    log("Loading data ...")
    train_ds = EnzymeDataset(TRAIN_CSV, SEQ_TRAIN, STR_TRAIN, is_train=True)
    le = train_ds.le; num_classes = len(le.classes_)
    log(f"  Train: {len(train_ds)} samples, {num_classes} classes")

    test_30_ds = EnzymeDataset(TEST_CSV, SEQ_TEST, STR_TEST, is_train=False, le=le)
    log(f"  Test <30%: {len(test_30_ds)} samples")

    test_50_ds = None
    if os.path.exists(TEST_50_CSV):
        test_50_ds = EnzymeDataset(TEST_50_CSV, SEQ_TEST_50, STR_TEST_50, is_train=False, le=le)
        log(f"  Test 30-50%: {len(test_50_ds)} samples")


    train_idx, val_idx = train_test_split(range(len(train_ds)), test_size=VAL_RATIO, random_state=SEED, shuffle=True)
    split = {"train": train_idx, "val": val_idx}
    log(f"  Train split: {len(train_idx)}, Val split: {len(val_idx)}")


    all_results = []


    log("\n>>> GaCA (Ours)")
    r = train_model(GaCA, "GaCA (Ours)", train_ds, split, test_30_ds, test_50_ds, le, num_classes)
    all_results.append(r)


    log("\n>>> ProteinF3S-style (Cross-Attention Fusion)")
    r = train_model(ProteinF3S_Style, "ProteinF3S-style", train_ds, split, test_30_ds, test_50_ds, le, num_classes)
    all_results.append(r)


    log("\n>>> SES-Adapter-style (Structure Adapters)")
    r = train_model(SESAdapter_Style, "SES-Adapter-style", train_ds, split, test_30_ds, test_50_ds, le, num_classes)
    all_results.append(r)


    log("\n" + "=" * 80)
    log("FINAL COMPARISON REPORT")
    log("=" * 80)

    for ds_name in ["<30%", "30-50%"]:
        print(f"\n{'='*60}")
        print(f"  Test Set: {ds_name} Sequence Identity")
        print(f"  {'Model':<22} {'Params':<10} {'L1':<8} {'L2':<8} {'L3':<8} {'L4':<8} {'Macro-F1':<10} {'MCC':<8} {'AUPR':<8}")
        print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")
        for r in all_results:
            m = r["results"].get(ds_name)
            if m is None: continue
            print(f"  {r['name']:<22} {r['params']:<10,} "
                  f"{m['L1']*100:>6.2f}% {m['L2']*100:>6.2f}% {m['L3']*100:>6.2f}% {m['L4']*100:>6.2f}% "
                  f"{m['macro_f1']:>8.4f}  {m['mcc']:>6.4f}  {m['macro_aupr']:>6.4f}")


    report = {
        "experiment": "ProteinF3S & SES-Adapter Comparison vs GaCA",
        "setup": {
            "features": "ESM-2 650M (residue-level) + ESM-IF1 (residue-level)",
            "training": f"AdamW(lr={LR}, wd={WD}), Batch={BATCH_SIZE}, Epochs={EPOCHS}, Patience={PATIENCE}",
            "train_val_split": f"{len(train_idx)}/{len(val_idx)}",
            "random_seed": SEED,
        },
        "models": {}
    }
    for r in all_results:
        report["models"][r["name"]] = {
            "params": r["params"],
            "results": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in r["results"].items()},
        }

    with open(os.path.join(SAVE_DIR, "comparison_report.json"), 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log(f"\nReport saved to {SAVE_DIR}/comparison_report.json")
    log("Done!")


if __name__ == "__main__":
    main()
