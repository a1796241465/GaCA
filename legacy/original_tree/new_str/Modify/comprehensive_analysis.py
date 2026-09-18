






































































































































































































































































































"""Comprehensive GaCA inference: save all predictions, find misclassified proteins, generate PR curves."""
import os, pickle, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, average_precision_score, precision_recall_curve
from sklearn.preprocessing import LabelEncoder, label_binarize
from tqdm import tqdm
import pandas as pd, random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE, HIDDEN_DIM, DROPOUT, SEED = 128, 512, 0.5, 42
SEQ_DIM, STR_DIM = 1280, 512

OUT_DIR = 'D:/EC/new_str/Modify/analysis_outputs'
os.makedirs(OUT_DIR, exist_ok=True)


SEQ_TRAIN  = 'D:/EC/Sequences_Embeddings_use_ESM-2/Modify/1024_train_seq_embeddings_esm2.pkl'
SEQ_T30    = 'D:/EC/Sequences_Embeddings_use_ESM-2/Modify/1024_test_seq_embeddings_esm2.pkl'
SEQ_T50    = 'D:/EC/Sequences_Embeddings_use_ESM-2/Modify/30_50_seq_embeddings_esm2.pkl'
STR_TRAIN  = 'D:/EC/new_str/Modify/train_structure_embeddings_esm_if.pkl'
STR_T30    = 'D:/EC/new_str/Modify/test_structure_embeddings_esm_if.pkl'
STR_T50    = 'D:/EC/new_str/Modify/30-50test_structure_embeddings_esm_if.pkl'
TRAIN_CSV  = 'D:/EC/train_cleaned_with_structure.csv'
T30_CSV    = 'D:/EC/test_30_cleaned_with_structure.csv'
T50_CSV    = 'D:/EC/test_30_50_clean.csv'
MODEL_PATH = 'D:/EC/new_str/Modify/GaCA/best_GaCA_final.pth'

def seed_everything(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def load_pkl(path):
    with open(path, 'rb') as f: return pickle.load(f)

def collate_fn(batch):
    ss = [b[0] for b in batch]; ts = [b[1] for b in batch]
    lbs = torch.cat([b[2] for b in batch])
    return pad_sequence(ss, batch_first=True), pad_sequence(ts, batch_first=True), lbs


class GaCADataset(Dataset):
    def __init__(self, csv_path, seq_path, str_path, is_train=True, le=None):
        df = pd.read_csv(csv_path); df['Entry'] = df['Entry'].astype(str)
        sd, td = load_pkl(seq_path), load_pkl(str_path)
        cm = set(df['Entry']) & set(sd.keys()) & set(td.keys())
        self.ids, self.seq_data, self.str_data, self.labels = [], [], [], []
        for _, r in df.iterrows():
            pid = str(r['Entry'])
            if pid in cm:
                self.ids.append(pid)
                self.seq_data.append(sd[pid]); self.str_data.append(td[pid])
                self.labels.append(str(r['EC number']))
        if is_train:
            self.le = LabelEncoder(); self.el = self.le.fit_transform(self.labels)
        else:
            self.le = le; kn = set(le.classes_)
            mask = np.array([L in kn for L in self.labels])
            self.ids = [self.ids[i] for i in range(len(self.ids)) if mask[i]]
            self.seq_data = [self.seq_data[i] for i in range(len(self.seq_data)) if mask[i]]
            self.str_data = [self.str_data[i] for i in range(len(self.str_data)) if mask[i]]
            self.labels = [self.labels[i] for i in range(len(self.labels)) if mask[i]]
            self.el = le.transform(self.labels)
    def __len__(self): return len(self.el)
    def __getitem__(self, i):
        s = torch.tensor(self.seq_data[i], dtype=torch.float32)
        t = torch.tensor(self.str_data[i], dtype=torch.float32)
        if s.dim()>1: s = s.view(-1, SEQ_DIM)
        if t.dim()>1: t = t.view(-1, STR_DIM)
        if s.dim()==1: s = s.unsqueeze(0)
        if t.dim()==1: t = t.unsqueeze(0)
        return s, t, torch.tensor([self.el[i]], dtype=torch.long)


class AttentionPooling(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(d,128), nn.Tanh(), nn.Linear(128,1))
    def forward(self, x):
        w = F.softmax(self.attention(x), dim=1); return torch.sum(w*x, dim=1)

class GaCA_Final(nn.Module):
    def __init__(self, nc, seq_dim=1280, str_dim=512, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.seq_pool = AttentionPooling(seq_dim); self.str_pool = AttentionPooling(str_dim)
        self.seq_proj = nn.Sequential(nn.Linear(seq_dim,hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.str_proj = nn.Sequential(nn.Linear(str_dim,hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.gating_net = nn.Sequential(nn.Linear(hidden_dim*2,hidden_dim), nn.ReLU(), nn.Linear(hidden_dim,hidden_dim), nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(hidden_dim,512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout), nn.Linear(512,nc))
    def forward(self, s, t):
        sh = self.seq_proj(self.seq_pool(s)); th = self.str_proj(self.str_pool(t))
        g = self.gating_net(torch.cat([sh,th], dim=1))
        return self.classifier(g*th + (1-g)*sh)


def full_inference(model, dataloader, le, ds, name):
    model.eval()
    all_ids, all_true_str, all_pred_str = [], [], []
    all_true_idx, all_pred_idx, all_probs = [], [], []

    with torch.no_grad():
        for batch_idx, (s, t, l) in enumerate(tqdm(dataloader, desc=f"Inference {name}")):
            s, t = s.to(device), t.to(device)
            out = model(s, t)
            probs = F.softmax(out, dim=1).cpu().numpy()
            preds = out.argmax(1).cpu().numpy()


            batch_start = batch_idx * BATCH_SIZE
            batch_ids = ds.ids[batch_start:batch_start + len(l)]
            batch_true_str = [ds.labels[batch_start + i] for i in range(len(l))]
            batch_pred_str = le.inverse_transform(preds)

            all_ids.extend(batch_ids)
            all_true_str.extend(batch_true_str)
            all_pred_str.extend(batch_pred_str)
            all_true_idx.extend([int(x) for x in l.numpy()])
            all_pred_idx.extend([int(x) for x in preds])
            all_probs.extend(probs.astype(np.float32))

    return {
        'name': name,
        'ids': all_ids,
        'true_ec': all_true_str,
        'pred_ec': all_pred_str,
        'true_idx': all_true_idx,
        'pred_idx': all_pred_idx,
        'probs': np.array(all_probs),
        'n_classes': len(le.classes_),
        'class_names': list(le.classes_)
    }


def analyze_misclassifications(result):
    misclassified = []
    ec4_mis, ec7_mis = [], []

    for i in range(len(result['ids'])):
        true_ec = result['true_ec'][i]
        pred_ec = result['pred_ec'][i]
        if true_ec != pred_ec:
            entry = {
                'id': result['ids'][i],
                'true_ec': true_ec,
                'pred_ec': pred_ec,
                'true_top': true_ec.split('.')[0],
                'pred_top': pred_ec.split('.')[0],
                'confidence': float(result['probs'][i].max()),
                'true_confidence': float(result['probs'][i][result['true_idx'][i]]),
            }
            misclassified.append(entry)
            if entry['true_top'] == '4':
                ec4_mis.append(entry)
            if entry['true_top'] == '7':
                ec7_mis.append(entry)

    return misclassified, ec4_mis, ec7_mis



def plot_pr_curves(results, le, out_dir):
    """Plot macro-averaged PR curves, 1:1 match original figure style."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, res in zip(axes, results):
        n_classes = res['n_classes']
        y_true = label_binarize(res['true_idx'], classes=range(n_classes))
        y_score = res['probs']


        all_precision = []
        all_recall = []
        auprs = []

        for i in range(n_classes):
            if np.sum(y_true[:, i]) > 0:
                p, r, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
                all_precision.append(p)
                all_recall.append(r)
                auprs.append(average_precision_score(y_true[:, i], y_score[:, i]))

        macro_aupr = np.mean(auprs) if auprs else 0



        all_recall_points = np.concatenate(all_recall)
        unique_recall = np.sort(np.unique(all_recall_points))[::-1]


        macro_precision = []
        for r_val in unique_recall:
            p_vals = []
            for i in range(len(all_precision)):

                mask = all_recall[i] >= r_val
                if np.any(mask):
                    p_vals.append(np.max(all_precision[i][mask]))
            if p_vals:
                macro_precision.append(np.mean(p_vals))
            else:
                macro_precision.append(0.0)


        ax.plot(unique_recall, macro_precision, linewidth=1.5, color='#4A90E2')
        ax.fill_between(unique_recall, macro_precision, alpha=0.1, color='#4A90E2')


        ax.set_title(f"{res['name']}\nMacro-AUPR = {macro_aupr:.4f}", fontsize=13, pad=10)
        ax.set_xlabel('Recall', fontsize=12)
        ax.set_ylabel('Precision', fontsize=12)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])


        ax.grid(True, alpha=0.3, linestyle='-', color='lightgray')

    plt.tight_layout()
    path = os.path.join(out_dir, 'precision_recall_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"PR curves saved to {path}")


if __name__ == "__main__":
    seed_everything(SEED)
    print(f"Device: {device}")


    print("Loading data...")
    tr = GaCADataset(TRAIN_CSV, SEQ_TRAIN, STR_TRAIN, is_train=True)
    le = tr.le; nc = len(le.classes_)
    t30 = GaCADataset(T30_CSV, SEQ_T30, STR_T30, is_train=False, le=le)
    t50 = GaCADataset(T50_CSV, SEQ_T50, STR_T50, is_train=False, le=le)
    print(f"Train: {len(tr)}, Test<30%: {len(t30)}, Test30-50%: {len(t50)}, Classes: {nc}")

    t30_dl = DataLoader(t30, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)
    t50_dl = DataLoader(t50, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_fn)


    model = GaCA_Final(nc, hidden_dim=HIDDEN_DIM, dropout=DROPOUT).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded.")


    r30 = full_inference(model, t30_dl, le, t30, "<30%")
    r50 = full_inference(model, t50_dl, le, t50, "30-50%")


    for r in [r30, r50]:
        save_path = os.path.join(OUT_DIR, f"predictions_{r['name'].replace('%','').replace('<','lt').replace(' ','_')}.json")

        r_light = {k: v for k, v in r.items() if k != 'probs'}
        with open(save_path, 'w') as f:
            json.dump(r_light, f, indent=2, ensure_ascii=False)
        np_path = save_path.replace('.json', '_probs.npy')
        np.save(np_path, r['probs'])
        print(f"Predictions saved: {save_path} (+ {np_path})")


    for r in [r30, r50]:
        mis, ec4, ec7 = analyze_misclassifications(r)
        print(f"\n{'='*60}")
        print(f"{r['name']}: {len(mis)} misclassified / {len(r['ids'])} total ({len(mis)/len(r['ids'])*100:.1f}%)")
        print(f"  EC4 (true) misclassified: {len(ec4)}")
        for m in ec4:
            print(f"    {m['id']}: true={m['true_ec']} -> pred={m['pred_ec']} (conf={m['confidence']:.3f})")
        print(f"  EC7 (true) misclassified: {len(ec7)}")
        for m in ec7:
            print(f"    {m['id']}: true={m['true_ec']} -> pred={m['pred_ec']} (conf={m['confidence']:.3f})")


        mis_path = os.path.join(OUT_DIR, f"misclassifications_{r['name'].replace('%','').replace('<','lt').replace(' ','_')}.json")
        with open(mis_path, 'w') as f:
            json.dump({'all': mis, 'ec4': ec4, 'ec7': ec7}, f, indent=2, ensure_ascii=False)


    plot_pr_curves([r30, r50], le, OUT_DIR)


    print(f"\n{'='*60}")
    print("Per-class-frequency analysis:")
    for r in [r30, r50]:
        print(f"\n{r['name']}:")

        class_counts = {}
        for t in r['true_ec']:
            class_counts[t] = class_counts.get(t, 0) + 1


        bins = {'1':0, '2-5':0, '6-20':0, '20+':0}
        bin_correct = {'1':0, '2-5':0, '6-20':0, '20+':0}
        for i in range(len(r['ids'])):
            count = class_counts[r['true_ec'][i]]
            if count == 1: b = '1'
            elif count <= 5: b = '2-5'
            elif count <= 20: b = '6-20'
            else: b = '20+'
            bins[b] += 1
            if r['true_ec'][i] == r['pred_ec'][i]:
                bin_correct[b] += 1

        for b in ['1', '2-5', '6-20', '20+']:
            if bins[b] > 0:
                acc = bin_correct[b] / bins[b] * 100
                print(f"  Frequency {b}: {bins[b]} samples, L4 acc = {acc:.1f}%")

    print(f"\nAll outputs saved to: {OUT_DIR}")
    print("Done.")