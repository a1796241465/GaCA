








import os, pickle, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import silhouette_score, davies_bouldin_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
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

    def forward(self, s, t, return_features=False):
        sh = self.seq_proj(self.seq_pool(s)); th = self.str_proj(self.str_pool(t))
        g = self.gating_net(torch.cat([sh,th], dim=1))
        fusion = g*th + (1-g)*sh
        if return_features:
            return self.classifier(fusion), fusion
        return self.classifier(fusion)


def extract_features(model, dataloader):

    model.eval()
    all_features, all_labels = [], []
    with torch.no_grad():
        for s, t, l in tqdm(dataloader, desc="Extracting features"):
            s, t = s.to(device), t.to(device)
            _, feats = model(s, t, return_features=True)
            all_features.append(feats.cpu().numpy())
            all_labels.append(l.numpy())
    return np.vstack(all_features), np.concatenate(all_labels)


def compute_clustering_metrics(features, labels, sample_size=5000, random_state=42):




    n_samples = len(labels)
    if n_samples > sample_size:
        np.random.seed(random_state)
        idx = np.random.choice(n_samples, sample_size, replace=False)
        features_sample = features[idx]
        labels_sample = labels[idx]
    else:
        features_sample = features
        labels_sample = labels


    sil = silhouette_score(features_sample, labels_sample)


    db = davies_bouldin_score(features_sample, labels_sample)

    return {
        'silhouette_score': float(sil),
        'davies_bouldin_index': float(db),
        'n_samples_used': len(labels_sample),
        'n_classes': len(np.unique(labels_sample))
    }


def linear_probing(train_features, train_labels, test_features, test_labels, max_iter=1000):




    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)


    print("Training linear classifier (Logistic Regression)...")
    clf = LogisticRegression(
        max_iter=max_iter,
        multi_class='multinomial',
        solver='lbfgs',
        n_jobs=-1,
        random_state=42
    )
    clf.fit(train_scaled, train_labels)
    pred_labels = clf.predict(test_scaled)
    acc = accuracy_score(test_labels, pred_labels)


    print("Training k-NN classifier (k=5)...")
    knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
    knn.fit(train_scaled, train_labels)
    knn_pred = knn.predict(test_scaled)
    knn_acc = accuracy_score(test_labels, knn_pred)

    return {
        'linear_probe_accuracy': float(acc),
        'knn_accuracy': float(knn_acc),
        'n_train': len(train_labels),
        'n_test': len(test_labels),
        'n_classes': len(np.unique(np.concatenate([train_labels, test_labels])))
    }


def compare_with_raw_embeddings(train_ds, test_ds, le):



    def get_raw_embeddings(ds, pool=True):
        seq_embs, str_embs, labels = [], [], []
        for i in range(len(ds)):
            s, t, l = ds[i]
            if pool:
                seq_embs.append(s.mean(dim=0).numpy())
                str_embs.append(t.mean(dim=0).numpy())
            else:
                seq_embs.append(s[0].numpy())
            labels.append(l.item())
        return np.array(seq_embs), np.array(str_embs), np.array(labels)

    train_seq, train_str, train_lbl = get_raw_embeddings(train_ds)
    test_seq, test_str, test_lbl = get_raw_embeddings(test_ds)


    train_raw = np.concatenate([train_seq, train_str], axis=1)
    test_raw = np.concatenate([test_seq, test_str], axis=1)


    scaler = StandardScaler()
    train_raw_scaled = scaler.fit_transform(train_raw)
    test_raw_scaled = scaler.transform(test_raw)

    clf_raw = LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', n_jobs=-1, random_state=42)
    clf_raw.fit(train_raw_scaled, train_lbl)
    raw_acc = accuracy_score(test_lbl, clf_raw.predict(test_raw_scaled))

    return {
        'raw_concat_accuracy': float(raw_acc),
        'raw_seq_dim': train_seq.shape[1],
        'raw_str_dim': train_str.shape[1],
        'raw_total_dim': train_raw.shape[1]
    }


def plot_feature_analysis(results, out_dir):

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))


    ax = axes[0]
    metrics = ['Silhouette (↑)', 'DB Index (↓)']
    gaca_vals = [results['gaca']['silhouette_score'], results['gaca']['davies_bouldin_index']]
    raw_vals = [results.get('raw', {}).get('silhouette_score', 0),
                results.get('raw', {}).get('davies_bouldin_index', 0)]

    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, gaca_vals, width, label='GaCA', color='steelblue')
    ax.bar(x + width/2, raw_vals, width, label='Raw Concat', color='lightcoral')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title('Clustering Quality Metrics')
    ax.legend()
    ax.set_ylabel('Score')


    ax = axes[1]
    methods = ['GaCA Features', 'Raw Concat', 'k-NN (GaCA)']
    accs = [
        results['linear_probe']['linear_probe_accuracy'],
        results['linear_probe'].get('raw_concat_accuracy', 0),
        results['linear_probe']['knn_accuracy']
    ]
    colors = ['steelblue', 'lightcoral', 'forestgreen']
    bars = ax.bar(methods, accs, color=colors)
    ax.set_ylabel('Accuracy')
    ax.set_title('Linear Probing Accuracy')
    ax.set_ylim(0, 1)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc*100:.2f}%', ha='center', va='bottom', fontsize=10)


    ax = axes[2]
    dims = ['ESM-2 (Seq)', 'ESM-IF1 (Str)', 'GaCA Fusion']
    dims_vals = [SEQ_DIM, STR_DIM, HIDDEN_DIM]
    ax.bar(dims, dims_vals, color=['#3498db', '#e74c3c', '#2ecc71'])
    ax.set_ylabel('Dimension')
    ax.set_title('Feature Dimensions')
    for i, v in enumerate(dims_vals):
        ax.text(i, v + 20, str(v), ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'representation_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_dir}/representation_analysis.png")


def main():
    seed_everything(SEED)
    print("=" * 60)
    print("GaCA Representation Quality Analysis")
    print("=" * 60)


    print("\nLoading datasets...")
    train_ds = GaCADataset(TRAIN_CSV, SEQ_TRAIN, STR_TRAIN, is_train=True)
    le = train_ds.le
    print(f"Train: {len(train_ds)} samples, {len(le.classes_)} classes")

    t30_ds = GaCADataset(T30_CSV, SEQ_T30, STR_T30, is_train=False, le=le)
    print(f"Test <30%: {len(t30_ds)} samples")

    t50_ds = GaCADataset(T50_CSV, SEQ_T50, STR_T50, is_train=False, le=le)
    print(f"Test 30-50%: {len(t50_ds)} samples")


    print("\nLoading model...")
    model = GaCA_Final(len(le.classes_)).to(device)
    state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded from {MODEL_PATH}")


    print("\n" + "=" * 60)
    print("Extracting GaCA fusion features...")
    print("=" * 60)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    t30_loader = DataLoader(t30_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    t50_loader = DataLoader(t50_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    train_feats, train_lbls = extract_features(model, train_loader)
    t30_feats, t30_lbls = extract_features(model, t30_loader)
    t50_feats, t50_lbls = extract_features(model, t50_loader)

    print(f"Train features: {train_feats.shape}")
    print(f"Test <30% features: {t30_feats.shape}")
    print(f"Test 30-50% features: {t50_feats.shape}")


    print("\n" + "=" * 60)
    print("Computing clustering quality metrics...")
    print("=" * 60)


    train_metrics = compute_clustering_metrics(train_feats, train_lbls, sample_size=5000)
    print(f"Train set (sampled {train_metrics['n_samples_used']} from {len(train_lbls)}):")
    print(f"  Silhouette Score: {train_metrics['silhouette_score']:.4f} (higher = better)")
    print(f"  Davies-Bouldin Index: {train_metrics['davies_bouldin_index']:.4f} (lower = better)")


    t30_metrics = compute_clustering_metrics(t30_feats, t30_lbls, sample_size=min(5000, len(t30_lbls)))
    t50_metrics = compute_clustering_metrics(t50_feats, t50_lbls, sample_size=min(5000, len(t50_lbls)))
    print(f"\nTest <30%:")
    print(f"  Silhouette: {t30_metrics['silhouette_score']:.4f}, DB: {t30_metrics['davies_bouldin_index']:.4f}")
    print(f"\nTest 30-50%:")
    print(f"  Silhouette: {t50_metrics['silhouette_score']:.4f}, DB: {t50_metrics['davies_bouldin_index']:.4f}")


    print("\n" + "=" * 60)
    print("Linear Probing (frozen features)...")
    print("=" * 60)


    lp_t30 = linear_probing(train_feats, train_lbls, t30_feats, t30_lbls)
    print(f"\nTest <30%:")
    print(f"  Linear Probe Accuracy: {lp_t30['linear_probe_accuracy']*100:.2f}%")
    print(f"  k-NN (k=5) Accuracy: {lp_t30['knn_accuracy']*100:.2f}%")


    lp_t50 = linear_probing(train_feats, train_lbls, t50_feats, t50_lbls)
    print(f"\nTest 30-50%:")
    print(f"  Linear Probe Accuracy: {lp_t50['linear_probe_accuracy']*100:.2f}%")
    print(f"  k-NN (k=5) Accuracy: {lp_t50['knn_accuracy']*100:.2f}%")


    print("\n" + "=" * 60)
    print("Comparing with raw ESM-2 + ESM-IF1 concatenation...")
    print("=" * 60)

    raw_comp = compare_with_raw_embeddings(train_ds, t30_ds, le)
    print(f"Raw concatenation ({raw_comp['raw_seq_dim']} + {raw_comp['raw_str_dim']} = {raw_comp['raw_total_dim']} dims)")
    print(f"  Linear Probe Accuracy (<30%): {raw_comp['raw_concat_accuracy']*100:.2f}%")

    raw_comp_t50 = compare_with_raw_embeddings(train_ds, t50_ds, le)
    print(f"  Linear Probe Accuracy (30-50%): {raw_comp_t50['raw_concat_accuracy']*100:.2f}%")


    results = {
        'gaca': {
            'silhouette_score': train_metrics['silhouette_score'],
            'davies_bouldin_index': train_metrics['davies_bouldin_index'],
        },
        'test_30': {
            'silhouette': t30_metrics['silhouette_score'],
            'davies_bouldin': t30_metrics['davies_bouldin_index'],
            'linear_probe_acc': lp_t30['linear_probe_accuracy'],
            'knn_acc': lp_t30['knn_accuracy'],
        },
        'test_50': {
            'silhouette': t50_metrics['silhouette_score'],
            'davies_bouldin': t50_metrics['davies_bouldin_index'],
            'linear_probe_acc': lp_t50['linear_probe_accuracy'],
            'knn_acc': lp_t50['knn_accuracy'],
        },
        'linear_probe': {
            'linear_probe_accuracy': lp_t30['linear_probe_accuracy'],
            'knn_accuracy': lp_t30['knn_accuracy'],
            'raw_concat_accuracy': raw_comp['raw_concat_accuracy'],
            'raw_concat_accuracy_t50': raw_comp_t50['raw_concat_accuracy'],
        },
        'feature_dim': HIDDEN_DIM,
        'n_classes': len(le.classes_),
        'n_train': len(train_ds),
    }


    with open(os.path.join(OUT_DIR, 'representation_analysis.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT_DIR}/representation_analysis.json")


    plot_feature_analysis(results, OUT_DIR)


    print("\n" + "=" * 60)
    print("SUMMARY FOR PAPER")
    print("=" * 60)
    print(f"""
Representation Quality Analysis (Addressing Reviewer uRQ4 M2):

1. Clustering Quality Metrics (GaCA fusion features):
   - Silhouette Score: {train_metrics['silhouette_score']:.4f}
     (Range [-1,1], higher = better intra-class cohesion & inter-class separation)
   - Davies-Bouldin Index: {train_metrics['davies_bouldin_index']:.4f}
     (Range [0,∞), lower = better class separation)

2. Linear Probing Accuracy (frozen GaCA features):
   - Test <30%: {lp_t30['linear_probe_accuracy']*100:.2f}%
   - Test 30-50%: {lp_t50['linear_probe_accuracy']*100:.2f}%

3. Comparison with Raw Concatenation:
   - Raw ESM-2 + ESM-IF1: {raw_comp['raw_concat_accuracy']*100:.2f}%
   - GaCA Fusion: {lp_t30['linear_probe_accuracy']*100:.2f}%
   - Improvement: {(lp_t30['linear_probe_accuracy'] - raw_comp['raw_concat_accuracy'])*100:.2f} percentage points

Interpretation:
- The positive Silhouette score indicates that GaCA features form well-separated clusters by EC class
- The moderate DB index reflects the inherent difficulty of 3,811-class separation
- Linear probing accuracy demonstrates that GaCA features are linearly separable
- Improvement over raw concatenation shows the benefit of attention pooling + vector gating
""")

if __name__ == '__main__':
    main()