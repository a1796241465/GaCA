import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import os


MODEL_PATH = '/home/lihaotian/new_ec/new_str/test_30/best_GACMA_final.pth'
TRAIN_CSV = '/home/lihaotian/new_ec/train_cleaned_with_structure.csv'
SEQ_TRAIN_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/1024_train_seq_embeddings_esm2_mean.pkl'
STR_TRAIN_PKL = '/home/lihaotian/new_ec/new_str/test_30/train_structure_embeddings_esm_if.pkl'

TEST_CSV = '/home/lihaotian/new_ec/test_30_cleaned_with_structure.csv'
SEQ_TEST_PKL = '/home/lihaotian/new_ec/Sequences_Embeddings_use_ESM-2/1024_test_seq_embeddings_esm2_mean.pkl'
STR_TEST_PKL = '/home/lihaotian/new_ec/new_str/test_30/test_structure_embeddings_esm_if.pkl'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams['font.family'] = 'serif'



class GACMA_With_Hooks(nn.Module):
    def __init__(self, num_classes, seq_dim=1280, str_dim=512, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.seq_proj = nn.Sequential(nn.Linear(seq_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.str_proj = nn.Sequential(nn.Linear(str_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout))
        self.gating_net = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1),
                                        nn.Sigmoid())
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(dropout),
                                        nn.Linear(512, num_classes))

    def forward(self, seq_emb, str_emb):
        seq_h = self.seq_proj(seq_emb)
        str_h = self.str_proj(str_emb)
        combined = torch.cat([seq_h, str_h], dim=1)
        g = self.gating_net(combined)
        fusion_feat = g * str_h + (1 - g) * seq_h
        logits = self.classifier(fusion_feat)
        return logits, fusion_feat, g, seq_h



def load_pkl(path):
    if not os.path.exists(path): raise FileNotFoundError(f"❌ 文件不存在: {path}")
    with open(path, 'rb') as f: return pickle.load(f)


class AnalysisDataset(Dataset):
    def __init__(self, csv_path, seq_pkl, str_pkl, le):
        df = pd.read_csv(csv_path)
        self.seq_data = load_pkl(seq_pkl)
        self.str_data = load_pkl(str_pkl)
        self.ids, self.labels, self.l1_labels = [], [], []
        valid_classes = set(le.classes_)
        seq_keys, str_keys = set(self.seq_data.keys()), set(self.str_data.keys())

        for _, row in df.iterrows():
            uid, ec = str(row['Entry']), str(row['EC number'])
            if uid in seq_keys and uid in str_keys and ec in valid_classes:
                self.ids.append(uid)
                self.labels.append(le.transform([ec])[0])
                self.l1_labels.append(ec.split('.')[0])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        uid = self.ids[idx]
        s, t = self.seq_data[uid], self.str_data[uid]


        s_tensor = torch.FloatTensor(s)
        t_tensor = torch.FloatTensor(t)
        if s_tensor.ndim > 1: s_tensor = s_tensor.view(-1)
        if t_tensor.ndim > 1: t_tensor = t_tensor.view(-1)


        s_tensor = s_tensor / (torch.norm(s_tensor) + 1e-6)
        t_tensor = t_tensor / (torch.norm(t_tensor) + 1e-6)

        return s_tensor, t_tensor, self.labels[idx], self.l1_labels[idx]




def plot_gating_dist_clean(g_values):
    print("🎨 正在绘制门控分布图 (裁剪版)...")
    g_values = np.array(g_values)

    plt.figure(figsize=(7, 5))



    sns.histplot(g_values, bins='auto', kde=True,
                 color="#e74c3c", edgecolor="white", alpha=0.7, line_kws={'linewidth': 2})


    plt.xlabel("Structural Confidence Weight ($g$)", fontweight='bold', fontsize=12)


    plt.ylabel("Frequency", fontweight='bold', fontsize=12)


    plt.title("Distribution of Gating Weights", fontweight='bold', fontsize=14, pad=15)



    max_val = g_values.max()
    plt.xlim(0, max_val * 1.15)

    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig("Fig_Advanced_Gating.pdf", dpi=300)
    print("✅ 门控分布图已保存: Fig_Advanced_Gating.pdf")


def plot_tsne(features_before, features_after, labels_l1):
    print("🎨 正在绘制 t-SNE...")
    unique_labels = sorted(list(set(labels_l1)))
    label_map = {l: i for i, l in enumerate(unique_labels)}
    c_ids = [label_map[l] for l in labels_l1]
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto', n_jobs=8)
    emb_before = tsne.fit_transform(features_before)
    emb_after = tsne.fit_transform(features_after)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    scatter1 = axes[0].scatter(emb_before[:, 0], emb_before[:, 1], c=c_ids, cmap='tab10', s=15, alpha=0.6)


    axes[0].set_title("Raw ESM-2 Features", fontweight='bold');
    axes[0].axis('off')

    scatter2 = axes[1].scatter(emb_after[:, 0], emb_after[:, 1], c=c_ids, cmap='tab10', s=15, alpha=0.6)


    axes[1].set_title("GACMA Fused Features", fontweight='bold');
    axes[1].axis('off')

    handles, _ = scatter1.legend_elements(prop="colors")
    fig.legend(handles, [f"Class {l}" for l in unique_labels], loc='lower center', ncol=7, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout()
    plt.savefig("Fig_Advanced_TSNE.pdf", dpi=300, bbox_inches='tight')


def plot_confusion_matrix(y_true_l1, y_pred_l1):
    print("🎨 正在绘制 CM...")
    unique_labels = sorted(list(set(y_true_l1)))
    cm = confusion_matrix(y_true_l1, y_pred_l1, labels=unique_labels)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-6)
    plt.figure(figsize=(8, 7))


    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=[f"EC {l}" for l in unique_labels],
                yticklabels=[f"EC {l}" for l in unique_labels])


    plt.xlabel('Predicted Label', fontweight='bold')
    plt.ylabel('True Label', fontweight='bold')

    plt.tight_layout()
    plt.savefig("Fig_Advanced_CM.pdf", dpi=300)



def run_analysis():
    print(">>> 初始化...")
    train_df = pd.read_csv(TRAIN_CSV)
    train_seq_keys = set(load_pkl(SEQ_TRAIN_PKL).keys())
    train_str_keys = set(load_pkl(STR_TRAIN_PKL).keys())
    valid_train_ids = set(train_df['Entry'].astype(str)) & train_seq_keys & train_str_keys
    train_df_filtered = train_df[train_df['Entry'].astype(str).isin(valid_train_ids)]
    le = LabelEncoder()
    le.fit(train_df_filtered['EC number'].astype(str))
    num_classes = len(le.classes_)

    ds = AnalysisDataset(TEST_CSV, SEQ_TEST_PKL, STR_TEST_PKL, le)
    dl = DataLoader(ds, batch_size=64, shuffle=False)

    model = GACMA_With_Hooks(num_classes).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    feats_before, feats_after, g_vals = [], [], []
    l1_trues, l1_preds = [], []

    print(">>> 推理中...")
    with torch.no_grad():
        for seq, stru, _, l1_label in tqdm(dl):
            seq, stru = seq.to(DEVICE), stru.to(DEVICE)
            logits, fusion_f, g, seq_f = model(seq, stru)

            feats_before.append(seq_f.cpu().numpy())
            feats_after.append(fusion_f.cpu().numpy())
            g_vals.extend(g.cpu().numpy().flatten())
            l1_trues.extend(l1_label)

            preds = logits.argmax(1).cpu().numpy()
            l1_preds.extend([ec.split('.')[0] for ec in le.inverse_transform(preds)])

    feats_before = np.concatenate(feats_before, axis=0)
    feats_after = np.concatenate(feats_after, axis=0)


    plot_tsne(feats_before, feats_after, l1_trues)
    plot_gating_dist_clean(g_vals)
    plot_confusion_matrix(l1_trues, l1_preds)
    print("\n🎉 完成！所有图表已保存。")


if __name__ == "__main__":
    run_analysis()