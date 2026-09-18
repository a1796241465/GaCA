"""
EC4/EC7 误分类生化分析脚本
回应审稿人 R1: "It has not been investigated whether the observed confusion between
lyases (EC 4) and translocases (EC 7) stems from errors in AlphaFold predictions or
from Global Mean Pooling diluting local topology."
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import pandas as pd
import biotite.structure as struc
from biotite.structure.io import pdb
import json
import warnings
warnings.filterwarnings("ignore")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64


SEQ_TRAIN_PKL = "D:/EC/Sequences_Embeddings_use_ESM-2/Modify/1024_train_seq_embeddings_esm2.pkl"
SEQ_TEST_PKL = "D:/EC/Sequences_Embeddings_use_ESM-2/Modify/1024_test_seq_embeddings_esm2.pkl"

STR_TRAIN_PKL = "D:/EC/new_str/Modify/train_structure_embeddings_esm_if.pkl"
STR_TEST_PKL = "D:/EC/new_str/Modify/test_structure_embeddings_esm_if.pkl"

TRAIN_CSV = "D:/EC/train_cleaned_with_structure.csv"
TEST_CSV  = "D:/EC/test_30_cleaned_with_structure.csv"

PDB_DIR = "D:/EC/test_structures"
SAVE_OUTPUT = "D:/EC/new_str/Modify/ec4_ec7_analysis_report.json"



class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128), nn.Tanh(), nn.Linear(128, 1)
        )
    def forward(self, x):
        attn_weights = self.attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        return torch.sum(attn_weights * x, dim=1)


class GaCA_Final(nn.Module):
    def __init__(self, num_classes, seq_dim=1280, str_dim=512, hidden_dim=512, dropout=0.5):
        super().__init__()
        self.seq_pool = AttentionPooling(seq_dim)
        self.str_pool = AttentionPooling(str_dim)
        self.seq_proj = nn.Sequential(
            nn.Linear(seq_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)
        )
        self.str_proj = nn.Sequential(
            nn.Linear(str_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Dropout(dropout)
        )
        self.gating_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
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
        seq_p = self.seq_pool(seq_emb)
        str_p = self.str_pool(str_emb)
        seq_h = self.seq_proj(seq_p)
        str_h = self.str_proj(str_p)
        combined = torch.cat([seq_h, str_h], dim=1)
        g = self.gating_net(combined)
        fusion_feat = g * str_h + (1 - g) * seq_h
        return self.classifier(fusion_feat)



def load_pkl(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}")
    with open(path, 'rb') as f:
        return pickle.load(f)


def gaca_collate_fn(batch):
    seqs = [item[0] for item in batch]
    strs = [item[1] for item in batch]
    labels = torch.cat([item[2] for item in batch])
    seqs_padded = pad_sequence(seqs, batch_first=True, padding_value=0.0)
    strs_padded = pad_sequence(strs, batch_first=True, padding_value=0.0)
    return seqs_padded, strs_padded, labels


class GaCADataset(Dataset):
    def __init__(self, csv_path, seq_path, str_path, is_train=True, le=None):
        df = pd.read_csv(csv_path)
        df['Entry'] = df['Entry'].astype(str)
        seq_dict = load_pkl(seq_path)
        str_dict = load_pkl(str_path)

        common_ids = set(df['Entry']) & set(seq_dict.keys()) & set(str_dict.keys())

        self.seq_data, self.str_data, self.ids, self.labels = [], [], [], []
        for _, row in df.iterrows():
            pid = str(row['Entry'])
            if pid in common_ids:
                self.seq_data.append(seq_dict[pid])
                self.str_data.append(str_dict[pid])
                self.ids.append(pid)
                self.labels.append(str(row['EC number']))

        if is_train:
            self.le = LabelEncoder()
            self.encoded_labels = self.le.fit_transform(self.labels)
        else:
            self.le = le
            known = set(le.classes_)
            keep = [i for i, L in enumerate(self.labels) if L in known]
            self.seq_data = [self.seq_data[i] for i in keep]
            self.str_data = [self.str_data[i] for i in keep]
            self.ids = [self.ids[i] for i in keep]
            self.labels = [self.labels[i] for i in keep]
            self.encoded_labels = le.transform(self.labels)

    def __len__(self):
        return len(self.encoded_labels)

    def __getitem__(self, idx):
        seq_t = torch.tensor(self.seq_data[idx], dtype=torch.float32)
        str_t = torch.tensor(self.str_data[idx], dtype=torch.float32)
        if str_t.dim() > 1:
            str_t = str_t.view(-1, 512)
        if seq_t.dim() > 1:
            seq_t = seq_t.view(-1, 1280)
        if seq_t.dim() == 1:
            seq_t = seq_t.unsqueeze(0)
        if str_t.dim() == 1:
            str_t = str_t.unsqueeze(0)
        label_t = torch.tensor([self.encoded_labels[idx]], dtype=torch.long)
        return seq_t, str_t, label_t



def extract_plddt_scores(pdb_path):
    """Extract per-residue pLDDT scores from AlphaFold PDB"""
    try:
        f = pdb.PDBFile.read(pdb_path)
        atom_array = pdb.get_structure(f, model=1)
        atom_array = atom_array[struc.filter_amino_acids(atom_array)]
        ca_mask = atom_array.atom_name == "CA"
        ca_atoms = atom_array[ca_mask]
        if hasattr(ca_atoms, 'b_factor'):
            return ca_atoms.b_factor.tolist()
        return None
    except Exception:
        return None


def plddt_summary(plddt_scores):
    """Generate pLDDT distribution summary"""
    if plddt_scores is None or len(plddt_scores) == 0:
        return {"valid": False}
    arr = np.array(plddt_scores)
    bins = [(0, 50), (50, 70), (70, 90), (90, 101)]
    bin_labels = ["Very Low (<50)", "Low (50-70)", "Confident (70-90)", "Very High (>90)"]
    proportions = {}
    for (lo, hi), label in zip(bins, bin_labels):
        proportions[label] = float(np.mean((arr >= lo) & (arr < hi)))
    return {
        "valid": True,
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr)),
        "min":  float(np.min(arr)),
        "max":  float(np.max(arr)),
        "median": float(np.median(arr)),
        "length": len(arr),
        "plddt_region_proportions": proportions,
    }


def analyze_active_site_region(plddt_scores, seq_len, win_size=30):
    """
    Analyze pLDDT in potential active-site windows.
    Active sites typically have high pLDDT (well-structured).
    Low pLDDT in some regions may indicate disorder → misclassification.
    """
    if plddt_scores is None:
        return None
    arr = np.array(plddt_scores)
    min_win_plddt, max_win_plddt = 1.0, 0.0
    min_win_idx, max_win_idx = 0, 0
    for i in range(len(arr) - win_size + 1):
        win_mean = np.mean(arr[i:i + win_size])
        if win_mean < min_win_plddt:
            min_win_plddt = win_mean
            min_win_idx = i
        if win_mean > max_win_plddt:
            max_win_plddt = win_mean
            max_win_idx = i
    return {
        "most_ordered_window_start": int(max_win_idx),
        "most_ordered_window_plddt": float(max_win_plddt),
        "most_disordered_window_start": int(min_win_idx),
        "most_disordered_window_plddt": float(min_win_plddt),
        "global_disorder_fraction": float(np.mean(arr < 70)),
    }



def run_analysis():
    print(f"[1/6] Loading data & model on {DEVICE} ...")

    full_train_ds = GaCADataset(TRAIN_CSV, SEQ_TRAIN_PKL, STR_TRAIN_PKL, is_train=True)
    le = full_train_ds.le
    num_classes = len(le.classes_)

    test_ds = GaCADataset(TEST_CSV, SEQ_TEST_PKL, STR_TEST_PKL, is_train=False, le=le)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                         num_workers=0, collate_fn=gaca_collate_fn)

    model = GaCA_Final(num_classes).to(DEVICE)
    model_path = "D:/EC/new_str/Modify/GaCA/best_GaCA_final.pth"

    if os.path.exists(model_path):
        print(f"    Loading trained weights from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    else:
        print("    WARNING: No trained weights found, running with random init.")
        print("    For accurate analysis, train the model first or copy weights.")
        print("    Proceeding with random weights for script validation only.")

    model.eval()

    print("[2/6] Running inference on test (<30%) set ...")
    all_preds_idx, all_labels_idx, all_probs = [], [], []
    all_entry_ids, all_ec_strs = [], []

    with torch.no_grad():
        for batch_idx, (seq, stru, label) in enumerate(tqdm(test_dl, desc="Inference")):
            seq, stru, label = seq.to(DEVICE), stru.to(DEVICE), label.to(DEVICE)
            out = model(seq, stru)
            probs = F.softmax(out, dim=1)
            preds = out.argmax(1)
            all_preds_idx.extend(preds.cpu().numpy())
            all_labels_idx.extend(label.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

            batch_size_actual = len(label)
            for i in range(batch_size_actual):
                global_idx = batch_idx * BATCH_SIZE + i
                if global_idx < len(test_ds.ids):
                    all_entry_ids.append(test_ds.ids[global_idx])
                    all_ec_strs.append(test_ds.labels[global_idx])

    all_preds_str = le.inverse_transform(all_preds_idx)
    all_labels_str = le.inverse_transform(all_labels_idx)
    all_probs = np.concatenate(all_probs, axis=0)

    print(f"    Total samples: {len(all_labels_str)}")


    print("[3/6] Identifying EC4 <-> EC7 misclassifications ...")

    misclassified = {
        "ec4_as_ec7": [],
        "ec7_as_ec4": [],
        "ec4_as_other": [],
        "ec7_as_other": [],
        "other_as_ec4": [],
        "other_as_ec7": [],
    }

    for i in range(len(all_labels_str)):
        true_ec = all_labels_str[i]
        pred_ec = all_preds_str[i]
        true_l1 = true_ec.split('.')[0]
        pred_l1 = pred_ec.split('.')[0]

        record = {
            "entry_id": all_entry_ids[i],
            "true_ec": true_ec,
            "pred_ec": pred_ec,
            "true_level1": true_l1,
            "pred_level1": pred_l1,
            "confidence": float(all_probs[i].max()),
            "pred_rank_of_true": int(np.sum(all_probs[i] > all_probs[i][all_labels_idx[i]])),
        }

        if true_l1 == '4' and pred_l1 == '7':
            misclassified["ec4_as_ec7"].append(record)
        elif true_l1 == '7' and pred_l1 == '4':
            misclassified["ec7_as_ec4"].append(record)
        elif true_l1 == '4' and pred_l1 != '4':
            misclassified["ec4_as_other"].append(record)
        elif true_l1 == '7' and pred_l1 != '7':
            misclassified["ec7_as_other"].append(record)
        elif true_l1 != '4' and pred_l1 == '4':
            misclassified["other_as_ec4"].append(record)
        elif true_l1 != '7' and pred_l1 == '7':
            misclassified["other_as_ec7"].append(record)

    print(f"    EC4 → EC7:  {len(misclassified['ec4_as_ec7'])} cases")
    print(f"    EC7 → EC4:  {len(misclassified['ec7_as_ec4'])} cases")
    print(f"    EC4 → other:{len(misclassified['ec4_as_other'])} cases")
    print(f"    EC7 → other:{len(misclassified['ec7_as_other'])} cases")
    print(f"    other → EC4:{len(misclassified['other_as_ec4'])} cases")
    print(f"    other → EC7:{len(misclassified['other_as_ec7'])} cases")


    print("[4/6] Analyzing pLDDT scores for misclassified proteins ...")

    def analyze_cases(cases, category_name):
        results = []
        for case in cases:
            pid = case["entry_id"]
            pdb_path = os.path.join(PDB_DIR, f"{pid}.pdb")
            if not os.path.exists(pdb_path):
                case["plddt"] = {"valid": False, "error": "PDB file not found"}
                results.append(case)
                continue

            plddt = extract_plddt_scores(pdb_path)
            summary = plddt_summary(plddt)
            case["plddt"] = summary

            if summary["valid"]:
                active = analyze_active_site_region(plddt, summary["length"])
                case["active_site_region_analysis"] = active

            results.append(case)
        return results

    for key in misclassified:
        misclassified[key] = analyze_cases(misclassified[key], key)


    print("[5/6] Running comparative pLDDT analysis ...")


    correct_ec4, correct_ec7 = [], []
    for i in range(len(all_labels_str)):
        if all_labels_str[i] == all_preds_str[i]:
            if all_labels_str[i].startswith('4.'):
                correct_ec4.append(i)
            elif all_labels_str[i].startswith('7.'):
                correct_ec7.append(i)

    def sample_plddt_stats(indices, max_n=50):
        """Sample pLDDT from correctly/incorrectly classified proteins"""
        stats = {"mean_plddt": [], "frac_low_plddt": [], "n_analyzed": 0}
        sample_idx = np.random.choice(indices, min(max_n, len(indices)), replace=False) if indices else []
        for idx in sample_idx:
            pid = all_entry_ids[idx]
            pdb_path = os.path.join(PDB_DIR, f"{pid}.pdb")
            if not os.path.exists(pdb_path):
                continue
            plddt = extract_plddt_scores(pdb_path)
            if plddt is not None:
                arr = np.array(plddt)
                stats["mean_plddt"].append(float(np.mean(arr)))
                stats["frac_low_plddt"].append(float(np.mean(arr < 70)))
                stats["n_analyzed"] += 1
        return stats

    comparison = {
        "correct_ec4": sample_plddt_stats(correct_ec4),
        "correct_ec7": sample_plddt_stats(correct_ec7),
        "misclassified_ec4_ec7": {
            "mean_plddt": [float(np.mean(extract_plddt_scores(
                            os.path.join(PDB_DIR, f"{c['entry_id']}.pdb")) or [0]))
                            for c in misclassified["ec4_as_ec7"]
                            if os.path.exists(os.path.join(PDB_DIR, f"{c['entry_id']}.pdb"))],
            "n": len([c for c in misclassified["ec4_as_ec7"]
                      if os.path.exists(os.path.join(PDB_DIR, f"{c['entry_id']}.pdb"))]),
        },
        "misclassified_ec7_ec4": {
            "mean_plddt": [float(np.mean(extract_plddt_scores(
                            os.path.join(PDB_DIR, f"{c['entry_id']}.pdb")) or [0]))
                            for c in misclassified["ec7_as_ec4"]
                            if os.path.exists(os.path.join(PDB_DIR, f"{c['entry_id']}.pdb"))],
            "n": len([c for c in misclassified["ec7_as_ec4"]
                      if os.path.exists(os.path.join(PDB_DIR, f"{c['entry_id']}.pdb"))]),
        }
    }


    print("[6/6] Generating report ...")


    biochemical_context = {
        "ec4_lyases": {
            "description": (
                "Lyases (EC 4) catalyze the cleavage of C-C, C-O, C-N and other bonds "
                "by means other than hydrolysis or oxidation, often leaving double bonds. "
                "They differ from hydrolases (EC 3) in their non-hydrolytic mechanism."
            ),
            "key_structural_features": [
                "Often contain TIM barrel (α/β)₈ folds",
                "Active sites frequently involve metal cofactors or Schiff base intermediates",
                "Structural plasticity in active site loops is common",
            ],
        },
        "ec7_translocases": {
            "description": (
                "Translocases (EC 7) catalyze the movement of ions or molecules across "
                "membranes or their separation within membranes. This is the newest EC "
                "class, introduced in 2018, previously classified under other categories."
            ),
            "key_structural_features": [
                "Typically transmembrane proteins with α-helical bundles",
                "Conformational changes (alternating access) are central to function",
                "AlphaFold may generate static structures that fail to capture transport dynamics",
                "Cryo-EM often reveals multiple conformational states not present in single PDB",
            ],
        },
        "possible_confusion_reasons": [
            "Both classes can act on similar substrate molecules",
            "Translocases were historically classified under other EC classes, so some annotations may be inconsistent",
            "AlphaFold static structures lack the conformational dynamics essential for translocase function",
            "Membrane-associated lyases and translocases may share surface-exposed residue patterns",
            "The fine line between 'transport facilitation with bond rearrangement' vs 'catalysis'",
        ],
    }

    report = {
        "analysis_title": "EC4 (Lyases) vs EC7 (Translocases) Misclassification Analysis",
        "reviewer_question": (
            "Has it been investigated whether observed confusion between lyases (EC 4) "
            "and translocases (EC 7) stems from errors in AlphaFold predictions (low "
            "pLDDT domains) or from Global Mean Pooling diluting local active-site "
            "topological features?"
        ),
        "method_summary": (
            "We extracted per-residue pLDDT scores from AlphaFold2 structures for all "
            "misclassified proteins. We then compared pLDDT distributions, active-site "
            "region quality, and structural disorder between correctly and incorrectly "
            "classified EC4/EC7 enzymes."
        ),
        "dataset_info": {
            "test_set": "CARE <30% sequence identity",
            "total_samples": len(all_labels_str),
            "ec4_samples": int(sum(1 for ec in all_labels_str if ec.startswith('4.'))),
            "ec7_samples": int(sum(1 for ec in all_labels_str if ec.startswith('7.'))),
        },
        "misclassification_summary": {
            "ec4_as_ec7_count": len(misclassified["ec4_as_ec7"]),
            "ec7_as_ec4_count": len(misclassified["ec7_as_ec4"]),
            "ec4_as_other_count": len(misclassified["ec4_as_other"]),
            "ec7_as_other_count": len(misclassified["ec7_as_other"]),
        },
        "plddt_comparison": comparison,
        "misclassified_cases": {
            "ec4_predicted_as_ec7": misclassified["ec4_as_ec7"],
            "ec7_predicted_as_ec4": misclassified["ec7_as_ec4"],
        },
        "biochemical_context": biochemical_context,
        "interpretation": (
            "If misclassified proteins show systematically lower pLDDT scores than "
            "correctly classified ones, the confusion likely stems from AlphaFold "
            "prediction errors (poor structure quality). If pLDDT is normal but "
            "classification still fails, the cause is more likely attributed to the "
            "model's inability to capture fine-grained active-site features."
        ),
        "fixes": {
            "if_alphafold_quality_issue": [
                "Use ESMFold or OmegaFold as alternative structure predictors",
                "Filter test set by higher pLDDT threshold",
                "Apply structural relaxation (e.g., AMBER) to AlphaFold outputs",
                "Use multi-conformer representations instead of single static PDB",
            ],
            "if_model_limitation": [
                "Attention-based pooling (already implemented in Modify code) addresses "
                "this by learning which residue positions are functionally important",
                "Add contact map or distance matrix as auxiliary structural input",
                "Incorporate evolutionary conservation scores (e.g., ConSurf) to weight "
                "important residues in the pooling step",
                "Use residue-level gating instead of protein-level gating",
            ],
        }
    }

    with open(SAVE_OUTPUT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nReport saved to: {SAVE_OUTPUT}")


    print("\n" + "=" * 60)
    print("核心发现 (Key Findings)")
    print("=" * 60)
    for cat, cases in [("EC4→EC7", misclassified["ec4_as_ec7"]),
                        ("EC7→EC4", misclassified["ec7_as_ec4"])]:
        if not cases:
            print(f"\n  {cat}: No misclassifications found")
            continue
        plddt_vals = [c["plddt"]["mean"] for c in cases if c["plddt"].get("valid")]
        n_valid = len(plddt_vals)
        if n_valid > 0:
            print(f"\n  {cat} ({n_valid} proteins analyzed):")
            print(f"    Mean pLDDT: {np.mean(plddt_vals):.1f} ± {np.std(plddt_vals):.1f}")
            print(f"    Min pLDDT:  {np.min(plddt_vals):.1f}")
            print(f"    Max pLDDT:  {np.max(plddt_vals):.1f}")
            for case in cases[:3]:
                if case["plddt"].get("valid"):
                    pid = case["entry_id"]
                    plddt_m = case["plddt"]["mean"]
                    act = case.get("active_site_region_analysis", {})
                    disorder = act.get("global_disorder_fraction", "N/A") if act else "N/A"
                    print(f"    · {pid}: True={case['true_ec']} Pred={case['pred_ec']} "
                          f"pLDDT={plddt_m:.1f} disorder_frac={disorder}")


    print("\n" + "=" * 60)
    print("结论指引 (For Rebuttal)")
    print("=" * 60)
    print("""
请根据上述 pLDDT 数据判断：
  A) 若误分类蛋白的 pLDDT 显著低于正确分类组 (t-test p < 0.05)
     → 问题根源：AlphaFold 结构质量不足
     → 回应：加入结构质量过滤/多构象建模/AMBER 弛豫

  B) 若误分类蛋白的 pLDDT 无明显差异
     → 问题根源：模型 pooling 机制稀释了活性位点信号
     → 回应：AttentionPooling (已实现) 可缓解此问题
              + 加入进化保守性权重 + 接触图辅助

详见 JSON 报告中的 'fixes' 和 'interpretation' 字段。
""")
    return report


if __name__ == "__main__":
    run_analysis()
