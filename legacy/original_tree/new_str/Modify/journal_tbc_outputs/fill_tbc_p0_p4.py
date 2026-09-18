"""
Generate journal TBC tables for GaCA P0-P4.

This script is intentionally read-only with respect to model outputs. It aligns
GaCA predictions with the strict closed-set BLASTp results, computes paired
statistics, long-tail analyses, per-EC4-label summaries, and case-study candidates.

Outputs are written to:
    D:/EC/new_str/Modify/journal_tbc_outputs
"""

from __future__ import annotations

import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, matthews_corrcoef


ROOT = Path(r"D:/EC")
MODIFY = ROOT / "new_str" / "Modify"
ANALYSIS = MODIFY / "analysis_outputs"
OUT = MODIFY / "journal_tbc_outputs"

TRAIN_FASTA_EFFECTIVE = ROOT / "baseline" / "test_30" / "blast_temp" / "train_strict.fasta"

DATASETS = {
    "lt30": {
        "label": "<30%",
        "gaca_json": ANALYSIS / "predictions_lt30.json",
        "gaca_probs": ANALYSIS / "predictions_lt30_probs.npy",
        "blast": ROOT / "baseline" / "test_30" / "blast_temp" / "results_strict.txt",
        "test_csv": ROOT / "test_30_cleaned_with_structure.csv",
        "seq_pkl": ROOT / "Sequences_Embeddings_use_ESM-2" / "Modify" / "1024_test_seq_embeddings_esm2.pkl",
        "str_pkl": MODIFY / "test_structure_embeddings_esm_if.pkl",
    },
    "30-50": {
        "label": "30-50%",
        "gaca_json": ANALYSIS / "predictions_30-50.json",
        "gaca_probs": ANALYSIS / "predictions_30-50_probs.npy",
        "blast": ROOT / "baseline" / "test_30-50" / "blast_temp_30_50" / "results.txt",
        "test_csv": ROOT / "test_30_50_clean.csv",
        "seq_pkl": ROOT / "Sequences_Embeddings_use_ESM-2" / "Modify" / "30_50_seq_embeddings_esm2.pkl",
        "str_pkl": MODIFY / "30-50test_structure_embeddings_esm_if.pkl",
    },
}

BOOTSTRAP_SEED = 42
N_BOOT = 5000


@dataclass
class BlastHit:
    query_id: str
    subject_id: str
    pred_ec: str
    pident: float | None
    evalue: float | None
    bitscore: float | None
    raw_subject: str


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def parse_ec_from_header(header: str) -> Tuple[str, str]:
    token = header.strip().lstrip(">").split()[0]
    if "|" not in token:
        return token, ""
    entry, ec = token.split("|", 1)
    return entry, ec


def parse_effective_train_counts(path: Path) -> Counter:
    counts: Counter = Counter()
    entries = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith(">"):
                _, ec = parse_ec_from_header(line)
                if ec:
                    counts[ec] += 1
                    entries += 1
    if not counts:
        raise RuntimeError(f"No EC labels parsed from {path}")
    return counts


def parse_blast(path: Path) -> Dict[str, BlastHit]:
    hits: Dict[str, BlastHit] = {}
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            query_id = parts[0].split("|")[0]
            subject = parts[1]
            subject_id, pred_ec = parse_ec_from_header(subject)
            if not pred_ec:
                pred_ec = "No_Hit"
            pident = safe_float(parts[2]) if len(parts) > 2 else None
            evalue = safe_float(parts[10]) if len(parts) > 10 else safe_float(parts[3]) if len(parts) > 3 else None
            bitscore = safe_float(parts[11]) if len(parts) > 11 else None
            if query_id not in hits:
                hits[query_id] = BlastHit(
                    query_id=query_id,
                    subject_id=subject_id,
                    pred_ec=pred_ec,
                    pident=pident,
                    evalue=evalue,
                    bitscore=bitscore,
                    raw_subject=subject,
                )
    return hits


def load_pkl_keys(path: Path) -> set:
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, dict):
        return set(str(k) for k in obj.keys())
    raise TypeError(f"Expected a dict-like embedding file: {path}")


def safe_float(value: str) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def load_gaca(json_path: Path, probs_path: Path) -> dict:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    probs = np.load(probs_path)
    if len(data["ids"]) != probs.shape[0]:
        raise RuntimeError(f"Probability row mismatch for {json_path}")
    data["probs"] = probs
    return data


def level_correct(true_ec: str, pred_ec: str, level: int) -> int:
    if not pred_ec or pred_ec == "No_Hit":
        return 0
    t = str(true_ec).split(".")
    p = str(pred_ec).split(".")
    idx = level - 1
    return int(len(t) > idx and len(p) > idx and t[:level] == p[:level])


def all_level_correct(true_ec: str, pred_ec: str) -> List[int]:
    return [level_correct(true_ec, pred_ec, level) for level in range(1, 5)]


def metric_summary(y_true: Sequence[str], y_pred: Sequence[str]) -> dict:
    n = len(y_true)
    levels = np.array([all_level_correct(t, p) for t, p in zip(y_true, y_pred)], dtype=float)
    labels = sorted(set(y_true))
    return {
        "n": n,
        "level1_acc": float(levels[:, 0].mean()),
        "level2_acc": float(levels[:, 1].mean()),
        "level3_acc": float(levels[:, 2].mean()),
        "level4_acc": float(levels[:, 3].mean()),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "level4_correct": int(levels[:, 3].sum()),
    }


def exact_mcnemar_p(gaca_correct: Sequence[int], blast_correct: Sequence[int]) -> Tuple[int, int, float]:
    g = np.asarray(gaca_correct, dtype=bool)
    b = np.asarray(blast_correct, dtype=bool)
    gaca_only = int(np.sum(g & ~b))
    blast_only = int(np.sum(b & ~g))
    discordant = gaca_only + blast_only
    if discordant == 0:
        return gaca_only, blast_only, 1.0
    k = min(gaca_only, blast_only)
    prob = sum(math.comb(discordant, i) for i in range(k + 1)) / (2 ** discordant)
    return gaca_only, blast_only, min(1.0, 2 * prob)


def bootstrap_diff_ci(
    y_true: Sequence[str],
    gaca_pred: Sequence[str],
    blast_pred: Sequence[str],
    metric: str,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
) -> Tuple[float, float, float, float]:
    y_true = np.asarray(y_true)
    gaca_pred = np.asarray(gaca_pred)
    blast_pred = np.asarray(blast_pred)
    n = len(y_true)

    if metric == "level4_acc":
        g_ok = np.array([level_correct(t, p, 4) for t, p in zip(y_true, gaca_pred)], dtype=float)
        b_ok = np.array([level_correct(t, p, 4) for t, p in zip(y_true, blast_pred)], dtype=float)
        observed = float(g_ok.mean() - b_ok.mean())
        boot = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            boot[i] = float(g_ok[idx].mean() - b_ok[idx].mean())
    elif metric == "macro_f1":
        labels = sorted(set(y_true))
        observed = float(
            f1_score(y_true, gaca_pred, labels=labels, average="macro", zero_division=0)
            - f1_score(y_true, blast_pred, labels=labels, average="macro", zero_division=0)
        )
        boot = np.empty(n_boot, dtype=float)
        for i in range(n_boot):
            idx = rng.integers(0, n, size=n)
            yt = y_true[idx]
            labels_i = sorted(set(yt))
            boot[i] = float(
                f1_score(yt, gaca_pred[idx], labels=labels_i, average="macro", zero_division=0)
                - f1_score(yt, blast_pred[idx], labels=labels_i, average="macro", zero_division=0)
            )
    else:
        raise ValueError(metric)

    low, high = np.percentile(boot, [2.5, 97.5])
    p_boot = 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
    return observed, float(low), float(high), min(1.0, p_boot)


def train_count_bin(count: int) -> str:
    if count <= 2:
        return "1-2"
    if count <= 5:
        return "3-5"
    if count <= 10:
        return "6-10"
    return ">10"


def aligned_records(dataset_key: str, train_counts: Counter) -> Tuple[pd.DataFrame, dict]:
    cfg = DATASETS[dataset_key]
    gaca = load_gaca(cfg["gaca_json"], cfg["gaca_probs"])
    blast = parse_blast(cfg["blast"])

    rows = []
    for i, pid in enumerate(gaca["ids"]):
        true_ec = str(gaca["true_ec"][i])
        gaca_pred = str(gaca["pred_ec"][i])
        hit = blast.get(pid)
        blast_pred = hit.pred_ec if hit else "No_Hit"
        probs = gaca["probs"][i]
        pred_idx = int(gaca["pred_idx"][i])
        true_idx = int(gaca["true_idx"][i])
        train_count = int(train_counts.get(true_ec, 0))

        rows.append(
            {
                "dataset": dataset_key,
                "dataset_label": cfg["label"],
                "id": pid,
                "true_ec": true_ec,
                "gaca_pred_ec": gaca_pred,
                "blast_pred_ec": blast_pred,
                "gaca_confidence": float(np.max(probs)),
                "gaca_true_confidence": float(probs[true_idx]),
                "gaca_pred_confidence": float(probs[pred_idx]),
                "gaca_correct_l1": level_correct(true_ec, gaca_pred, 1),
                "gaca_correct_l2": level_correct(true_ec, gaca_pred, 2),
                "gaca_correct_l3": level_correct(true_ec, gaca_pred, 3),
                "gaca_correct_l4": level_correct(true_ec, gaca_pred, 4),
                "blast_correct_l1": level_correct(true_ec, blast_pred, 1),
                "blast_correct_l2": level_correct(true_ec, blast_pred, 2),
                "blast_correct_l3": level_correct(true_ec, blast_pred, 3),
                "blast_correct_l4": level_correct(true_ec, blast_pred, 4),
                "blast_subject_id": hit.subject_id if hit else "",
                "blast_pident": hit.pident if hit else np.nan,
                "blast_evalue": hit.evalue if hit else np.nan,
                "blast_bitscore": hit.bitscore if hit else np.nan,
                "train_count_true_ec": train_count,
                "train_count_bin": train_count_bin(train_count),
            }
        )
    df = pd.DataFrame(rows)
    return df, gaca


def make_metrics(aligned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, df in aligned.items():
        for method, pred_col in [("BLASTp_strict", "blast_pred_ec"), ("GaCA", "gaca_pred_ec")]:
            s = metric_summary(df["true_ec"].tolist(), df[pred_col].tolist())
            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "n": s["n"],
                    "level1_acc_pct": 100 * s["level1_acc"],
                    "level2_acc_pct": 100 * s["level2_acc"],
                    "level3_acc_pct": 100 * s["level3_acc"],
                    "level4_acc_pct": 100 * s["level4_acc"],
                    "macro_f1": s["macro_f1"],
                    "mcc": s["mcc"],
                    "level4_correct": s["level4_correct"],
                }
            )
    return pd.DataFrame(rows)


def make_paired(aligned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows = []
    for dataset, df in aligned.items():
        y_true = df["true_ec"].tolist()
        gaca_pred = df["gaca_pred_ec"].tolist()
        blast_pred = df["blast_pred_ec"].tolist()
        gaca_l4 = df["gaca_correct_l4"].tolist()
        blast_l4 = df["blast_correct_l4"].tolist()
        gaca_only, blast_only, p_mcnemar = exact_mcnemar_p(gaca_l4, blast_l4)

        for metric in ["level4_acc", "macro_f1"]:
            observed, low, high, p_boot = bootstrap_diff_ci(y_true, gaca_pred, blast_pred, metric, rng)
            g_metric = metric_summary(y_true, gaca_pred)
            b_metric = metric_summary(y_true, blast_pred)
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "blast": b_metric[metric] if metric in b_metric else b_metric["level4_acc"],
                    "gaca": g_metric[metric] if metric in g_metric else g_metric["level4_acc"],
                    "difference_gaca_minus_blast": observed,
                    "diff_ci_low": low,
                    "diff_ci_high": high,
                    "bootstrap_p_two_sided": p_boot,
                    "mcnemar_exact_p_l4": p_mcnemar if metric == "level4_acc" else np.nan,
                    "gaca_only_l4": gaca_only if metric == "level4_acc" else np.nan,
                    "blast_only_l4": blast_only if metric == "level4_acc" else np.nan,
                    "n_bootstrap": N_BOOT,
                }
            )
    return pd.DataFrame(rows)


def make_long_tail(aligned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    order = ["1-2", "3-5", "6-10", ">10"]
    rows = []
    for dataset, df in aligned.items():
        for bin_name in order:
            sub = df[df["train_count_bin"] == bin_name]
            if sub.empty:
                rows.append(
                    {
                        "dataset": dataset,
                        "train_examples_per_ec_bin": bin_name,
                        "number_of_ec_classes": 0,
                        "test_proteins": 0,
                        "blast_acc_pct": np.nan,
                        "gaca_acc_pct": np.nan,
                        "delta_pct": np.nan,
                        "blast_macro_f1": np.nan,
                        "gaca_macro_f1": np.nan,
                    }
                )
                continue
            b = metric_summary(sub["true_ec"].tolist(), sub["blast_pred_ec"].tolist())
            g = metric_summary(sub["true_ec"].tolist(), sub["gaca_pred_ec"].tolist())
            rows.append(
                {
                    "dataset": dataset,
                    "train_examples_per_ec_bin": bin_name,
                    "number_of_ec_classes": int(sub["true_ec"].nunique()),
                    "test_proteins": int(len(sub)),
                    "blast_acc_pct": 100 * b["level4_acc"],
                    "gaca_acc_pct": 100 * g["level4_acc"],
                    "delta_pct": 100 * (g["level4_acc"] - b["level4_acc"]),
                    "blast_macro_f1": b["macro_f1"],
                    "gaca_macro_f1": g["macro_f1"],
                }
            )
    return pd.DataFrame(rows)


def make_per_class(aligned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset, df in aligned.items():
        for ec, sub in df.groupby("true_ec"):
            for method, pred_col, ok_col in [
                ("BLASTp_strict", "blast_pred_ec", "blast_correct_l4"),
                ("GaCA", "gaca_pred_ec", "gaca_correct_l4"),
            ]:
                rows.append(
                    {
                        "dataset": dataset,
                        "ec": ec,
                        "train_count": int(sub["train_count_true_ec"].iloc[0]),
                        "train_count_bin": sub["train_count_bin"].iloc[0],
                        "method": method,
                        "test_proteins": int(len(sub)),
                        "correct_l4": int(sub[ok_col].sum()),
                        "accuracy_l4_pct": 100 * float(sub[ok_col].mean()),
                    }
                )
    return pd.DataFrame(rows)


def choose_case_rows(aligned: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    case_specs = [
        ("GaCA_correct_BLASTp_wrong", lambda d: (d["gaca_correct_l4"] == 1) & (d["blast_correct_l4"] == 0)),
        ("Both_correct", lambda d: (d["gaca_correct_l4"] == 1) & (d["blast_correct_l4"] == 1)),
        ("BLASTp_correct_GaCA_wrong", lambda d: (d["gaca_correct_l4"] == 0) & (d["blast_correct_l4"] == 1)),
    ]
    for dataset, df in aligned.items():
        for case_type, mask_fn in case_specs:
            sub = df[mask_fn(df)].copy()
            if sub.empty:
                continue
            if case_type == "BLASTp_correct_GaCA_wrong":
                sub = sub.sort_values(["gaca_confidence", "blast_pident"], ascending=[False, False])
            else:
                sub = sub.sort_values(["gaca_confidence", "blast_pident"], ascending=[False, False])
            r = sub.iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "case_type": case_type,
                    "query_id": r["id"],
                    "true_ec": r["true_ec"],
                    "blast_pred_ec": r["blast_pred_ec"],
                    "gaca_pred_ec": r["gaca_pred_ec"],
                    "gaca_confidence": r["gaca_confidence"],
                    "gaca_true_confidence": r["gaca_true_confidence"],
                    "blast_subject_id": r["blast_subject_id"],
                    "blast_pident": r["blast_pident"],
                    "blast_evalue": r["blast_evalue"],
                    "blast_bitscore": r["blast_bitscore"],
                    "interpretation_stub": interpretation_stub(case_type),
                }
            )
    return pd.DataFrame(rows)


def interpretation_stub(case_type: str) -> str:
    if case_type == "GaCA_correct_BLASTp_wrong":
        return (
            "Candidate example where adaptive sequence-structure fusion corrects a homology-transfer error. "
            "Verify against UniProt/Rhea/BRENDA or active-site evidence before making a mechanistic claim."
        )
    if case_type == "Both_correct":
        return (
            "Positive-control example supported by both homology transfer and GaCA. "
            "Use it to show agreement between the model and conventional annotation evidence."
        )
    return (
        "Failure case where GaCA misses a label recovered by BLASTp. "
        "Discuss whether the error is hierarchy-consistent, rare-class driven, or due to missing biochemical context."
    )


def make_dataset_accounting(aligned: Dict[str, pd.DataFrame], train_counts: Counter) -> pd.DataFrame:
    train_csv = pd.read_csv(ROOT / "train_cleaned_with_structure.csv")
    rows = [
        {
            "subset": "training",
            "csv_rows": int(len(train_csv)),
            "effective_closed_set_rows": int(sum(train_counts.values())),
            "unique_ec_csv": int(train_csv["EC number"].astype(str).nunique()),
            "unique_ec_effective_or_closed": int(len(train_counts)),
        }
    ]
    for dataset, cfg in DATASETS.items():
        test_csv = pd.read_csv(cfg["test_csv"])
        test_csv["Entry"] = test_csv["Entry"].astype(str)
        seq_keys = load_pkl_keys(cfg["seq_pkl"])
        str_keys = load_pkl_keys(cfg["str_pkl"])
        feature_complete = test_csv[test_csv["Entry"].isin(seq_keys & str_keys)].copy()
        known_ec = set(train_counts.keys())
        closed_set = feature_complete[feature_complete["EC number"].astype(str).isin(known_ec)]
        df = aligned[dataset]
        rows.append(
            {
                "subset": dataset,
                "csv_rows": int(len(test_csv)),
                "feature_complete_rows": int(len(feature_complete)),
                "effective_closed_set_rows": int(len(df)),
                "unique_ec_csv": int(test_csv["EC number"].astype(str).nunique()),
                "unique_ec_feature_complete": int(feature_complete["EC number"].astype(str).nunique()),
                "unique_ec_effective_or_closed": int(df["true_ec"].nunique()),
                "removed_missing_feature_rows": int(len(test_csv) - len(feature_complete)),
                "removed_unseen_ec_rows_after_feature_filter": int(len(feature_complete) - len(closed_set)),
            }
        )
    return pd.DataFrame(rows)


def pct(x: float) -> str:
    return f"{100 * x:.2f}"


def write_latex(metrics: pd.DataFrame, paired: pd.DataFrame, long_tail: pd.DataFrame, cases: pd.DataFrame) -> None:
    lines: List[str] = []
    lines.append("% Auto-generated by fill_tbc_p0_p4.py. Review before pasting into the manuscript.\n")

    lines.append("% Strict closed-set main metrics")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Strict closed-set hierarchical EC accuracy after aligning BLASTp and GaCA on identical proteins.}")
    lines.append("\\label{tab:strict_closed_set_main}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llrrrrrr}")
    lines.append("\\toprule")
    lines.append("Dataset & Method & N & Level 1 & Level 2 & Level 3 & Level 4 & Macro-F1 \\\\")
    lines.append("\\midrule")
    label_map = {k: v["label"].replace("%", "\\%") for k, v in DATASETS.items()}
    for _, r in metrics.iterrows():
        dataset_label = label_map.get(r["dataset"], r["dataset"])
        lines.append(
            f"{dataset_label} & {r['method']} & {int(r['n'])} & "
            f"{r['level1_acc_pct']:.2f} & {r['level2_acc_pct']:.2f} & "
            f"{r['level3_acc_pct']:.2f} & {r['level4_acc_pct']:.2f} & {r['macro_f1']:.4f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table*}\n")

    lines.append("% Paired comparison table")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Paired comparison between strict BLASTp and GaCA. Differences are GaCA minus BLASTp.}")
    lines.append("\\label{tab:paired_strict}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llrrrrrr}")
    lines.append("\\toprule")
    lines.append("Dataset & Metric & BLASTp & GaCA & Difference & 95\\% CI & $p$-value & Discordant L4 \\\\")
    lines.append("\\midrule")
    for _, r in paired.iterrows():
        dataset_label = label_map.get(r["dataset"], r["dataset"])
        if r["metric"] == "level4_acc":
            b = 100 * r["blast"]
            g = 100 * r["gaca"]
            d = 100 * r["difference_gaca_minus_blast"]
            ci = f"[{100*r['diff_ci_low']:.2f}, {100*r['diff_ci_high']:.2f}]"
            pval = f"{r['mcnemar_exact_p_l4']:.4g}"
            disc = f"{int(r['gaca_only_l4'])}/{int(r['blast_only_l4'])}"
            metric_label = "Level-4 accuracy"
        else:
            b = r["blast"]
            g = r["gaca"]
            d = r["difference_gaca_minus_blast"]
            ci = f"[{r['diff_ci_low']:.4f}, {r['diff_ci_high']:.4f}]"
            pval = f"{r['bootstrap_p_two_sided']:.4g}"
            disc = "--"
            metric_label = "Macro-F1"
        lines.append(f"{dataset_label} & {metric_label} & {b:.4f} & {g:.4f} & {d:.4f} & {ci} & {pval} & {disc} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table*}\n")

    lines.append("% Long-tail table")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Long-tail analysis by the number of effective training proteins per fourth-level EC label.}")
    lines.append("\\label{tab:long_tail_filled}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llrrrrrr}")
    lines.append("\\toprule")
    lines.append("Dataset & Training examples per EC4 label & No. of EC4 labels & Test proteins & BLASTp Acc. & GaCA Acc. & Delta & GaCA Macro-F1 \\\\")
    lines.append("\\midrule")
    for _, r in long_tail.iterrows():
        dataset_label = label_map.get(r["dataset"], r["dataset"])
        lines.append(
            f"{dataset_label} & {r['train_examples_per_ec_bin']} & {int(r['number_of_ec_classes'])} & "
            f"{int(r['test_proteins'])} & {r['blast_acc_pct']:.2f} & {r['gaca_acc_pct']:.2f} & "
            f"{r['delta_pct']:.2f} & {r['gaca_macro_f1']:.4f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table*}\n")

    lines.append("% Case-study candidates")
    lines.append("\\begin{table*}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Representative case-study candidates selected from strict closed-set predictions.}")
    lines.append("\\label{tab:case_studies_filled}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{lllllll}")
    lines.append("\\toprule")
    lines.append("Dataset & Case type & Query ID & True EC & BLASTp EC & GaCA EC & GaCA confidence \\\\")
    lines.append("\\midrule")
    for _, r in cases.iterrows():
        dataset_label = label_map.get(r["dataset"], r["dataset"])
        lines.append(
            f"{dataset_label} & {r['case_type'].replace('_', ' ')} & {r['query_id']} & "
            f"{r['true_ec']} & {r['blast_pred_ec']} & {r['gaca_pred_ec']} & {r['gaca_confidence']:.3f} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table*}\n")

    (OUT / "latex_table_snippets_p0_p4.tex").write_text("\n".join(lines), encoding="utf-8")


def write_manifest() -> None:
    rows = [
        ("strict_closed_set_predictions_lt30.csv", "Aligned per-protein predictions for <30% closed-set proteins."),
        ("strict_closed_set_predictions_30-50.csv", "Aligned per-protein predictions for 30-50% closed-set proteins."),
        ("strict_protocol_metrics.csv", "P0 main metrics for strict BLASTp and GaCA."),
        ("paired_bootstrap_tests.csv", "P1 paired bootstrap confidence intervals and p-values."),
        ("long_tail_analysis.csv", "P2 long-tail frequency-bin analysis."),
        ("per_class_performance.csv", "P4 supplementary per-EC4-label performance table."),
        ("case_study_candidates.csv", "P3 candidate examples for manuscript case studies."),
        ("dataset_accounting.csv", "Dataset row and EC4-label accounting."),
        ("latex_table_snippets_p0_p4.tex", "LaTeX snippets for manuscript replacement."),
        ("summary_p0_p4.json", "Machine-readable summary of key results."),
        ("Fig_Main_Results_strict.pdf", "Strict closed-set BLASTp vs GaCA main result figure."),
        ("Fig_Main_Results_strict.png", "PNG version of the strict main result figure."),
        ("fill_tbc_p0_p4.py", "Reproducible analysis script."),
    ]
    with (OUT / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file", "description"])
        writer.writerows(rows)


def make_main_results_figure(metrics: pd.DataFrame) -> None:
    level_cols = ["level1_acc_pct", "level2_acc_pct", "level3_acc_pct", "level4_acc_pct"]
    level_names = ["Level 1", "Level 2", "Level 3", "Level 4"]
    dataset_order = ["lt30", "30-50"]
    dataset_titles = {"lt30": "<30% identity", "30-50": "30-50% identity"}
    method_order = ["BLASTp_strict", "GaCA"]
    colors = {"BLASTp_strict": "#4C78A8", "GaCA": "#F58518"}

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    x = np.arange(len(level_cols))
    width = 0.34
    for ax, dataset in zip(axes, dataset_order):
        sub = metrics[metrics["dataset"] == dataset].set_index("method")
        for offset, method in zip([-width / 2, width / 2], method_order):
            vals = [sub.loc[method, col] for col in level_cols]
            label = "BLASTp" if method == "BLASTp_strict" else "GaCA"
            ax.bar(
                x + offset,
                vals,
                width=width,
                label=label,
                color=colors[method],
                edgecolor="white",
                linewidth=0.8,
            )
            for xi, yi in zip(x + offset, vals):
                ax.text(xi, yi + 1.0, f"{yi:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(dataset_titles[dataset], fontsize=11, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(level_names, fontsize=9)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=9)
    axes[0].set_ylabel("Accuracy (%)", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        fontsize=10,
        handlelength=1.8,
        columnspacing=1.6,
    )
    fig.suptitle("Strict closed-set hierarchical EC accuracy", fontsize=12, y=1.04)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    fig.savefig(OUT / "Fig_Main_Results_strict.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig_Main_Results_strict.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ensure_out()
    train_counts = parse_effective_train_counts(TRAIN_FASTA_EFFECTIVE)

    aligned: Dict[str, pd.DataFrame] = {}
    for dataset in DATASETS:
        df, _ = aligned_records(dataset, train_counts)
        aligned[dataset] = df
        df.to_csv(OUT / f"strict_closed_set_predictions_{dataset}.csv", index=False, encoding="utf-8")

    metrics = make_metrics(aligned)
    paired = make_paired(aligned)
    long_tail = make_long_tail(aligned)
    per_class = make_per_class(aligned)
    cases = choose_case_rows(aligned)
    accounting = make_dataset_accounting(aligned, train_counts)
    make_main_results_figure(metrics)

    metrics.to_csv(OUT / "strict_protocol_metrics.csv", index=False, encoding="utf-8")
    paired.to_csv(OUT / "paired_bootstrap_tests.csv", index=False, encoding="utf-8")
    long_tail.to_csv(OUT / "long_tail_analysis.csv", index=False, encoding="utf-8")
    per_class.to_csv(OUT / "per_class_performance.csv", index=False, encoding="utf-8")
    cases.to_csv(OUT / "case_study_candidates.csv", index=False, encoding="utf-8")
    accounting.to_csv(OUT / "dataset_accounting.csv", index=False, encoding="utf-8")

    write_latex(metrics, paired, long_tail, cases)
    write_manifest()

    summary = {
        "bootstrap_seed": BOOTSTRAP_SEED,
        "n_bootstrap": N_BOOT,
        "effective_train_proteins_from_train_strict_fasta": int(sum(train_counts.values())),
        "effective_train_ec_classes": int(len(train_counts)),
        "strict_protocol_metrics": metrics.to_dict(orient="records"),
        "paired_tests": paired.to_dict(orient="records"),
        "output_dir": str(OUT),
    }
    (OUT / "summary_p0_p4.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote P0-P4 outputs to: {OUT}")
    print(metrics.to_string(index=False))
    print("\nPaired tests:")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
