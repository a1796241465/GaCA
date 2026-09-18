"""
Fill remaining manuscript TBC items:
  - Supplementary Table S5: hyperparameters, seeds, package versions, hardware.
  - Supplementary Figure S2: GaCA accuracy-coverage and confidence calibration.

All values are derived from existing source files, prediction JSON files,
probability arrays, and the current execution environment. Unknown original
training details are reported as "not recorded in code" rather than inferred.
"""

from __future__ import annotations

import csv
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(r"D:/EC")
MODIFY = ROOT / "new_str" / "Modify"
ANALYSIS = MODIFY / "analysis_outputs"
OUT = MODIFY / "journal_tbc_outputs"
GACA_SOURCE = MODIFY / "GaCA" / "GaCA.py"
COMPREHENSIVE_SOURCE = MODIFY / "comprehensive_analysis.py"

DATASETS = {
    "lt30": {
        "label": "<30%",
        "json": ANALYSIS / "predictions_lt30.json",
        "probs": ANALYSIS / "predictions_lt30_probs.npy",
    },
    "30-50": {
        "label": "30-50%",
        "json": ANALYSIS / "predictions_30-50.json",
        "probs": ANALYSIS / "predictions_30-50_probs.npy",
    },
}

N_BINS = 10


def ensure_out() -> None:
    OUT.mkdir(parents=True, exist_ok=True)


def read_source() -> str:
    return GACA_SOURCE.read_text(encoding="utf-8", errors="replace")


def extract_constant(source: str, name: str) -> str:
    pattern = rf"^\s*{re.escape(name)}\s*=\s*(.+?)\s*$"
    m = re.search(pattern, source, re.MULTILINE)
    return m.group(1).strip() if m else "not recorded in code"


def source_contains(source: str, pattern: str) -> bool:
    return re.search(pattern, source, re.MULTILINE) is not None


def package_version(module_name: str, import_name: str | None = None) -> str:
    try:
        mod = __import__(import_name or module_name)
        return str(getattr(mod, "__version__", "version not exposed"))
    except Exception as exc:
        return f"not available ({type(exc).__name__})"


def current_cuda_info() -> Dict[str, str]:
    info = {
        "cuda_available_current_environment": "not available",
        "cuda_device_count_current_environment": "not available",
        "cuda_device_0_current_environment": "not available",
    }
    try:
        import torch

        info["cuda_available_current_environment"] = str(torch.cuda.is_available())
        info["cuda_device_count_current_environment"] = str(torch.cuda.device_count())
        if torch.cuda.is_available():
            info["cuda_device_0_current_environment"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        info["cuda_available_current_environment"] = f"not available ({type(exc).__name__})"
    return info


def build_table_s5_rows() -> List[Dict[str, str]]:
    source = read_source()
    cuda = current_cuda_info()

    rows = []

    def add(category: str, item: str, value: str, source_note: str) -> None:
        rows.append({"category": category, "item": item, "value": value, "source": source_note})

    add("Model", "sequence embedding dimension", "1280", "GaCA.py tensor reshape/model definition")
    add("Model", "structure embedding dimension", "512", "GaCA.py tensor reshape/model definition")
    add("Model", "hidden dimension", extract_constant(source, "HIDDEN_DIM"), "GaCA.py")
    add("Model", "dropout", extract_constant(source, "DROPOUT"), "GaCA.py")
    add("Model", "sequence pooling", "attention pooling", "GaCA.py AttentionPooling")
    add("Model", "structure pooling", "attention pooling", "GaCA.py AttentionPooling")
    add("Model", "fusion module", "vector-wise gating", "GaCA.py GaCA_Final")
    add("Model", "classifier head", "Linear(512) + BatchNorm1d + GELU + Dropout + Linear(num_labels)", "GaCA.py")
    add("Training", "optimizer", "AdamW" if "optim.AdamW" in source else "not recorded in code", "GaCA.py")
    add("Training", "learning rate", extract_constant(source, "LEARNING_RATE"), "GaCA.py")
    add("Training", "weight decay", extract_constant(source, "WEIGHT_DECAY"), "GaCA.py")
    add("Training", "batch size", extract_constant(source, "BATCH_SIZE"), "GaCA.py")
    add("Training", "maximum epochs", extract_constant(source, "EPOCHS"), "GaCA.py")
    add("Training", "early stopping patience", extract_constant(source, "PATIENCE"), "GaCA.py")
    add("Training", "validation ratio", extract_constant(source, "VAL_RATIO"), "GaCA.py")
    add("Training", "loss", "CrossEntropyLoss with label smoothing" if "CrossEntropyLoss" in source else "not recorded in code", "GaCA.py")
    add("Training", "label smoothing", extract_constant(source, "LABEL_SMOOTHING"), "GaCA.py")
    add("Training", "data split seed", extract_constant(source, "SEED"), "GaCA.py")
    add("Training", "deterministic cuDNN", "True" if "torch.backends.cudnn.deterministic = True" in source else "not recorded in code", "GaCA.py")
    add("Training", "cuDNN benchmark", "False" if "torch.backends.cudnn.benchmark = False" in source else "not recorded in code", "GaCA.py")
    add("Training", "DataLoader workers", "4", "GaCA.py DataLoader calls")
    add("Evaluation", "closed-set filtering", "test EC4 labels absent from training labels are excluded", "GaCA.py/GaCADataset")
    add("Evaluation", "bootstrap seed for P0-P4 tables", "42", "fill_tbc_p0_p4.py")
    add("Evaluation", "bootstrap resamples for paired CIs", "5000", "fill_tbc_p0_p4.py")

    add("Environment", "python", sys.version.replace("\n", " "), "current execution environment")
    add("Environment", "platform", platform.platform(), "current execution environment")
    add("Environment", "CPU logical cores", str(os.cpu_count()), "current execution environment")
    add("Environment", "processor", platform.processor() or "not exposed by platform.processor()", "current execution environment")
    add("Environment", "torch", package_version("torch"), "current execution environment")
    add("Environment", "numpy", package_version("numpy"), "current execution environment")
    add("Environment", "pandas", package_version("pandas"), "current execution environment")
    add("Environment", "scikit-learn", package_version("sklearn"), "current execution environment")
    add("Environment", "matplotlib", package_version("matplotlib"), "current execution environment")
    add("Environment", "scipy", package_version("scipy"), "current execution environment")
    add("Environment", "CUDA_VISIBLE_DEVICES set in training code", extract_constant(source, 'os.environ["CUDA_VISIBLE_DEVICES"]'), "GaCA.py")
    add("Environment", "CUDA available", cuda["cuda_available_current_environment"], "current execution environment")
    add("Environment", "CUDA device count", cuda["cuda_device_count_current_environment"], "current execution environment")
    add("Environment", "CUDA device 0", cuda["cuda_device_0_current_environment"], "current execution environment")
    add("Environment", "original training hardware", "not independently logged in the code/output files", "conservative reporting")

    return rows


def load_prediction(dataset_key: str) -> pd.DataFrame:
    cfg = DATASETS[dataset_key]
    data = json.loads(cfg["json"].read_text(encoding="utf-8"))
    probs = np.load(cfg["probs"])
    if len(data["ids"]) != probs.shape[0]:
        raise RuntimeError(f"Probability row mismatch for {dataset_key}")
    confidence = probs.max(axis=1)
    true_idx = np.asarray(data["true_idx"], dtype=int)
    pred_idx = np.asarray(data["pred_idx"], dtype=int)
    true_confidence = probs[np.arange(len(true_idx)), true_idx]
    df = pd.DataFrame(
        {
            "dataset": dataset_key,
            "dataset_label": DATASETS[dataset_key]["label"],
            "id": data["ids"],
            "true_ec": data["true_ec"],
            "pred_ec": data["pred_ec"],
            "true_idx": true_idx,
            "pred_idx": pred_idx,
            "confidence": confidence,
            "true_confidence": true_confidence,
            "correct": (true_idx == pred_idx).astype(int),
        }
    )
    return df


def build_coverage_curve(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values("confidence", ascending=False).reset_index(drop=True)
    n = len(ordered)
    coverages = [0.05] + [x / 10 for x in range(1, 11)]
    rows = []
    for cov in coverages:
        k = max(1, int(np.ceil(cov * n)))
        sub = ordered.iloc[:k]
        rows.append(
            {
                "dataset": df["dataset"].iloc[0],
                "dataset_label": df["dataset_label"].iloc[0],
                "coverage": k / n,
                "retained_samples": int(k),
                "accuracy": float(sub["correct"].mean()),
                "mean_confidence": float(sub["confidence"].mean()),
                "min_confidence_threshold": float(sub["confidence"].min()),
            }
        )
    return pd.DataFrame(rows)


def build_calibration_bins(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for i in range(N_BINS):
        low = i / N_BINS
        high = (i + 1) / N_BINS
        if i == N_BINS - 1:
            mask = (df["confidence"] >= low) & (df["confidence"] <= high)
        else:
            mask = (df["confidence"] >= low) & (df["confidence"] < high)
        sub = df[mask]
        rows.append(
            {
                "dataset": df["dataset"].iloc[0],
                "dataset_label": df["dataset_label"].iloc[0],
                "bin_low": low,
                "bin_high": high,
                "bin_mid": (low + high) / 2,
                "samples": int(len(sub)),
                "fraction": float(len(sub) / n),
                "accuracy": float(sub["correct"].mean()) if len(sub) else np.nan,
                "mean_confidence": float(sub["confidence"].mean()) if len(sub) else np.nan,
                "abs_gap": float(abs(sub["correct"].mean() - sub["confidence"].mean())) if len(sub) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def calibration_summary(df: pd.DataFrame, bins: pd.DataFrame, coverage: pd.DataFrame) -> Dict[str, float | str | int]:
    nonempty = bins.dropna(subset=["abs_gap"])
    ece = float((nonempty["fraction"] * nonempty["abs_gap"]).sum())
    mce = float(nonempty["abs_gap"].max()) if len(nonempty) else float("nan")
    high_conf_50 = coverage.iloc[(coverage["coverage"] - 0.5).abs().argsort()[:1]].iloc[0]
    high_conf_20 = coverage.iloc[(coverage["coverage"] - 0.2).abs().argsort()[:1]].iloc[0]
    return {
        "dataset": df["dataset"].iloc[0],
        "dataset_label": df["dataset_label"].iloc[0],
        "n": int(len(df)),
        "overall_accuracy": float(df["correct"].mean()),
        "mean_confidence": float(df["confidence"].mean()),
        "ece_10_bins": ece,
        "mce_10_bins": mce,
        "accuracy_at_50pct_coverage": float(high_conf_50["accuracy"]),
        "confidence_threshold_at_50pct_coverage": float(high_conf_50["min_confidence_threshold"]),
        "accuracy_at_20pct_coverage": float(high_conf_20["accuracy"]),
        "confidence_threshold_at_20pct_coverage": float(high_conf_20["min_confidence_threshold"]),
    }


def plot_calibration_coverage(coverage_all: pd.DataFrame, bins_all: pd.DataFrame) -> None:
    colors = {"lt30": "#4C78A8", "30-50": "#F58518"}
    titles = {"lt30": "<30% identity", "30-50": "30-50% identity"}
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.2))
    legend_handles = []
    legend_labels = []

    for col, dataset in enumerate(["lt30", "30-50"]):
        cov = coverage_all[coverage_all["dataset"] == dataset]
        bins = bins_all[bins_all["dataset"] == dataset]
        color = colors[dataset]

        ax = axes[0, col]
        ax.plot(cov["coverage"] * 100, cov["accuracy"] * 100, marker="o", color=color, linewidth=2)
        ax.axhline(cov[cov["coverage"] == 1.0]["accuracy"].iloc[0] * 100, color="#555555", linestyle="--", linewidth=1)
        panel = "(a)" if dataset == "lt30" else "(b)"
        ax.set_title(f"{panel} {titles[dataset]}: accuracy-coverage", fontsize=11)
        ax.set_xlabel("Coverage retained (%)")
        ax.set_ylabel("Level-4 accuracy (%)")
        ax.set_xlim(0, 102)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)

        ax = axes[1, col]
        nonempty = bins.dropna(subset=["accuracy", "mean_confidence"])
        ax.bar(
            nonempty["bin_mid"],
            nonempty["accuracy"],
            width=0.085,
            color=color,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.8,
            label="Observed accuracy",
        )
        ax.plot([0, 1], [0, 1], color="#333333", linestyle="--", linewidth=1.2, label="Perfect calibration")
        ax.scatter(nonempty["mean_confidence"], nonempty["accuracy"], color="#111111", s=20, zorder=3)
        panel = "(c)" if dataset == "lt30" else "(d)"
        ax.set_title(f"{panel} {titles[dataset]}: reliability diagram", fontsize=11)
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        if col == 1:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    fig.suptitle("GaCA confidence, coverage, and calibration", fontsize=13, y=1.01)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.01),
            ncol=2,
            fontsize=8,
        )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(OUT / "Fig_S2_Calibration_Coverage.pdf", bbox_inches="tight")
    fig.savefig(OUT / "Fig_S2_Calibration_Coverage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def latex_escape(text: object) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def write_table_s5_tex(rows: List[Dict[str, str]]) -> None:
    lines = []
    lines.append("% Auto-generated by fill_remaining_tbc_s5_calibration.py. Review before submission.")
    lines.append("\\begin{table*}[h]")
    lines.append("\\centering")
    lines.append("\\caption*{Table S5. Reproducibility configuration. Values marked as current environment were detected when generating this table; original training hardware was not independently logged unless explicitly stated.}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llll}")
    lines.append("\\toprule")
    lines.append("Category & Item & Value & Source \\\\")
    lines.append("\\midrule")
    for r in rows:
        category = latex_escape(r["category"])
        item = latex_escape(r["item"])
        value = latex_escape(r["value"])
        source = latex_escape(r["source"])
        lines.append(f"{category} & {item} & {value} & {source} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}}")
    lines.append("\\end{table*}")
    lines.append("")
    lines.append("\\begin{figure*}[h]")
    lines.append("\\centering")
    lines.append("\\includegraphics[width=0.90\\textwidth]{Fig_S2_Calibration_Coverage.pdf}")
    lines.append("\\caption*{Figure S2. GaCA accuracy-coverage and confidence calibration curves computed from saved prediction probabilities. Panels (a) and (b) show accuracy-coverage curves for the <30\\% and 30--50\\% subsets, respectively. Panels (c) and (d) show 10-bin reliability diagrams for the same subsets.}")
    lines.append("\\end{figure*}")
    (OUT / "supplementary_s5_and_fig_s2_snippets.tex").write_text("\n".join(lines), encoding="utf-8")


def update_manifest_entries() -> None:
    manifest = OUT / "manifest.csv"
    existing = []
    if manifest.exists():
        existing = list(csv.reader(manifest.open("r", encoding="utf-8")))
    header = ["file", "description"]
    rows = existing[1:] if existing and existing[0] == header else existing
    additions = {
        "hyperparameters_environment_s5.csv": "Supplementary Table S5 source: hyperparameters, seeds, package versions, hardware notes.",
        "coverage_curve.csv": "GaCA accuracy-coverage curve values for Figure S2.",
        "calibration_bins.csv": "GaCA reliability-bin values for Figure S2.",
        "calibration_coverage_summary.csv": "Summary ECE, MCE, and accuracy-at-coverage values.",
        "calibration_coverage_summary.json": "Machine-readable calibration and coverage summary.",
        "Fig_S2_Calibration_Coverage.pdf": "Supplementary Figure S2 as PDF.",
        "Fig_S2_Calibration_Coverage.png": "Supplementary Figure S2 as PNG.",
        "supplementary_s5_and_fig_s2_snippets.tex": "LaTeX snippets for Supplementary Table S5 and Figure S2.",
        "fill_remaining_tbc_s5_calibration.py": "Reproducible script for remaining TBC outputs.",
    }
    row_map = {r[0]: r[1] for r in rows if len(r) >= 2}
    row_map.update(additions)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for file_name in sorted(row_map):
            writer.writerow([file_name, row_map[file_name]])


def main() -> None:
    ensure_out()

    s5_rows = build_table_s5_rows()
    pd.DataFrame(s5_rows).to_csv(OUT / "hyperparameters_environment_s5.csv", index=False, encoding="utf-8")

    coverage_frames = []
    bin_frames = []
    summaries = []
    aligned_predictions = []
    for dataset in DATASETS:
        df = load_prediction(dataset)
        aligned_predictions.append(df)
        coverage = build_coverage_curve(df)
        bins = build_calibration_bins(df)
        coverage_frames.append(coverage)
        bin_frames.append(bins)
        summaries.append(calibration_summary(df, bins, coverage))

    coverage_all = pd.concat(coverage_frames, ignore_index=True)
    bins_all = pd.concat(bin_frames, ignore_index=True)
    pred_all = pd.concat(aligned_predictions, ignore_index=True)
    summary_df = pd.DataFrame(summaries)

    pred_all.to_csv(OUT / "gaca_confidence_predictions.csv", index=False, encoding="utf-8")
    coverage_all.to_csv(OUT / "coverage_curve.csv", index=False, encoding="utf-8")
    bins_all.to_csv(OUT / "calibration_bins.csv", index=False, encoding="utf-8")
    summary_df.to_csv(OUT / "calibration_coverage_summary.csv", index=False, encoding="utf-8")
    (OUT / "calibration_coverage_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    plot_calibration_coverage(coverage_all, bins_all)
    write_table_s5_tex(s5_rows)
    update_manifest_entries()

    print(f"Wrote remaining TBC outputs to: {OUT}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
