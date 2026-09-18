"""Generate manuscript Figures 2-6 from the final unified GaCA rerun."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pointbiserialr
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve


DATASET_LABELS = {"lt30": "<30% identity", "30_50": "30-50% identity"}
DATASET_COLORS = {"lt30": "#4C78A8", "30_50": "#F58518"}
PR_COLOR = "#2F80ED"
GATE_COLOR = "#E74C3C"
MAIN_METHODS = [
    "KNN",
    "BLASTp",
    "Foldseek",
    "SVM",
    "Random Forest",
    "LightGBM",
    "ESM-2 sequence only",
    "GaCA",
]
METHOD_LABELS = {
    "ESM-2 sequence only": "ESM-2 + MLP",
    "GaCA": "GaCA (Ours)",
}
METHOD_COLORS = {
    "GaCA": "#F58518",
    "Foldseek": "#4C78A8",
    "BLASTp": "#737B88",
    "ESM-2 sequence only": "#59A14F",
}


def configure_style():
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def save_figure(figure, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def load_predictions(path):
    frame = pd.read_csv(path, dtype={"protein_id": str})
    if "level_4_correct" not in frame:
        frame["level_4_correct"] = frame["true_ec"].astype(str) == frame[
            "predicted_ec"
        ].astype(str)
    frame["level_4_correct"] = frame["level_4_correct"].astype(int)
    return frame


def plot_main_results(metrics, output):
    frame = metrics.loc[metrics["method"].isin(MAIN_METHODS)].copy()
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 5.0), sharey=True)
    values_for_audit = {}
    for axis, dataset in zip(axes, ("lt30", "30_50")):
        subset = frame.loc[frame["dataset"] == dataset].set_index("method")
        subset = subset.reindex(MAIN_METHODS)
        values = 100.0 * subset["level_4_accuracy"].astype(float)
        labels = [METHOD_LABELS.get(method, method) for method in MAIN_METHODS]
        colors = [METHOD_COLORS.get(method, "#A6ACB5") for method in MAIN_METHODS]
        bars = axis.barh(labels, values, color=colors, edgecolor="white", linewidth=0.5)
        axis.invert_yaxis()
        axis.set_title(DATASET_LABELS[dataset])
        axis.set_xlabel("Level-4 accuracy (%)")
        axis.set_xlim(0, 100)
        axis.grid(axis="x", linestyle=":", alpha=0.4)
        axis.grid(axis="y", visible=False)
        for bar, value in zip(bars, values):
            axis.text(
                value + 0.7,
                bar.get_y() + bar.get_height() / 2,
                f"{value:.2f}",
                va="center",
                fontsize=8,
            )
        values_for_audit[dataset] = {
            method: round(float(value), 6)
            for method, value in zip(MAIN_METHODS, values)
        }
    figure.suptitle("Level-4 EC accuracy across the methods in Table 2", y=1.01)
    figure.tight_layout()
    save_figure(figure, output / "Fig_Main_Results.pdf")
    return values_for_audit


def macro_pr_curve(npz_path):
    data = np.load(npz_path, allow_pickle=False)
    y_true = data["true_indices"].astype(int)
    probabilities = data["probabilities"].astype(float)
    precisions = []
    recalls = []
    average_precisions = []
    represented = np.unique(y_true)
    for class_index in represented:
        binary = (y_true == class_index).astype(int)
        precision, recall, _ = precision_recall_curve(
            binary, probabilities[:, class_index]
        )
        precisions.append(precision)
        recalls.append(recall)
        average_precisions.append(
            average_precision_score(binary, probabilities[:, class_index])
        )
    recall_grid = np.sort(np.unique(np.concatenate(recalls)))[::-1]
    macro_precision = []
    for recall_value in recall_grid:
        envelope_values = [
            np.max(precision[recall >= recall_value])
            for precision, recall in zip(precisions, recalls)
            if np.any(recall >= recall_value)
        ]
        macro_precision.append(float(np.mean(envelope_values)))
    return recall_grid, np.asarray(macro_precision), float(np.mean(average_precisions)), len(represented)


def plot_precision_recall(artifact_dir, output):
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharex=True, sharey=True)
    statistics = {}
    for panel, axis, dataset in zip(("a", "b"), axes, ("lt30", "30_50")):
        recall, precision, macro_aupr, class_count = macro_pr_curve(
            artifact_dir / f"probabilities_{dataset}.npz"
        )
        axis.plot(recall, precision, color=PR_COLOR, linewidth=2)
        axis.fill_between(recall, precision, color=PR_COLOR, alpha=0.10)
        axis.set_title(
            f"({panel}) {DATASET_LABELS[dataset]}\nMacro-AUPR = {macro_aupr:.2f}"
        )
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(linestyle=":", alpha=0.4)
        statistics[dataset] = {
            "macro_aupr": macro_aupr,
            "represented_level4_classes": class_count,
        }
    figure.suptitle("Macro-averaged one-vs-rest precision-recall curves")
    figure.tight_layout()
    save_figure(figure, output / "precision_recall_curves.png")

    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), sharex=True, sharey=True)
    for panel, axis, dataset in zip(("a", "b"), axes, ("lt30", "30_50")):
        recall, precision, macro_aupr, _ = macro_pr_curve(
            artifact_dir / f"probabilities_{dataset}.npz"
        )
        axis.plot(recall, precision, color=PR_COLOR, linewidth=2)
        axis.fill_between(recall, precision, color=PR_COLOR, alpha=0.10)
        axis.set_title(
            f"({panel}) {DATASET_LABELS[dataset]}\nMacro-AUPR = {macro_aupr:.2f}"
        )
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.grid(linestyle=":", alpha=0.4)
    figure.suptitle("Macro-averaged one-vs-rest precision-recall curves")
    figure.tight_layout()
    save_figure(figure, output / "precision_recall_curves.pdf")
    return statistics


def score_statistics(frame):
    ranked = frame.sort_values("top_class_softmax_score", ascending=False).reset_index(drop=True)
    fractions = np.asarray([0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0])
    accuracies = []
    counts = []
    for fraction in fractions:
        count = max(1, int(np.ceil(fraction * len(ranked))))
        counts.append(count)
        accuracies.append(float(ranked.iloc[:count]["level_4_correct"].mean()))
    bins = pd.qcut(frame["top_class_softmax_score"], q=10, duplicates="drop")
    grouped = frame.assign(score_bin=bins).groupby("score_bin", observed=True).agg(
        mean_score=("top_class_softmax_score", "mean"),
        observed_accuracy=("level_4_correct", "mean"),
        proteins=("protein_id", "count"),
    )
    correlation = float(
        pointbiserialr(frame["level_4_correct"], frame["top_class_softmax_score"]).statistic
    )
    return ranked, fractions, np.asarray(accuracies), counts, grouped, correlation


def plot_softmax_scores(predictions, output):
    figure, axes = plt.subplots(2, 2, figsize=(10.4, 7.5))
    statistics = {}
    for column, dataset in enumerate(("lt30", "30_50")):
        frame = predictions[dataset]
        ranked, fractions, accuracies, counts, grouped, correlation = score_statistics(frame)
        top = axes[0, column]
        bottom = axes[1, column]
        top.plot(
            100 * fractions,
            100 * accuracies,
            marker="o",
            markersize=4,
            color=DATASET_COLORS[dataset],
            linewidth=1.8,
        )
        full_accuracy = 100 * float(ranked["level_4_correct"].mean())
        top.axhline(full_accuracy, color="#555555", linestyle="--", linewidth=1)
        top.set_title(f"({'ab'[column]}) {DATASET_LABELS[dataset]}: retained-set accuracy")
        top.set_xlabel("Highest-score predictions retained (%)")
        top.set_ylabel("Level-4 accuracy (%)")
        top.set_ylim(0, 105)
        top.grid(linestyle=":", alpha=0.4)
        top.legend(["Score-ranked subset", "All-protein accuracy"], loc="lower left")

        bottom.plot(
            grouped["mean_score"],
            grouped["observed_accuracy"],
            marker="o",
            markersize=4,
            color=DATASET_COLORS[dataset],
            linewidth=1.8,
        )
        bottom.axhline(
            full_accuracy / 100.0,
            color="#777777",
            linestyle="--",
            linewidth=1,
        )
        bottom.set_title(f"({'cd'[column]}) {DATASET_LABELS[dataset]}: score-decile accuracy")
        bottom.text(
            0.03,
            0.95,
            f"Point-biserial $r$ = {correlation:.2f}",
            transform=bottom.transAxes,
            ha="left",
            va="top",
        )
        bottom.set_xlabel("Mean top-class softmax score in decile")
        bottom.set_ylabel("Observed Level-4 accuracy")
        bottom.set_xlim(0, 1.02)
        bottom.set_ylim(0, 1.03)
        bottom.grid(linestyle=":", alpha=0.4)
        statistics[dataset] = {
            "samples": len(frame),
            "full_accuracy": full_accuracy / 100.0,
            "retained_50_count": counts[list(fractions).index(0.5)],
            "retained_50_accuracy": float(accuracies[list(fractions).index(0.5)]),
            "point_biserial_correlation": correlation,
            "score_bins": grouped.reset_index(drop=True).to_dict(orient="records"),
        }
    figure.suptitle("GaCA top-class softmax score and Level-4 correctness")
    figure.tight_layout()
    save_figure(figure, output / "Fig_S2_Softmax_Scores.pdf")
    return statistics


def confusion_data(frame, labels):
    true = frame["true_ec"].astype(str).str.split(".").str[0]
    predicted = frame["predicted_ec"].astype(str).str.split(".").str[0]
    counts = confusion_matrix(true, predicted, labels=labels)
    normalized = confusion_matrix(true, predicted, labels=labels, normalize="true")
    return counts, normalized


def draw_confusion(axis, normalized, labels, dataset):
    sns.heatmap(
        normalized,
        ax=axis,
        cmap="Blues",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        annot_kws={"fontsize": 8},
        xticklabels=labels,
        yticklabels=labels,
        square=True,
        cbar_kws={"label": "Row proportion"},
    )
    axis.set_title(DATASET_LABELS[dataset])
    axis.set_xlabel("Predicted Level-1 class")
    axis.set_ylabel("Reference Level-1 class")


def plot_confusion_matrices(predictions, output):
    labels = sorted(
        {
            str(ec).split(".")[0]
            for frame in predictions.values()
            for column in ("true_ec", "predicted_ec")
            for ec in frame[column]
        },
        key=int,
    )
    statistics = {}
    combined, axes = plt.subplots(1, 2, figsize=(10.7, 4.6))
    for axis, dataset in zip(axes, ("lt30", "30_50")):
        counts, normalized = confusion_data(predictions[dataset], labels)
        draw_confusion(axis, normalized, labels, dataset)
        statistics[dataset] = {
            "labels": labels,
            "counts": counts.tolist(),
            "row_normalized": normalized.tolist(),
        }
        panel, panel_axis = plt.subplots(figsize=(5.5, 4.6))
        draw_confusion(panel_axis, normalized, labels, dataset)
        panel.tight_layout()
        filename = "Fig_30_CM.pdf" if dataset == "lt30" else "Fig_30-50_CM.pdf"
        save_figure(panel, output / filename)
    combined.suptitle("Row-normalized Level-1 confusion matrices")
    combined.tight_layout()
    save_figure(combined, output / "Fig_Level1_Confusion.pdf")
    return statistics


def draw_gates(axis, gates, dataset, upper):
    sns.histplot(
        gates,
        kde=True,
        bins="auto",
        ax=axis,
        color=GATE_COLOR,
        edgecolor="white",
        alpha=0.70,
        line_kws={"linewidth": 2},
    )
    subset = "<30%" if dataset == "lt30" else "30-50%"
    axis.set_title(
        f"Distribution of Mean Gate Values ({subset})",
        fontweight="bold",
        fontsize=14,
        pad=15,
    )
    axis.set_xlabel(
        "Mean gate value for the structure branch", fontweight="bold", fontsize=12
    )
    axis.set_ylabel("Frequency", fontweight="bold", fontsize=12)
    axis.set_xlim(0, upper)
    axis.grid(True, linestyle=":", alpha=0.6)


def plot_gate_distributions(artifact_dir, output):
    gates = {
        dataset: np.load(artifact_dir / f"gate_vectors_{dataset}.npy").mean(axis=1)
        for dataset in ("lt30", "30_50")
    }

    statistics = {}
    combined, axes = plt.subplots(1, 2, figsize=(10.8, 4.5))
    for axis, dataset in zip(axes, ("lt30", "30_50")):
        values = gates[dataset]
        upper = min(1.0, float(np.max(values)) * 1.15)
        draw_gates(axis, values, dataset, upper)
        statistics[dataset] = {
            "samples": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "standard_deviation": float(np.std(values, ddof=1)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "above_0_5": int(np.sum(values > 0.5)),
        }
        panel, panel_axis = plt.subplots(figsize=(7.0, 5.0))
        draw_gates(panel_axis, values, dataset, upper)
        panel.tight_layout()
        filename = "Fig_30_Gating.pdf" if dataset == "lt30" else "Fig_30-50_Gating.pdf"
        save_figure(panel, output / filename)
    combined.suptitle("Distribution of per-protein mean gate values for the structure branch")
    combined.tight_layout()
    save_figure(combined, output / "Fig_Gating_Distributions.pdf")
    return statistics


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--gaca-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.metrics)
    predictions = {
        dataset: load_predictions(args.gaca_results / f"predictions_{dataset}.csv")
        for dataset in ("lt30", "30_50")
    }
    statistics = {
        "figure_2": plot_main_results(metrics, args.output_dir),
        "figure_3": plot_precision_recall(args.gaca_results, args.output_dir),
        "figure_4": plot_softmax_scores(predictions, args.output_dir),
        "figure_5": plot_confusion_matrices(predictions, args.output_dir),
        "figure_6": plot_gate_distributions(args.gaca_results, args.output_dir),
    }
    (args.output_dir / "updated_figure_statistics.json").write_text(
        json.dumps(statistics, indent=2), encoding="utf-8"
    )
    print(f"[done] manuscript figures written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
