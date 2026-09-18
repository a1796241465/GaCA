from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, matthews_corrcoef

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


OUT = Path(r"D:\EC\new_str\Modify\journal_tbc_outputs")
DATASETS = {
    "lt30": {
        "label": "<30%",
        "strict": OUT / "strict_closed_set_predictions_lt30.csv",
        "foldseek": OUT / "foldseek_strict_predictions_lt30.csv",
    },
    "30-50": {
        "label": "30-50%",
        "strict": OUT / "strict_closed_set_predictions_30-50.csv",
        "foldseek": OUT / "foldseek_strict_predictions_30-50.csv",
    },
}
METHOD_COLUMNS = {
    "BLASTp": "blast_pred_ec",
    "Foldseek": "foldseek_pred_ec",
    "GaCA": "gaca_pred_ec",
}
METHOD_COLORS = {
    "BLASTp": "#6B7280",
    "Foldseek": "#4C78A8",
    "GaCA": "#F58518",
}
PLOT_LEVEL4_RESULTS = {
    "lt30": {
        "KNN": 48.56,
        "BLASTp": 66.67,
        "Foldseek": 69.96,
        "SVM": 34.98,
        "Random Forest": 57.61,
        "LightGBM": 56.38,
        "ESM-2 + MLP": 60.91,
        "GaCA": 69.96,
    },
    "30-50": {
        "KNN": 57.02,
        "BLASTp": 81.13,
        "Foldseek": 81.34,
        "SVM": 38.36,
        "Random Forest": 60.80,
        "LightGBM": 60.80,
        "ESM-2 + MLP": 64.15,
        "GaCA": 84.07,
    },
}
PLOT_METHOD_ORDER = [
    "KNN",
    "BLASTp",
    "Foldseek",
    "SVM",
    "Random Forest",
    "LightGBM",
    "ESM-2 + MLP",
    "GaCA",
]
PLOT_COLORS = {
    "KNN": "#B8C1CC",
    "SVM": "#B8C1CC",
    "Random Forest": "#A8B5C2",
    "LightGBM": "#A8B5C2",
    "ESM-2 + MLP": "#6AA5A9",
    "BLASTp": "#6B7280",
    "Foldseek": "#4C78A8",
    "GaCA": "#F58518",
}
RNG = np.random.default_rng(42)


def ec_prefix(ec: object, level: int) -> str:
    parts = str(ec).split(".")
    return ".".join(parts[:level]) if len(parts) >= level else str(ec)


def level_correct(true: pd.Series, pred: pd.Series, level: int) -> np.ndarray:
    return (
        true.astype(str).map(lambda x: ec_prefix(x, level)).to_numpy()
        == pred.astype(str).map(lambda x: ec_prefix(x, level)).to_numpy()
    )


def load_joined_predictions(dataset: str, cfg: dict) -> pd.DataFrame:
    strict = pd.read_csv(cfg["strict"])
    foldseek = pd.read_csv(cfg["foldseek"])
    foldseek_cols = [
        "id",
        "foldseek_target_id",
        "foldseek_pred_ec",
        "foldseek_bits",
        "foldseek_alntmscore",
        "foldseek_qtmscore",
        "foldseek_ttmscore",
    ]
    joined = strict.merge(foldseek[foldseek_cols], on="id", how="left", validate="one_to_one")
    if joined["foldseek_pred_ec"].isna().any():
        missing = joined.loc[joined["foldseek_pred_ec"].isna(), "id"].head().tolist()
        raise RuntimeError(f"{dataset}: missing Foldseek predictions for {missing}")
    return joined


def compute_metrics(joined_by_dataset: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []
    for dataset, df in joined_by_dataset.items():
        label = DATASETS[dataset]["label"]
        true = df["true_ec"].astype(str)
        labels = sorted(set(true))
        for method, pred_col in METHOD_COLUMNS.items():
            pred = df[pred_col].astype(str)
            row = {
                "dataset": dataset,
                "dataset_label": label,
                "method": method,
                "n": len(df),
            }
            for level in [1, 2, 3, 4]:
                row[f"level{level}_acc_pct"] = float(level_correct(true, pred, level).mean() * 100)
            row["macro_f1"] = float(f1_score(true, pred, labels=labels, average="macro", zero_division=0))
            row["mcc"] = float(matthews_corrcoef(true, pred))
            row["level4_correct"] = int((true == pred).sum())
            rows.append(row)
    return pd.DataFrame(rows)


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    try:
        df.to_csv(path, index=False)
    except PermissionError:
        print(f"Skipped writing locked file: {path}")


def bootstrap_ci(values: np.ndarray, n_resamples: int = 5000) -> tuple[float, float]:
    n = len(values)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = RNG.integers(0, n, n)
        means[i] = values[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def macro_f1_diff_sample(true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, idx: np.ndarray) -> float:
    labels = sorted(set(true[idx]))
    return float(
        f1_score(true[idx], pred_a[idx], labels=labels, average="macro", zero_division=0)
        - f1_score(true[idx], pred_b[idx], labels=labels, average="macro", zero_division=0)
    )


def paired_tests(joined_by_dataset: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: List[dict] = []
    comparisons = [("GaCA", "Foldseek", "gaca_pred_ec", "foldseek_pred_ec")]
    for dataset, df in joined_by_dataset.items():
        label = DATASETS[dataset]["label"]
        true = df["true_ec"].astype(str).to_numpy()
        for method_a, method_b, col_a, col_b in comparisons:
            pred_a = df[col_a].astype(str).to_numpy()
            pred_b = df[col_b].astype(str).to_numpy()
            correct_a = pred_a == true
            correct_b = pred_b == true
            diff = correct_a.astype(float) - correct_b.astype(float)
            ci_low, ci_high = bootstrap_ci(diff)
            a_only = int(np.sum(correct_a & ~correct_b))
            b_only = int(np.sum(correct_b & ~correct_a))
            discordant = a_only + b_only
            mcnemar_p = float(binomtest(min(a_only, b_only), discordant, 0.5).pvalue) if discordant else 1.0
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": label,
                    "comparison": f"{method_a} minus {method_b}",
                    "metric": "level4_accuracy",
                    "method_a": method_a,
                    "method_b": method_b,
                    "method_a_value": float(correct_a.mean()),
                    "method_b_value": float(correct_b.mean()),
                    "difference": float(diff.mean()),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "mcnemar_exact_p": mcnemar_p,
                    "bootstrap_p_two_sided": np.nan,
                    "paired_p": mcnemar_p,
                    "paired_test": "exact two-sided McNemar",
                    "a_only_correct": a_only,
                    "b_only_correct": b_only,
                    "n": len(df),
                }
            )

            n = len(df)
            f1_diffs = np.empty(5000)
            for i in range(5000):
                idx = RNG.integers(0, n, n)
                f1_diffs[i] = macro_f1_diff_sample(true, pred_a, pred_b, idx)
            labels = sorted(set(true))
            f1_a = f1_score(true, pred_a, labels=labels, average="macro", zero_division=0)
            f1_b = f1_score(true, pred_b, labels=labels, average="macro", zero_division=0)
            p_boot = min(
                1.0,
                2 * min(float(np.mean(f1_diffs <= 0)), float(np.mean(f1_diffs >= 0))),
            )
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": label,
                    "comparison": f"{method_a} minus {method_b}",
                    "metric": "macro_f1",
                    "method_a": method_a,
                    "method_b": method_b,
                    "method_a_value": float(f1_a),
                    "method_b_value": float(f1_b),
                    "difference": float(f1_a - f1_b),
                    "ci95_low": float(np.percentile(f1_diffs, 2.5)),
                    "ci95_high": float(np.percentile(f1_diffs, 97.5)),
                    "mcnemar_exact_p": np.nan,
                    "bootstrap_p_two_sided": p_boot,
                    "paired_p": p_boot,
                    "paired_test": "two-sided paired bootstrap",
                    "a_only_correct": np.nan,
                    "b_only_correct": np.nan,
                    "n": len(df),
                }
            )
    return pd.DataFrame(rows)


def plot_main_results(metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.8), sharex=True, sharey=True)
    y = np.arange(len(PLOT_METHOD_ORDER))

    for ax, dataset in zip(axes, ["lt30", "30-50"]):
        values = [PLOT_LEVEL4_RESULTS[dataset][method] for method in PLOT_METHOD_ORDER]
        colors = [PLOT_COLORS[method] for method in PLOT_METHOD_ORDER]
        edges = ["#111111" if method == "GaCA" else "white" for method in PLOT_METHOD_ORDER]
        widths = [0.9 if method == "GaCA" else 0.6 for method in PLOT_METHOD_ORDER]
        bars = ax.barh(y, values, color=colors, edgecolor=edges, linewidth=widths)
        for bar, val in zip(bars, values):
            ax.text(
                val + 1.0,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}",
                va="center",
                ha="left",
                fontsize=8.5,
            )

        n = int(metrics[metrics["dataset"] == dataset]["n"].iloc[0])
        ax.set_title(f"{DATASETS[dataset]['label']} identity (n={n})", fontsize=11)
        ax.set_xlabel("Level-4 accuracy (%)")
        ax.set_xlim(0, 92)
        ax.set_yticks(y)
        ax.set_yticklabels(PLOT_METHOD_ORDER, fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[1].tick_params(labelleft=False)
    fig.suptitle("Strict closed-set Level-4 EC accuracy across all evaluated methods", fontsize=13, y=1.02)
    fig.tight_layout()
    try:
        fig.savefig(OUT / "Fig_Main_Results_strict.pdf", bbox_inches="tight")
        fig.savefig(OUT / "Fig_Main_Results_strict.png", dpi=300, bbox_inches="tight")
    except PermissionError:
        fig.savefig(OUT / "Fig_Main_Results_strict_allmethods.pdf", bbox_inches="tight")
        fig.savefig(OUT / "Fig_Main_Results_strict_allmethods.png", dpi=300, bbox_inches="tight")
        print("Main figure was locked; wrote Fig_Main_Results_strict_allmethods.pdf/png instead.")
    plt.close(fig)


def main() -> None:
    joined = {dataset: load_joined_predictions(dataset, cfg) for dataset, cfg in DATASETS.items()}
    for dataset, df in joined.items():
        safe_to_csv(df, OUT / f"strict_predictions_with_foldseek_{dataset}.csv")
    metrics = compute_metrics(joined)
    safe_to_csv(metrics, OUT / "strict_protocol_metrics_with_foldseek.csv")
    tests = paired_tests(joined)
    safe_to_csv(tests, OUT / "gaca_vs_foldseek_paired_tests.csv")
    plot_main_results(metrics)
    print(metrics.to_string(index=False))
    print("\nPaired tests:")
    print(tests.to_string(index=False))


if __name__ == "__main__":
    main()
