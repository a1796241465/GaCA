from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, matthews_corrcoef


N_BOOT = 5000
SEED = 42


def level_correct(true_ec: str, pred_ec: str, level: int) -> int:
    true_parts = str(true_ec).split(".")
    pred_parts = str(pred_ec).split(".")
    return int(len(true_parts) >= level and len(pred_parts) >= level and true_parts[:level] == pred_parts[:level])


def summary(true: np.ndarray, pred: np.ndarray) -> dict:
    levels = np.asarray(
        [[level_correct(t, p, level) for level in range(1, 5)] for t, p in zip(true, pred)],
        dtype=float,
    )
    labels = sorted(set(true))
    return {
        "n": int(len(true)),
        "level_1_accuracy": float(levels[:, 0].mean()),
        "level_2_accuracy": float(levels[:, 1].mean()),
        "level_3_accuracy": float(levels[:, 2].mean()),
        "level_4_accuracy": float(levels[:, 3].mean()),
        "macro_f1": float(f1_score(true, pred, labels=labels, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(true, pred)),
    }


def paired_bootstrap(
    true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: str,
    rng: np.random.Generator,
) -> tuple[float, float, float, float]:
    n = len(true)
    if metric == "level4_accuracy":
        a_ok = pred_a == true
        b_ok = pred_b == true
        observed = float(a_ok.mean() - b_ok.mean())
        boot = np.empty(N_BOOT, dtype=float)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, n)
            boot[i] = float(a_ok[idx].mean() - b_ok[idx].mean())
    elif metric == "macro_f1":
        labels = sorted(set(true))
        observed = float(
            f1_score(true, pred_a, labels=labels, average="macro", zero_division=0)
            - f1_score(true, pred_b, labels=labels, average="macro", zero_division=0)
        )
        boot = np.empty(N_BOOT, dtype=float)
        for i in range(N_BOOT):
            idx = rng.integers(0, n, n)
            sampled_true = true[idx]
            labels_i = sorted(set(sampled_true))
            boot[i] = float(
                f1_score(sampled_true, pred_a[idx], labels=labels_i, average="macro", zero_division=0)
                - f1_score(sampled_true, pred_b[idx], labels=labels_i, average="macro", zero_division=0)
            )
    else:
        raise ValueError(metric)

    low, high = np.percentile(boot, [2.5, 97.5])
    p_value = min(1.0, 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0))))
    return observed, float(low), float(high), p_value


def load_joined(
    dataset: str,
    rerun: Path,
    blastp_dir: Path,
    foldseek_dir: Path,
    train_counts: dict[str, int],
) -> pd.DataFrame:
    suffix = "lt30" if dataset == "lt30" else "30_50"
    gaca = pd.read_csv(rerun / f"predictions_{suffix}.csv", dtype={"protein_id": str})
    blastp = pd.read_csv(blastp_dir / f"predictions_{suffix}.csv", dtype={"protein_id": str})
    foldseek = pd.read_csv(foldseek_dir / f"predictions_{suffix}.csv", dtype={"protein_id": str})
    gaca_labels = None
    for name, frame in (("GaCA", gaca), ("BLASTp", blastp), ("Foldseek", foldseek)):
        required = {"protein_id", "true_ec"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{dataset}: {name} is missing columns: {sorted(missing)}")
        if frame["protein_id"].isna().any() or frame["protein_id"].duplicated().any():
            raise ValueError(f"{dataset}: {name} has missing or duplicate protein IDs")
        if frame["true_ec"].isna().any():
            raise ValueError(f"{dataset}: {name} has missing ground-truth EC labels")
        if set(frame["protein_id"]) != set(gaca["protein_id"]):
            raise ValueError(f"{dataset}: {name} test IDs differ from GaCA")
        labels = frame.set_index("protein_id")["true_ec"].astype(str).sort_index()
        if gaca_labels is None:
            gaca_labels = labels
        elif not labels.equals(gaca_labels):
            mismatched = labels.index[labels != gaca_labels].tolist()
            raise ValueError(
                f"{dataset}: {name} ground-truth EC labels differ from GaCA "
                f"for {len(mismatched)} protein IDs"
            )
    gaca = gaca.rename(
        columns={
            "protein_id": "id",
            "predicted_ec": "gaca_pred_ec",
            "top_class_softmax_score": "gaca_score",
        }
    )
    blastp = blastp.rename(
        columns={
            "protein_id": "id",
            "predicted_ec": "blast_pred_ec",
            "hit_identity": "blast_pident",
        }
    )
    foldseek = foldseek.rename(
        columns={"protein_id": "id", "predicted_ec": "foldseek_pred_ec"}
    )
    joined = gaca[["id", "true_ec", "gaca_pred_ec", "gaca_score"]].merge(
        blastp[["id", "true_ec", "blast_pred_ec", "blast_pident"]],
        on="id",
        validate="one_to_one",
        suffixes=("", "_blastp"),
    ).merge(
        foldseek[["id", "true_ec", "foldseek_pred_ec"]],
        on="id",
        validate="one_to_one",
        suffixes=("", "_foldseek"),
    )
    if len(joined) != len(gaca):
        raise RuntimeError(f"{dataset}: retrieval predictions do not cover every GaCA protein")
    joined["true_ec"] = joined["true_ec"].astype(str)
    for column in ("gaca_pred_ec", "blast_pred_ec", "foldseek_pred_ec"):
        joined[column] = joined[column].fillna("").astype(str)
    joined["gaca_correct_l4"] = (joined["gaca_pred_ec"] == joined["true_ec"]).astype(int)
    joined["blast_correct_l4"] = (joined["blast_pred_ec"] == joined["true_ec"]).astype(int)
    joined["foldseek_correct_l4"] = (joined["foldseek_pred_ec"] == joined["true_ec"]).astype(int)
    joined["train_count"] = joined["true_ec"].map(train_counts).fillna(0).astype(int)
    joined["train_count_bin"] = joined["train_count"].map(
        lambda count: "1-2" if count <= 2 else "3-5" if count <= 5 else "6-10" if count <= 10 else ">10"
    )
    return joined

def paired_tables(joined: dict[str, pd.DataFrame], comparator: str) -> pd.DataFrame:
    pred_col = "blast_pred_ec" if comparator == "BLASTp" else "foldseek_pred_ec"
    correct_col = "blast_correct_l4" if comparator == "BLASTp" else "foldseek_correct_l4"
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    for dataset in ["lt30", "30_50"]:
        df = joined[dataset]
        true = df["true_ec"].to_numpy(str)
        gaca = df["gaca_pred_ec"].to_numpy(str)
        other = df[pred_col].to_numpy(str)
        g_ok = df["gaca_correct_l4"].to_numpy(bool)
        o_ok = df[correct_col].to_numpy(bool)
        g_only = int(np.sum(g_ok & ~o_ok))
        o_only = int(np.sum(o_ok & ~g_ok))
        discordant = g_only + o_only
        mcnemar = float(binomtest(min(g_only, o_only), discordant, 0.5).pvalue) if discordant else 1.0
        g_summary = summary(true, gaca)
        o_summary = summary(true, other)
        for metric in ["level4_accuracy", "macro_f1"]:
            observed, low, high, bootstrap_p = paired_bootstrap(true, gaca, other, metric, rng)
            summary_key = "level_4_accuracy" if metric == "level4_accuracy" else metric
            rows.append(
                {
                    "dataset": dataset,
                    "metric": metric,
                    "comparator": comparator,
                    "comparator_value": o_summary[summary_key],
                    "gaca_value": g_summary[summary_key],
                    "difference": observed,
                    "ci95_low": low,
                    "ci95_high": high,
                    "paired_p": mcnemar if metric == "level4_accuracy" else bootstrap_p,
                    "paired_test": "exact two-sided McNemar" if metric == "level4_accuracy" else "two-sided paired bootstrap",
                    "gaca_only_correct": g_only if metric == "level4_accuracy" else None,
                    "comparator_only_correct": o_only if metric == "level4_accuracy" else None,
                    "n": len(df),
                }
            )
    return pd.DataFrame(rows)


def long_tail(joined: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict] = []
    for dataset in ["lt30", "30_50"]:
        df = joined[dataset]
        for bin_name in ["1-2", "3-5", "6-10", ">10"]:
            sub = df[df["train_count_bin"] == bin_name]
            true = sub["true_ec"].to_numpy(str)
            blast = sub["blast_pred_ec"].to_numpy(str)
            gaca = sub["gaca_pred_ec"].to_numpy(str)
            b_summary = summary(true, blast)
            g_summary = summary(true, gaca)
            rows.append(
                {
                    "dataset": dataset,
                    "training_examples_per_ec": bin_name,
                    "number_of_ec_classes": int(sub["true_ec"].nunique()),
                    "test_proteins": int(len(sub)),
                    "blastp_accuracy_pct": 100 * b_summary["level_4_accuracy"],
                    "gaca_accuracy_pct": 100 * g_summary["level_4_accuracy"],
                    "delta_pct": 100 * (g_summary["level_4_accuracy"] - b_summary["level_4_accuracy"]),
                    "blastp_macro_f1": b_summary["macro_f1"],
                    "gaca_macro_f1": g_summary["macro_f1"],
                }
            )
    return pd.DataFrame(rows)


def cases(joined: dict[str, pd.DataFrame]) -> pd.DataFrame:
    specs = [
        ("BLASTp wrong, GaCA correct", lambda d: (d.gaca_correct_l4 == 1) & (d.blast_correct_l4 == 0)),
        ("Both correct", lambda d: (d.gaca_correct_l4 == 1) & (d.blast_correct_l4 == 1)),
        ("BLASTp correct, GaCA wrong", lambda d: (d.gaca_correct_l4 == 0) & (d.blast_correct_l4 == 1)),
    ]
    rows: list[dict] = []
    for dataset in ["lt30", "30_50"]:
        df = joined[dataset]
        for case_type, mask in specs:
            sub = df[mask(df)].sort_values(["gaca_score", "blast_pident"], ascending=[False, False])
            if sub.empty:
                continue
            row = sub.iloc[0]
            rows.append(
                {
                    "dataset": dataset,
                    "case_type": case_type,
                    "query_id": row["id"],
                    "true_ec": row["true_ec"],
                    "blastp_prediction": row["blast_pred_ec"],
                    "gaca_prediction": row["gaca_pred_ec"],
                    "gaca_top_class_softmax_score": float(row["gaca_score"]),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaca-predictions", type=Path, required=True)
    parser.add_argument("--blastp-predictions", type=Path, required=True)
    parser.add_argument("--foldseek-predictions", type=Path, required=True)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(args.train_csv, dtype={"EC number": str})
    train_counts = train["EC number"].astype(str).value_counts().to_dict()
    joined = {
        dataset: load_joined(
            dataset,
            args.gaca_predictions,
            args.blastp_predictions,
            args.foldseek_predictions,
            train_counts,
        )
        for dataset in ["lt30", "30_50"]
    }
    blast = paired_tables(joined, "BLASTp")
    foldseek = paired_tables(joined, "Foldseek")
    bins = long_tail(joined)
    selected_cases = cases(joined)
    blast.to_csv(args.output_dir / "updated_gaca_vs_blastp.csv", index=False)
    foldseek.to_csv(args.output_dir / "updated_gaca_vs_foldseek.csv", index=False)
    bins.to_csv(args.output_dir / "updated_long_tail.csv", index=False)
    selected_cases.to_csv(args.output_dir / "updated_table8_cases.csv", index=False)
    payload = {
        "paired_blastp": blast.to_dict(orient="records"),
        "paired_foldseek": foldseek.to_dict(orient="records"),
        "long_tail": bins.to_dict(orient="records"),
        "cases": selected_cases.to_dict(orient="records"),
    }
    (args.output_dir / "updated_derived_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
