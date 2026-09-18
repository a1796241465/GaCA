from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef


EC_ROOT = Path(r"D:\EC")
OUT = EC_ROOT / "new_str" / "Modify" / "journal_tbc_outputs"
CLEAN_APP = EC_ROOT / "CLEAN-main" / "app"

TRAIN_CSV = EC_ROOT / "train_cleaned_with_structure.csv"
TEST_LT30_CSV = EC_ROOT / "test_30_cleaned_with_structure.csv"
TEST_3050_CSV = EC_ROOT / "test_30_50_clean.csv"

STRICT = {
    "lt30": {
        "label": "<30%",
        "strict_csv": OUT / "strict_closed_set_predictions_lt30.csv",
        "clean_result": CLEAN_APP / "results" / "your_test30_tab_maxsep.csv",
        "retrained_result": CLEAN_APP / "results" / "gaca_test_lt30_strict_maxsep.csv",
        "test_csv": TEST_LT30_CSV,
        "prepared_name": "gaca_test_lt30_strict",
    },
    "30-50": {
        "label": "30-50%",
        "strict_csv": OUT / "strict_closed_set_predictions_30-50.csv",
        "clean_result": CLEAN_APP / "results" / "your_test3050_tab_maxsep.csv",
        "retrained_result": CLEAN_APP / "results" / "gaca_test_30_50_strict_maxsep.csv",
        "test_csv": TEST_3050_CSV,
        "prepared_name": "gaca_test_30_50_strict",
    },
}


def ec_prefix(ec: object, level: int) -> str:
    parts = str(ec).split(".")
    return ".".join(parts[:level]) if len(parts) >= level else str(ec)


def parse_clean_maxsep(path: Path) -> Dict[str, Tuple[str, str, int]]:

    result: Dict[str, Tuple[str, str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            query_id = parts[0]
            top = parts[1]
            pred_ec = top.replace("EC:", "").split("/")[0]
            score = top.split("/")[-1] if "/" in top else ""
            result[query_id] = (pred_ec, score, len(parts) - 1)
    return result


def metric_rows(df: pd.DataFrame, pred_col: str, method: str, dataset: str, label: str) -> List[dict]:
    rows = []
    true = df["true_ec"].astype(str)
    pred = df[pred_col].astype(str)
    for level in [1, 2, 3, 4]:
        acc = (true.map(lambda x: ec_prefix(x, level)) == pred.map(lambda x: ec_prefix(x, level))).mean()
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": label,
                "method": method,
                "metric": f"level_{level}_accuracy",
                "value": acc,
                "n": len(df),
            }
        )
    rows.append(
        {
            "dataset": dataset,
            "dataset_label": label,
            "method": method,
            "metric": "macro_f1_level4",
            "value": f1_score(true, pred, average="macro", zero_division=0),
            "n": len(df),
        }
    )
    rows.append(
        {
            "dataset": dataset,
            "dataset_label": label,
            "method": method,
            "metric": "mcc_level4",
            "value": matthews_corrcoef(true, pred),
            "n": len(df),
        }
    )
    return rows


def write_clean_style_table(df: pd.DataFrame, path: Path) -> None:

    cols = ["Entry", "EC number", "Sequence"]
    df.loc[:, cols].to_csv(path, sep="\t", index=False)


def prepare_strict_clean_inputs() -> None:
    prepared = OUT / "unified_baseline_inputs"
    prepared.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_CSV)
    train_out = prepared / "gaca_train_strict.csv"
    write_clean_style_table(train.rename(columns={"EC number": "EC number"}), train_out)

    for dataset, cfg in STRICT.items():
        strict_ids = list(pd.read_csv(cfg["strict_csv"])["id"].astype(str))
        full_test = pd.read_csv(cfg["test_csv"])
        full_test = full_test.set_index(full_test["Entry"].astype(str), drop=False)
        missing = [pid for pid in strict_ids if pid not in full_test.index]
        if missing:
            raise RuntimeError(f"{dataset}: strict IDs missing from full test CSV: {missing[:5]}")
        strict_test = full_test.loc[strict_ids].reset_index(drop=True)
        write_clean_style_table(strict_test, prepared / f"{cfg['prepared_name']}.csv")

    commands = [
        "# Copy these files to D:/EC/CLEAN-main/app/data or the corresponding Ubuntu CLEAN app/data directory.",
        "# Then run from CLEAN-main/app after installing/building CLEAN.",
        "",
        "python - <<'PY'",
        "from CLEAN.utils import csv_to_fasta, retrive_esm1b_embedding, compute_esm_distance",
        "for name in ['gaca_train_strict', 'gaca_test_lt30_strict', 'gaca_test_30_50_strict']:",
        "    csv_to_fasta(f'data/{name}.csv', f'data/{name}.fasta')",
        "    retrive_esm1b_embedding(name)",
        "compute_esm_distance('gaca_train_strict')",
        "PY",
        "",
        "# Triplet version, closest to the pretrained split70 checkpoint dimensionality.",
        "python train-triplet.py --training_data gaca_train_strict --model_name gaca_train_strict_triplet --epoch 2000",
        "",
        "python - <<'PY'",
        "from CLEAN.infer import infer_maxsep",
        "infer_maxsep('gaca_train_strict', 'gaca_test_lt30_strict', report_metrics=True, pretrained=False, model_name='gaca_train_strict_triplet')",
        "infer_maxsep('gaca_train_strict', 'gaca_test_30_50_strict', report_metrics=True, pretrained=False, model_name='gaca_train_strict_triplet')",
        "PY",
    ]
    (prepared / "clean_strict_retrain_commands.sh").write_text("\n".join(commands) + "\n", encoding="utf-8")


def evaluate_clean_outputs() -> None:
    all_metrics: List[dict] = []
    result_groups = [
        ("CLEAN-pretrained(split70)", "clean_result", "clean_pretrained_split70"),
        ("CLEAN-retrained(GaCA-train)", "retrained_result", "clean_retrained_gaca_train"),
    ]
    for method, result_key, output_prefix in result_groups:
        group_metrics: List[dict] = []
        missing_any = False
        for dataset, cfg in STRICT.items():
            result_path = cfg[result_key]
            if not result_path.exists():
                missing_any = True
                continue
            strict = pd.read_csv(cfg["strict_csv"])
            clean = parse_clean_maxsep(result_path)
            rows = []
            for _, row in strict.iterrows():
                query_id = str(row["id"])
                pred_ec, score, n_calls = clean.get(query_id, ("", "", 0))
                rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": cfg["label"],
                        "id": query_id,
                        "true_ec": str(row["true_ec"]),
                        "clean_pred_ec": pred_ec,
                        "clean_score": score,
                        "clean_num_ec_calls": n_calls,
                    }
                )
            pred_df = pd.DataFrame(rows)
            pred_df.to_csv(OUT / f"{output_prefix}_strict_predictions_{dataset}.csv", index=False)
            group_metrics.extend(metric_rows(pred_df, "clean_pred_ec", method, dataset, cfg["label"]))

        if group_metrics:
            metrics = pd.DataFrame(group_metrics)
            metrics.to_csv(OUT / f"{output_prefix}_strict_metrics.csv", index=False)
            all_metrics.extend(group_metrics)
        elif missing_any:
            print(f"Skipping {method}: expected result files are not present yet.")

    if all_metrics:
        all_metrics_df = pd.DataFrame(all_metrics)
        all_metrics_df.to_csv(OUT / "clean_all_available_strict_metrics.csv", index=False)
        print(
            all_metrics_df.pivot_table(
                index=["dataset_label", "method"],
                columns="metric",
                values="value",
            )
            .round(4)
        )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prepare_strict_clean_inputs()
    evaluate_clean_outputs()
    print(f"\nWrote CLEAN strict protocol files to: {OUT}")
    print(f"Prepared CLEAN retraining inputs under: {OUT / 'unified_baseline_inputs'}")


if __name__ == "__main__":
    main()
