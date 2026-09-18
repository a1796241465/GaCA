from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef


EC_ROOT = Path(r"D:\EC")
OUT = EC_ROOT / "new_str" / "Modify" / "journal_tbc_outputs"

TRAIN_CSV = EC_ROOT / "train_cleaned_with_structure.csv"
TRAIN_STRUCTURES = EC_ROOT / "train_structures"
TEST_STRUCTURES = {
    "lt30": EC_ROOT / "test_structures",
    "30-50": EC_ROOT / "test_30-50_structures",
}
STRICT = {
    "lt30": {
        "label": "<30%",
        "strict_csv": OUT / "strict_closed_set_predictions_lt30.csv",
        "foldseek_tsv": OUT / "foldseek_lt30.tsv",
    },
    "30-50": {
        "label": "30-50%",
        "strict_csv": OUT / "strict_closed_set_predictions_30-50.csv",
        "foldseek_tsv": OUT / "foldseek_30_50.tsv",
    },
}


def normalize_structure_id(value: object) -> str:
    """Map Foldseek query/target names or paths back to UniProt-style IDs."""
    text = str(value).strip().replace("\\", "/")
    base = text.rsplit("/", 1)[-1]
    for suffix in [".pdb.gz", ".cif.gz", ".mmcif.gz", ".pdb", ".cif", ".mmcif"]:
        if base.lower().endswith(suffix):
            base = base[: -len(suffix)]
            break
    if base.startswith("AF-") and "-F" in base:
        base = base[3:].split("-F", 1)[0]
    return base


def ec_prefix(ec: object, level: int) -> str:
    parts = str(ec).split(".")
    return ".".join(parts[:level]) if len(parts) >= level else str(ec)


def read_foldseek_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Foldseek result file not found: {path}. Run Foldseek first or place the TSV at this path."
        )
    columns = [
        "query",
        "target",
        "fident",
        "evalue",
        "bits",
        "alntmscore",
        "qtmscore",
        "ttmscore",
    ]
    df = pd.read_csv(path, sep="\t", names=columns, comment="#")
    df["query_id"] = df["query"].map(normalize_structure_id)
    df["target_id"] = df["target"].map(normalize_structure_id)
    for col in ["fident", "evalue", "bits", "alntmscore", "qtmscore", "ttmscore"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def choose_top_hit(df: pd.DataFrame, query_id: str) -> Optional[pd.Series]:
    subset = df[df["query_id"] == query_id].copy()
    if subset.empty:
        return None
    subset = subset.sort_values(
        by=["bits", "evalue", "alntmscore"],
        ascending=[False, True, False],
        na_position="last",
    )
    return subset.iloc[0]


def metric_rows(df: pd.DataFrame, pred_col: str, method: str, dataset: str, label: str) -> List[dict]:
    rows = []
    true = df["true_ec"].astype(str)
    pred = df[pred_col].fillna("").astype(str)
    labels = sorted(set(true))
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
            "value": f1_score(true, pred, labels=labels, average="macro", zero_division=0),
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


def write_id_and_command_files() -> None:
    prepared = OUT / "unified_baseline_inputs"
    prepared.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(TRAIN_CSV)
    train_ids = sorted(train["Entry"].astype(str).unique())
    (prepared / "foldseek_train_ids.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")

    lines = [
        "# Strict Foldseek baseline using only the user's CARE-derived structure folders.",
        "# Do not replace the target with PDB/AFDB or another external database for the main fair baseline.",
        "# Run on Ubuntu/Linux from any working directory; change the paths if your data are mounted elsewhere.",
        "",
        "TRAIN_STRUCTURES=/home/lihaotian/new_ec/train_structures",
        "TEST_LT30_STRUCTURES=/home/lihaotian/new_ec/test_structures",
        "TEST_3050_STRUCTURES=/home/lihaotian/new_ec/test_30-50_structures",
        "OUT_DIR=/home/lihaotian/new_ec/foldseek_strict_outputs",
        "mkdir -p ${OUT_DIR}",
        "",
        "foldseek easy-search ${TEST_LT30_STRUCTURES} ${TRAIN_STRUCTURES} ${OUT_DIR}/foldseek_lt30.tsv ${OUT_DIR}/tmp_lt30 "
        "--format-output \"query,target,fident,evalue,bits,alntmscore,qtmscore,ttmscore\"",
        "",
        "foldseek easy-search ${TEST_3050_STRUCTURES} ${TRAIN_STRUCTURES} ${OUT_DIR}/foldseek_30_50.tsv ${OUT_DIR}/tmp_30_50 "
        "--format-output \"query,target,fident,evalue,bits,alntmscore,qtmscore,ttmscore\"",
        "",
        "# Copy foldseek_lt30.tsv and foldseek_30_50.tsv back to:",
        f"# {OUT}",
        "# Then run evaluate_foldseek_strict_protocol.py.",
    ]
    (prepared / "foldseek_strict_commands.sh").write_text("\n".join(lines) + "\n", encoding="utf-8")

    for dataset, cfg in STRICT.items():
        strict = pd.read_csv(cfg["strict_csv"])
        ids = list(strict["id"].astype(str))
        (prepared / f"foldseek_{dataset}_strict_query_ids.txt").write_text(
            "\n".join(ids) + "\n", encoding="utf-8"
        )


def evaluate_foldseek_outputs() -> None:
    train = pd.read_csv(TRAIN_CSV)
    train_ec: Dict[str, str] = dict(zip(train["Entry"].astype(str), train["EC number"].astype(str)))

    metrics: List[dict] = []
    for dataset, cfg in STRICT.items():
        hits = read_foldseek_tsv(cfg["foldseek_tsv"])
        strict = pd.read_csv(cfg["strict_csv"])
        rows = []
        for _, row in strict.iterrows():
            query_id = str(row["id"])
            hit = choose_top_hit(hits, query_id)
            if hit is None:
                target_id = ""
                pred_ec = ""
                hit_values = {}
            else:
                target_id = str(hit["target_id"])
                pred_ec = train_ec.get(target_id, "")
                hit_values = {
                    "foldseek_fident": hit.get("fident"),
                    "foldseek_evalue": hit.get("evalue"),
                    "foldseek_bits": hit.get("bits"),
                    "foldseek_alntmscore": hit.get("alntmscore"),
                    "foldseek_qtmscore": hit.get("qtmscore"),
                    "foldseek_ttmscore": hit.get("ttmscore"),
                }
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": cfg["label"],
                    "id": query_id,
                    "true_ec": str(row["true_ec"]),
                    "foldseek_target_id": target_id,
                    "foldseek_pred_ec": pred_ec,
                    **hit_values,
                }
            )
        pred_df = pd.DataFrame(rows)
        pred_df.to_csv(OUT / f"foldseek_strict_predictions_{dataset}.csv", index=False)
        metrics.extend(metric_rows(pred_df, "foldseek_pred_ec", "Foldseek(top1-train-structure)", dataset, cfg["label"]))

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUT / "foldseek_strict_metrics.csv", index=False)
    print(metrics_df.pivot_table(index=["dataset_label", "method"], columns="metric", values="value").round(4))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_id_and_command_files()
    try:
        evaluate_foldseek_outputs()
    except FileNotFoundError as exc:
        print(exc)
        print("\nCommand and ID files were still written under:")
        print(OUT / "unified_baseline_inputs")


if __name__ == "__main__":
    main()
