"""Run the unified LightGBM protocol with the exact legacy pooled features."""

import importlib.util
import json
import pickle
from pathlib import Path

import numpy as np


BASE_SCRIPT = Path(
    r"D:\EC\new_str\Modify\journal_rerun_20260727"
    r"\code\run_lightgbm_unified.py"
)
OUTPUT_ROOT = Path(
    r"D:\EC\new_str\Modify\journal_rerun_20260727"
)
FEATURE_FILES = {
    "seq_train": Path(
        r"D:\EC\Sequences_Embeddings_use_ESM-2"
        r"\1024_train_seq_embeddings_esm2_mean.pkl"
    ),
    "seq_lt30": Path(
        r"D:\EC\Sequences_Embeddings_use_ESM-2"
        r"\1024_test_seq_embeddings_esm2_mean.pkl"
    ),
    "seq_30_50": Path(
        r"D:\EC\Sequences_Embeddings_use_ESM-2"
        r"\30_50_seq_embeddings_esm2_mean.pkl"
    ),
    "str_train": Path(
        r"D:\EC\new_str\test_30"
        r"\train_structure_embeddings_esm_if.pkl"
    ),
    "str_lt30": Path(
        r"D:\EC\new_str\test_30"
        r"\test_structure_embeddings_esm_if.pkl"
    ),
    "str_30_50": Path(
        r"D:\EC\new_str\test_30-50"
        r"\30-50test_structure_embeddings_esm_if.pkl"
    ),
}


def load_base_module():
    specification = importlib.util.spec_from_file_location(
        "lightgbm_unified", BASE_SCRIPT
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


class ExactPooledStore:
    def __init__(self, directory):
        name = Path(directory).name
        source = FEATURE_FILES[name]
        with source.open("rb") as handle:
            self.features = pickle.load(handle)

    def __contains__(self, protein_id):
        return str(protein_id) in self.features

    def mean(self, protein_id):
        value = self.features[str(protein_id)]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        if value.ndim > 1:
            value = value.mean(axis=0, dtype=np.float32)
        return value


def main():
    module = load_base_module()
    module.MemmapFeatureStore = ExactPooledStore
    module.main()

    metrics_path = (
        OUTPUT_ROOT / "baselines" / "lightgbm" / "metrics.json"
    )
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    metrics["feature_definition"] = (
        "exact pooled feature files used by the original baseline: "
        "mean-pooled ESM-2 vectors concatenated with the stored "
        "protein-level L2-normalised ESM-IF1 vectors"
    )
    metrics["feature_files"] = {
        name: str(path) for name, path in FEATURE_FILES.items()
    }
    metrics_path.write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
