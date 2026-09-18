import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.classical.train import model_and_parameters
from common.protocol import validate_reference_tables
from embeddings.build_store import as_matrix, build_store
from evaluation.paired_statistics import load_joined

ROOT = Path(__file__).resolve().parents[1]


class ReleaseGuards(unittest.TestCase):
    def test_reference_and_label_mutation(self):
        tables = {s: pd.read_csv(ROOT / 'metadata' / f'effective_{s}.csv')
                  for s in ('train', 'lt30', '30_50')}
        validate_reference_tables(tables)
        tables['lt30'].loc[0, 'EC number'] = '0.0.0.0'
        with self.assertRaises(ValueError):
            validate_reference_tables(tables)

    def test_row_order_mutation(self):
        table = pd.read_csv(ROOT / 'metadata/effective_train.csv').iloc[::-1]
        with self.assertRaises(ValueError):
            validate_reference_tables({'train': table})

    def test_shapes(self):
        self.assertEqual(as_matrix(np.ones((3, 1, 512)), 512).shape, (3, 512))
        for a in (np.ones((512, 3)), np.ones((512,)), np.empty((0, 512))):
            with self.assertRaises(ValueError):
                as_matrix(a, 512)

    def test_store_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'features.pkl'
            source.write_bytes(pickle.dumps({'A': np.ones((2, 512), np.float16)}))
            build_store('test', source, 512, root)
            metadata = json.loads((root / 'test/metadata.json').read_text())
            for key in ('source', 'source_size_bytes', 'source_mtime_ns', 'feature_dim'):
                self.assertIn(key, metadata)
            build_store('test', source, 512, root)
            values = np.load(root / 'test/values.npy')
            np.testing.assert_array_equal(values, np.ones((2, 512)))
            source.write_bytes(pickle.dumps({'A': np.ones((3, 512), np.float16)}))
            with self.assertRaises(ValueError):
                build_store('test', source, 512, root)

    def test_canonical_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'features.pkl'
            source.write_bytes(pickle.dumps({1: np.ones((2,512)), '1': np.ones((2,512))}))
            with self.assertRaises(ValueError):
                build_store('test', source, 512, root)

    def test_invalid_feature_values(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'features.pkl'
            for value in (np.nan, 1e10):
                source.write_bytes(pickle.dumps({'A': np.full((2,512),value)}))
                with self.assertRaises(ValueError):
                    build_store('test', source, 512, root)

    def test_paired_extra_id(self):
        refs = ROOT / 'reference_results'
        counts = pd.read_csv(ROOT / 'metadata/effective_train.csv')['EC number'].value_counts().to_dict()
        joined = load_joined('lt30', refs/'neural/gaca', refs/'retrieval/blastp', refs/'retrieval/foldseek', counts)
        self.assertEqual(len(joined), 243)
        with tempfile.TemporaryDirectory() as temp:
            base = pd.read_csv(refs/'retrieval/blastp/predictions_lt30.csv')
            extra = base.iloc[[0]].copy()
            extra['protein_id'] = 'unexpected-protein'
            pd.concat([base, extra]).to_csv(Path(temp)/'predictions_lt30.csv',index=False)
            with self.assertRaises(ValueError):
                load_joined('lt30', refs/'neural/gaca', Path(temp), refs/'retrieval/foldseek', counts)

    def test_paired_true_label_mismatch(self):
        refs = ROOT / 'reference_results'
        counts = pd.read_csv(ROOT / 'metadata/effective_train.csv')['EC number'].value_counts().to_dict()
        with tempfile.TemporaryDirectory() as temp:
            base = pd.read_csv(refs/'retrieval/blastp/predictions_lt30.csv')
            base.loc[0, 'true_ec'] = '0.0.0.0'
            base.to_csv(Path(temp)/'predictions_lt30.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'ground-truth EC labels differ'):
                load_joined('lt30', refs/'neural/gaca', Path(temp), refs/'retrieval/foldseek', counts)

    def test_svm_original_solver(self):
        model, _ = model_and_parameters('svm')
        self.assertIs(model.dual, True)


if __name__ == '__main__':
    unittest.main()
