import tempfile
import subprocess
import sys
import json
import unittest
from pathlib import Path

from steering.experiment_utils import read_aligned, row_indices, fingerprint
from compare_steering_runs import summarize


class DataTests(unittest.TestCase):
    def test_split_disjoint_shared_and_stable(self):
        tune = set(row_indices(100, 'tune'))
        held = set(row_indices(100, 'holdout'))
        self.assertFalse(tune & held)
        self.assertEqual(tune | held, set(range(100)))
        self.assertEqual(tune, set(row_indices(100, 'tune')))
        self.assertEqual(set(row_indices(110, 'tune')) & set(range(100)), tune)

    def test_alignment_rejects_blank_and_length_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            (d/'en.txt').write_text('one\ntwo\n')
            (d/'mt.txt').write_text('one\n\n')
            with self.assertRaises(ValueError):
                read_aligned(d, ['en', 'mt'])
            (d/'mt.txt').write_text('one\n')
            with self.assertRaises(ValueError):
                read_aligned(d, ['en', 'mt'])
            (d/'mt.txt').write_text('a\nb\n')
            self.assertEqual(read_aligned(d, ['en', 'mt'])['mt'], ['a', 'b'])

    def test_config_hash_order_and_parameter_change(self):
        self.assertEqual(fingerprint({'a': 1, 'b': 2}), fingerprint({'b': 2, 'a': 1}))
        self.assertNotEqual(fingerprint({'layer': 20}), fingerprint({'layer': 24}))

    def test_cli_dry_run_and_scope_guard(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temp:
            d = Path(temp)
            (d/'en.txt').write_text('question one\nquestion two\n')
            (d/'artifact.pt').write_text('dry-run does not deserialize this')
            command = [sys.executable, str(root/'subspace_generate.py'), '--artifact',
                       str(d/'artifact.pt'), '--questions-dir', str(d), '--pairs', 'en:mt',
                       '--split', 'all', '--dry-run']
            result = subprocess.run(command, cwd=root, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)['question_indices'], [0, 1])
            bad = subprocess.run(command + ['--backend', 'vllm'], cwd=root,
                                 text=True, capture_output=True)
            self.assertNotEqual(bad.returncode, 0)
            self.assertIn('--scope all', bad.stderr)

    def test_aggregations_are_distinct(self):
        r = summarize({'by_pair': {'a': {'lfs': 1., 'or': 0.}, 'b': {'lfs': 0., 'or': 1.}}})
        self.assertEqual(r['lss_from_macro_components'], 0.5)
        self.assertEqual(r['mean_pair_lss'], 0.)


if __name__ == '__main__':
    unittest.main()
