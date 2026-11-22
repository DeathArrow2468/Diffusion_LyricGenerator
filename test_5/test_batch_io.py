"""CPU-only checks: python -m unittest discover -s test_5 -p test_batch_io.py"""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('batch_test', Path(__file__).with_name('batch_test.py'))
batch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch)


class BatchIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_multiline_unicode_csv_round_trip(self):
        prompts = ['A comma, and "quotes"', 'Two\nlines — café']
        path = self.root / 'results.csv'
        batch.write_results(prompts, '7639', path, lambda i, p: 'Verse\n"Hello", café')
        with path.open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([r['prompt'] for r in rows], prompts)
        self.assertEqual(rows[1]['output'], 'Verse\n"Hello", café')
        self.assertTrue(all(r['model_job_id'] == '7639' for r in rows))
        self.assertFalse(path.with_suffix('.partial.csv').exists())

    def test_failure_preserves_completed_rows(self):
        path = self.root / 'results.csv'
        def generate(i, p):
            if i == 1:
                raise RuntimeError('simulated inference failure')
            return 'completed output'
        with self.assertRaises(RuntimeError):
            batch.write_results(['first', 'second'], '7639', path, generate)
        self.assertFalse(path.exists())
        with path.with_suffix('.partial.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['prompt'], 'first')

    def test_existing_results_not_overwritten(self):
        path = self.root / 'results.csv'
        path.write_text('keep me')
        with self.assertRaises(FileExistsError):
            batch.write_results(['p'], '7639', path, lambda i, p: 'output')
        self.assertEqual(path.read_text(), 'keep me')

    def test_json_and_text_prompts(self):
        path = self.root / 'prompts.json'
        values = ['First\nsecond', 'Third']
        path.write_text(json.dumps(values))
        self.assertEqual(batch.read_prompts(path), values)
        path = self.root / 'prompts.txt'
        path.write_text('First\n\nSecond\n')
        self.assertEqual(batch.read_prompts(path), ['First', 'Second'])

    def test_reject_invalid_prompt_lists(self):
        path = self.root / 'prompts.json'
        for value in ([], [''], ['good', None], {'prompt': 'bad'}, 'not a list'):
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                batch.read_prompts(path)


if __name__ == '__main__':
    unittest.main()
