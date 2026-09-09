"""CPU regressions for the audio import and emitted dashboard behavior.

Run: python3 -m unittest discover -s tests -v
Set NODE to a node executable to exercise the inline browser script as well.
"""
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import make_dashboard as dashboard

REPORT = ROOT / 'docs/audio/audio_benchmark_results.html'
LOW_LABEL = 'Apertus-1.5-8B-SFT-RL-DPO-SDPO-Low-Less-Refuse-Feedback'
LOW_KEY = 'sft-rl-dpo-sdpo-low-less-refuse-feedback'
PRETRAIN_LABEL = 'Apertus 8B 1.5 pretrain long context'
PRETRAIN_KEY = 'apertus 8b 1.5 pretrain long context'
EXPECTED_PRETRAIN = {
    'fleurs_it_it': 9.60, 'fleurs_en_us': 14.23, 'fleurs_pt_br': 14.36,
    'fleurs_de_de': 17.21, 'fleurs_th_th': 20.02, 'fleurs_fr_fr': 21.30,
    'fleurs_es_419': 25.20, 'fleurs_vi_vn': 26.06, 'fleurs_ca_es': 26.26,
    'google_fleurs_cmn_hans_cn': 29.01, 'fleurs_pl_pl': 31.45,
    'fleurs_uk_ua': 35.47, 'fleurs_hi_in': 39.70,
}


def generate(root, *args):
    output = root / 'dashboard.html'
    subprocess.run([sys.executable, str(ROOT / 'scripts/make_dashboard.py'),
                    '--runs-root', str(root), '--audio-report', str(REPORT),
                    '-o', str(output), *args], check=True, capture_output=True, text=True)
    return output, json.loads(re.search(
        r'<script id="data" type="application/json">(.*?)</script>',
        output.read_text(), re.S)[1])


class AudioCollectionTests(unittest.TestCase):
    def test_report_models_use_native_checkpoint_identities(self):
        models, table = dashboard.collect_audio_report(REPORT, None)
        self.assertEqual(set(models), {
            LOW_KEY, 'sft-rl-dpo-sdpo-mix-less-refuse-feedback',
            'apertus-v1.5-8b', 'apertus-v1.5-70b',
            'apertus-1.5-70b-sft-rl-dpo-sdpo', 'qwen2.5-omni-7b',
            'qwen2-audio-7b-instruct', 'kimi-audio-7b-instruct', PRETRAIN_KEY,
        })
        self.assertEqual(len(table), 73)
        self.assertEqual(sum(len(r['cells']) for r in table), 570)
        fleurs = next(r for r in table if r['task'] == 'fleurs_it_it')
        self.assertEqual(fleurs['cells'][LOW_KEY]['raw'], 8.0486)

    def test_pretrain_filter_retains_all_supplied_values(self):
        for model_filter in [PRETRAIN_LABEL, PRETRAIN_KEY]:
            with self.subTest(model_filter=model_filter):
                models, table = dashboard.collect_audio_report(REPORT, [model_filter])
                self.assertEqual(models, [PRETRAIN_KEY])
                self.assertEqual({r['task']: next(iter(r['cells'].values()))['raw']
                                  for r in table}, EXPECTED_PRETRAIN)
                self.assertTrue(all(len(r['cells']) == 1 for r in table))

    def test_curated_native_result_replaces_report_cell_and_keeps_display_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / LOW_LABEL
            model.mkdir()
            for name, value, mtime in [('old', 10.0, 10), ('fresh', 1.2, 20)]:
                path = model / f'{name}_results.json'
                path.write_text(json.dumps({'results': {'fleurs_it_it': {'wer,none': value}}}))
                os.utime(path, (mtime, mtime))
            _, data = generate(root, '--only', LOW_KEY)
            self.assertEqual(data['models'], [LOW_KEY])
            self.assertEqual(data['labels'][LOW_KEY], LOW_LABEL)
            row = next(r for r in data['table'] if r['task'] == 'fleurs_it_it')
            self.assertEqual(row['cells'][LOW_KEY], {'v': 1.2, 'raw': 1.2, 'run': 'fresh_results.json'})
            self.assertEqual(row['cat'], 'Multilingual ASR')
            self.assertEqual(row['dir'], -1)
            # Historical cells still supply tasks without native results.
            self.assertEqual(len(data['table']), 73)

    def test_manifest_keeps_all_audio_columns_and_native_alias_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            native = root / 'sdpo-mix-less-refuse-feedback'
            native.mkdir()
            (native / 'fresh_results.json').write_text(json.dumps({
                'results': {'fleurs_it_it': {'wer,none': 1.2}}}))
            _, data = generate(root, '--models-file', str(ROOT / 'scripts/dashboard_models.txt'))
            self.assertEqual(len(data['models']), 9)
            self.assertEqual(len(data['table']), 73)
            self.assertEqual(sum(len(r['cells']) for r in data['table']), 570)
            self.assertIn(PRETRAIN_KEY, data['models'])
            fleurs = next(r for r in data['table'] if r['task'] == 'fleurs_it_it')
            self.assertEqual(fleurs['cells']['sdpo-mix-less-refuse-feedback']['raw'], 1.2)

    def test_canonical_model_filter_keeps_native_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / LOW_LABEL
            model.mkdir()
            (model / 'fresh_results.json').write_text(json.dumps({
                'results': {'fleurs_it_it': {'wer,none': 1.2}}}))
            _, data = generate(root, '--models', LOW_KEY)
            self.assertEqual(data['models'], [LOW_KEY])
            row = next(r for r in data['table'] if r['task'] == 'fleurs_it_it')
            self.assertEqual(row['cells'][LOW_KEY]['raw'], 1.2)

    def test_full_report_label_filter_keeps_its_native_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'Apertus-v1.5-8B'
            model.mkdir()
            (model / 'fresh_results.json').write_text(json.dumps({
                'results': {'fleurs_it_it': {'wer,none': 1.2}}}))
            _, data = generate(root, '--models', 'swiss-ai/Apertus-v1.5-8B')
            self.assertEqual(data['models'], ['apertus-v1.5-8b'])
            row = next(r for r in data['table'] if r['task'] == 'fleurs_it_it')
            self.assertEqual(row['cells']['apertus-v1.5-8b']['raw'], 1.2)

    def test_full_report_label_filter_resolves_manifest_native_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / 'sdpo-mix-less-refuse-feedback'
            model.mkdir()
            (model / 'fresh_results.json').write_text(json.dumps({
                'results': {'fleurs_it_it': {'wer,none': 1.2}}}))
            for model_filter in [
                'Apertus-1.5-8B-SFT-RL-DPO-SDPO-Mix-Less-Refuse-Feedback',
                'sft-rl-dpo-sdpo-mix-less-refuse-feedback',
                'sdpo-mix-less-refuse-feedback',
            ]:
                with self.subTest(model_filter=model_filter):
                    _, data = generate(root, '--models-file', str(ROOT / 'scripts/dashboard_models.txt'),
                                       '--models', model_filter)
                    self.assertEqual(data['models'], ['sdpo-mix-less-refuse-feedback'])
                    self.assertEqual(len(data['table']), 73)
                    row = next(r for r in data['table'] if r['task'] == 'fleurs_it_it')
                    self.assertEqual(row['cells']['sdpo-mix-less-refuse-feedback'],
                                     {'v': 1.2, 'raw': 1.2, 'run': 'fresh_results.json'})

    @unittest.skipUnless(os.environ.get('NODE') or shutil.which('node'), 'Node.js unavailable (set NODE)')
    def test_browser_summary_matrix_and_selection_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = generate(Path(tmp))
            result = subprocess.run([os.environ.get('NODE') or shutil.which('node'),
                                     str(ROOT / 'tests/dashboard_audio_behavior.cjs'), str(path)],
                                    text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
