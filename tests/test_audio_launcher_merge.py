"""Exercise the launcher options that overlap the audio/base merge."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class AudioLauncherMergeTests(unittest.TestCase):
    def launch(self, backend='apertus_1p5_vllm', extra=()):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            model = base / 'model'
            model.mkdir()
            env = {**os.environ, 'MODEL_BACKEND': backend,
                   'WANDB_API_KEY': 'test-unused', 'HF_TOKEN': 'test-unused',
                   'RUN_ID': 'merge-test', 'GEN_KWARGS': '', 'ENABLE_IMAGE_TOKEN_CACHE': ''}
            for name in ['LOG_DIR', 'OUTPUT_PATH', 'CACHE_BASE', 'HF_HOME', 'NLTK_DATA',
                         'XDG_CACHE_HOME', 'VLLM_CACHE_ROOT', 'LMMS_EVAL_MODELS_CACHE']:
                env[name] = str(base / name.lower())
            command = ['bash', str(ROOT / 'launchers/eval.sh'), '--eval-framework', 'lmms-eval',
                       '--model', str(model), '--tasks', 'google_fleurs', '--dry-run', *extra]
            result = subprocess.run(command, env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            line = next(line for line in result.stdout.splitlines() if 'dry-run:' in line)
            return shlex.split(line.split('dry-run:', 1)[1])

    def test_audio_overrides_and_job_passthrough_survive_dispatch(self):
        args = self.launch(extra=['--size', '70b', '--num-processes', '2',
                                 '--gpu-memory-utilization', '0.7', '--max-num-batched-tokens', '2048',
                                 '--enable-image-token-cache', 'false', '--extra-model-args', 'tensor_parallel_size=2',
                                 '--', '--max-model-len', '32768'])
        for option, value in [('--num-processes', '2'), ('--gpu-memory-utilization', '0.7'),
                              ('--max-num-batched-tokens', '2048'), ('--enable-image-token-cache', 'false'),
                              ('--extra-model-args', 'tensor_parallel_size=2'), ('--max-model-len', '32768'),
                              ('--gen-kwargs', '')]:
            self.assertEqual(args[args.index(option) + 1], value)

    def test_backend_cache_default_preserves_explicit_audio_override(self):
        for backend, extra, expected in [('apertus_1p5_vllm', [], 'true'),
                                          ('qwen2_audio', [], 'false'),
                                          ('qwen2_audio', ['--enable-image-token-cache', 'true'], 'true')]:
            with self.subTest(backend=backend, extra=extra):
                args = self.launch(backend, extra)
                self.assertEqual(args[args.index('--enable-image-token-cache') + 1], expected)


if __name__ == '__main__':
    unittest.main()
