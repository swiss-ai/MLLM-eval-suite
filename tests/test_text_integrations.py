"""CPU regressions for dependency packaging, Evaluator config and lm-eval imports.

Run: python -m unittest discover -s tests -v (requires PyYAML).
"""
import configparser
import json
import os
import shlex
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import make_dashboard as dashboard


class TextIntegrations(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = {k: v for k, v in os.environ.items() if not k.startswith(("NEL_", "SBATCH_", "SLURM_"))}
        self.env.update(ORCH_REPO_ROOT=str(ROOT), LOG_DIR=str(self.root / "logs"),
                        NEL_APERTUS_MODELS_CACHE=str(self.root / "cache"),
                        PREFETCH_EMU35_VISION_TOKENIZER="false")

    def run_command(self, *args, env=None):
        result = subprocess.run(args, cwd=ROOT, env=env or self.env, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def render_evaluator(self, overrides=None, template=None, extra=()):
        # Only the external dependency's presence is stubbed; both real shell
        # entrypoints and the env/config renderer execute in dry-run mode.
        repo = self.root / "Evaluator"
        repo.mkdir(exist_ok=True)
        self.env["EVALUATOR_REPO_DIR"] = str(repo)
        self.env.update(overrides or {})
        output = self.root / "output"
        args = ["bash", "launchers/eval.sh", "--eval-framework", "Evaluator", "--tasks", "gsm8k",
                "--output-dir", str(output), "--dry-run", *extra]
        if template:
            path = self.root / "template.yaml"
            path.write_text(template)
            args += ["--config-template", str(path)]
        self.run_command(*args)
        task = output / "gsm8k"
        self.run_command("bash", "slurm/Evaluator/eval_job.slurm", "--task", "gsm8k",
                         "--config", str(task / "config.template.yaml"),
                         "--env-file", str(task / "config.env"), "--output-dir", str(task), "--dry-run")
        return yaml.safe_load((task / "config.resolved.yaml").read_text())

    def test_merged_lmms_launcher_preserves_both_flag_interfaces(self):
        model = self.root / "model"
        model.mkdir()
        self.env["WANDB_API_KEY"] = "fixture-secret-for-redaction"
        for name in ("CACHE_BASE", "HF_HOME", "NLTK_DATA", "XDG_CACHE_HOME", "VLLM_CACHE_ROOT",
                     "LMMS_EVAL_MODELS_CACHE", "OUTPUT_PATH", "TOKENIZER_PATH"):
            self.env[name] = str(self.root / name.lower())
        for size, extra, wanted in (("70b", (), "tensor_parallel_size=4,enable_thinking=True"),
                                     ("8b", ("--extra-model-args", "max_num_seqs=8"),
                                      "max_num_seqs=8,enable_thinking=True")):
            with self.subTest(size=size):
                output = self.run_command("bash", "launchers/eval.sh", "--eval-framework", "lmms-eval",
                                          "--model", str(model), "--tasks", "gqa", "--size", size,
                                          "--thinking", "--debug-mode", "--extra-framework-config", "--verbosity",
                                          "--extra-framework-config", "DEBUG", *extra, "--dry-run", "--",
                                          "--max-model-len", "32768")
                line = next(line for line in output.splitlines() if "dry-run:" in line)
                argv = shlex.split(line.split("dry-run:", 1)[1])
                self.assertEqual(argv[argv.index("--extra-model-args") + 1], wanted)
                self.assertEqual(argv.count("--extra-model-args"), 1)
                self.assertIn("--debug-mode", argv)
                self.assertEqual(argv[argv.index("--max-model-len") + 1], "32768")
                self.assertEqual([argv[i + 1] for i, arg in enumerate(argv) if arg == "--extra-framework-config"],
                                 ["--verbosity", "DEBUG"])
                self.assertNotIn("fixture-secret-for-redaction", output)
                self.assertEqual(argv[argv.index("--wandb-api-key") + 1], "***")

    def test_declared_text_dependencies_are_pinned_gitlinks(self):
        modules = configparser.ConfigParser()
        modules.read(ROOT / ".gitmodules")
        for name in ("Evaluator", "lm-evaluation-harness"):
            with self.subTest(name=name):
                path = modules[f'submodule "third_party/{name}"']["path"]
                entry = self.run_command("git", "ls-files", "--stage", "--", path).strip()
                self.assertTrue(entry, f"declared dependency {path} has no committed gitlink")
                self.assertEqual(entry.split()[0], "160000")
                self.assertEqual(entry.split()[2], "0")

    def test_environment_mappings_survive_outer_and_inner_renderer(self):
        names = ["NEL_BENCHMARK_PARAMS", "NEL_APERTUS_HF_OVERRIDES", "NEL_OUTPUT_EXPORT_CONFIG",
                 "NEL_HARBOR_AGENT_KWARGS", "NEL_HARBOR_CONTAINER_ENV"]
        expected = {"foo": 1, "nested": {"items": ["x", "closing } brace"]}}
        template = "\n".join(name + ": ${" + name + "}" for name in names)
        config = self.render_evaluator({name: json.dumps(expected) for name in names}, template)
        for name in names:
            with self.subTest(name=name):
                self.assertEqual(config[name], expected)

    def test_empty_and_unset_environment_mappings_use_defaults(self):
        for value in (None, ""):
            with self.subTest(value=value):
                overrides = {} if value is None else {"NEL_BENCHMARK_PARAMS": value, "NEL_APERTUS_HF_OVERRIDES": value}
                config = self.render_evaluator(overrides)
                self.assertEqual(config["benchmarks"][0]["params"], {})
                argv = config["services"]["apertus"]["extra_args"]
                self.assertEqual(json.loads(argv[argv.index("--hf-overrides") + 1]), {"max_position_embeddings": 131072})

    def test_real_gsm8k_template_preserves_supplied_hf_overrides(self):
        config = self.render_evaluator({"NEL_BENCHMARK_PARAMS": '{"foo":1}',
                                        "NEL_APERTUS_HF_OVERRIDES": '{"max_position_embeddings":8192}'})
        self.assertEqual(config["benchmarks"][0]["params"], {"foo": 1})
        argv = config["services"]["apertus"]["extra_args"]
        self.assertEqual(json.loads(argv[argv.index("--hf-overrides") + 1]), {"max_position_embeddings": 8192})

    def write_result(self, relative, score=0.5, mtime=100):
        path = self.root / relative / "results_20260909.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"results": {"gsm8k": {"exact_match,none": score}}}))
        os.utime(path, (mtime, mtime))
        return path

    def test_run_first_uses_model_identity_and_run_provenance(self):
        self.write_result("run-1/Apertus-1p5-8B-model/gsm8k")
        models, rows = dashboard.collect_lm_eval_harness(self.root, ["model"])
        self.assertEqual(models, ["model"])
        self.assertEqual(rows[0]["cells"]["model"]["run"], "run-1")
        self.assertEqual(rows[0]["cells"]["model"]["v"], 50.0)

    def test_model_first_uses_model_identity_and_newest_run(self):
        self.write_result("Apertus-1p5-8B-model/old-run/gsm8k", 0.9, 100)
        self.write_result("Apertus-1p5-8B-model/new-run/gsm8k", 0.5, 200)
        models, rows = dashboard.collect_lm_eval_harness(self.root, ["model"], layout="model-first")
        self.assertEqual(models, ["model"])
        self.assertEqual(rows[0]["cells"]["model"]["run"], "new-run")
        self.assertEqual(rows[0]["cells"]["model"]["v"], 50.0)

    def test_run_first_merges_repeated_model_runs_by_result_mtime(self):
        self.write_result("z-old/Apertus-1p5-8B-model/gsm8k", 0.9, 100)
        self.write_result("a-new/Apertus-1p5-8B-model/gsm8k", 0.5, 200)
        models, rows = dashboard.collect_lm_eval_harness(self.root, None)
        self.assertEqual(models, ["model"])
        self.assertEqual(rows[0]["cells"]["model"]["v"], 50.0)
        self.assertEqual(rows[0]["cells"]["model"]["run"], "a-new")

    def test_missing_optional_results_root_is_skipped(self):
        self.assertEqual(dashboard.collect_lm_eval_harness(self.root / "missing", None), ([], []))

    def test_collision_audit_uses_same_layout_contract(self):
        empty = self.root / "empty"
        empty.mkdir()
        for layout, paths in (
            ("run-first", ("run/Apertus-1p5-8B-model/gsm8k", "run/model/gsm8k")),
            ("model-first", ("Apertus-1p5-8B-model/run/gsm8k", "model/run/gsm8k")),
        ):
            with self.subTest(layout=layout):
                for relative, score in zip(paths, (0.5, 0.9)):
                    self.write_result(str(Path(layout) / relative), score)
                args = [sys.executable, "scripts/verify_dashboard.py", "--runs-root", str(empty),
                        "--lm-eval-root", str(self.root / layout)]
                if layout == "model-first":
                    args += ["--lm-eval-layout", layout]
                result = subprocess.run(args, cwd=ROOT, env=self.env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn("COLLISION  key 'model'", result.stdout)

    def test_model_first_cli_preserves_base_curation_and_categories(self):
        self.write_result("lm-eval/Apertus-1p5-8B-model/run-1/gsm8k")
        manifest = self.root / "models.txt"
        manifest.write_text("curated-model|model=My model\n")
        empty = self.root / "lmms"
        empty.mkdir()
        out = self.root / "dashboard.html"
        self.run_command(sys.executable, "scripts/make_dashboard.py", "--runs-root", str(empty),
                         "--lm-eval-root", str(self.root / "lm-eval"), "--lm-eval-layout", "model-first",
                         "--models-file", str(manifest), "-o", str(out))
        html = out.read_text()
        payload = json.loads(html.split('<script id="data" type="application/json">', 1)[1].split("</script>", 1)[0])
        self.assertEqual(payload["models"], ["curated-model"])
        self.assertEqual(payload["labels"]["curated-model"], "My model")
        self.assertEqual(payload["table"][0]["framework"], "lm-evaluation-harness")
        self.assertIn("Medical", payload["categories"])
        self.assertEqual(payload["modality"]["Medical"], "text")
        self.assertEqual(payload["table"][0]["cells"]["curated-model"]["v"], 50.0)


if __name__ == "__main__":
    unittest.main()
