import os
import textwrap

from suite.preflight import EXIT_PREFLIGHT, run_checks
from suite.tasks import load_registry

REG = textwrap.dedent('''
    version = 1
    [tasks.pope]
    framework = "lmms-eval"
    harness_task = "pope"
    [tasks.mathvista_mini]
    framework = "VLMEvalKit"
    harness_task = "MathVista_MINI"
    judge = "openai"
    [tasks.frieda]
    framework = "lmms-eval"
    harness_task = "frieda"
    [[tasks.frieda.assets]]
    env = "FRIEDA_DIR"
    relative = "images"
    source = "hf://datasets/x/y/images.tar"
    extract = "tar"
    min_files = 3
    [tasks.viewspatial]
    framework = "VLMEvalKit"
    harness_task = "ViewSpatialBench"
    max_model_len = 262144
    [tasks.blink]
    framework = "VLMEvalKit"
    harness_task = "BLINK"
    lmms_task = "blink"
''')


def _setup(tmp_path):
    (tmp_path / "tasks.toml").write_text(REG)
    m = tmp_path / "model"
    m.mkdir()
    (m / "config.json").write_text("{}")
    (m / "model-00001-of-00001.safetensors").write_bytes(b"\0" * 8)
    t = tmp_path / "tok"
    t.mkdir()
    (t / "tokenizer.json").write_text("{}")
    (t / "chat_template.jinja").write_text("x")
    v = tmp_path / "vq"
    v.mkdir()
    (v / "config.yaml").write_text("a")
    (v / "model.ckpt").write_bytes(b"\0")
    img = tmp_path / "image.sqsh"
    img.write_bytes(b"\0")
    fr = tmp_path / "frieda"
    (fr / "images").mkdir(parents=True)
    for i in range(3):
        (fr / "images" / f"{i}.png").write_bytes(b"\0")
    return load_registry(tmp_path / "tasks.toml"), m, t, v, img, fr


def _failed(checks):
    return sorted(c.name for c in checks if not c.ok)


def _run(reg, m, t, v, img, framework, tasks, env, max_model_len=131072):
    return run_checks(registry=reg, framework=framework, model_path=m, tasks=tasks, thinking=False, tokenizer_path=t,
                      vision_tokenizer_dir=v, container_image=img, env=env, max_model_len=max_model_len)


def test_all_good(tmp_path):
    reg, m, t, v, img, fr = _setup(tmp_path)
    assert _failed(_run(reg, m, t, v, img, "lmms-eval", ["pope", "frieda"], {"FRIEDA_DIR": str(fr)})) == []


def test_dangling_weight_symlink(tmp_path):
    reg, m, t, v, img, fr = _setup(tmp_path)
    (m / "model-00001-of-00001.safetensors").unlink()
    os.symlink(tmp_path / "gone", m / "model-00001-of-00001.safetensors")
    assert "model:weights" in _failed(_run(reg, m, t, v, img, "lmms-eval", ["pope"], {}))


def test_weight_index_references_missing_shard(tmp_path):
    reg, m, t, v, img, fr = _setup(tmp_path)
    (m / "model.safetensors.index.json").write_text(
        '{"weight_map": {"layer.weight": "model-00002-of-00002.safetensors"}}'
    )
    assert "model:weights" in _failed(_run(reg, m, t, v, img, "lmms-eval", ["pope"], {}))


def test_empty_weight_index(tmp_path):
    reg, m, t, v, img, fr = _setup(tmp_path)
    (m / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    assert "model:weights" in _failed(_run(reg, m, t, v, img, "lmms-eval", ["pope"], {}))


def test_dangling_weight_index(tmp_path):
    reg, m, t, v, img, fr = _setup(tmp_path)
    (m / "model.safetensors.index.json").symlink_to(tmp_path / "missing-index")
    assert "model:weights" in _failed(_run(reg, m, t, v, img, "lmms-eval", ["pope"], {}))


def test_missing_asset_judge_and_context(tmp_path):
    reg, m, t, v, img, fr = _setup(tmp_path)
    assert "task:frieda:assets" in _failed(_run(reg, m, t, v, img, "lmms-eval", ["frieda"], {"FRIEDA_DIR": str(tmp_path / "nowhere")}))
    failed = _failed(_run(reg, m, t, v, img, "VLMEvalKit", ["MathVista_MINI", "ViewSpatialBench"], {}))
    assert "task:MathVista_MINI:judge" in failed and "task:ViewSpatialBench:context" in failed
    assert "task:MathVista_MINI:judge" not in _failed(_run(reg, m, t, v, img, "VLMEvalKit", ["MathVista_MINI"], {"OPENAI_API_KEY": "k"}))
    assert "task:ViewSpatialBench:context" not in _failed(_run(reg, m, t, v, img, "VLMEvalKit", ["ViewSpatialBench"], {}, max_model_len=262144))


def test_unknown_task_wrong_framework_and_alias(tmp_path):
    reg, m, t, v, img, fr = _setup(tmp_path)
    failed = _failed(_run(reg, m, t, v, img, "lmms-eval", ["nope", "mathvista_mini", "blink"], {}))
    assert "task:nope:registry" in failed and "task:mathvista_mini:framework" in failed
    assert not any(n.startswith("task:blink") for n in failed)


def test_cli_exit_code(tmp_path):
    from suite.preflight import main
    reg, m, t, v, img, fr = _setup(tmp_path)
    base = ["--registry", str(tmp_path / "tasks.toml"), "--framework", "lmms-eval", "--tasks", "pope",
            "--tokenizer", str(t), "--vision-tokenizer", str(v), "--container-image", str(img)]
    assert main(base + ["--model", str(m)]) == 0
    assert main(base + ["--model", str(tmp_path / "missing")]) == EXIT_PREFLIGHT
    assert main(["--registry", str(tmp_path / "tasks.toml"), "--framework", "VLMEvalKit", "--tasks", "BLINK",
                 "--model", "Qwen3-VL", "--skip-model", "--container-image", str(img), "--max-model-len", "262144"]) == 0


def _harness(tmp_path, decls):
    root = tmp_path / "harness"
    for rel, text in decls.items():
        f = root / "lmms_eval" / "tasks" / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return root


def test_task_declared_twice_fails(tmp_path):
    _setup(tmp_path)
    registry = load_registry(tmp_path / "tasks.toml")
    root = _harness(tmp_path, {
        "pope/pope.yaml": "task: pope\n",
        "other/pope.yaml": "group: pope\ntask:\n  - pope_en\n",
        "frieda/frieda.yaml": "task: frieda\n",
    })
    from suite.preflight import check_task_names
    by_name = {c.name: c for c in check_task_names(registry, "lmms-eval", ["pope", "frieda"], root)}
    assert not by_name["task:pope:definition"].ok and "2 times" in by_name["task:pope:definition"].detail
    assert by_name["task:frieda:definition"].ok
    assert check_task_names(registry, "VLMEvalKit", ["blink"], root) == []


def test_task_declared_nowhere_fails(tmp_path):
    _setup(tmp_path)
    registry = load_registry(tmp_path / "tasks.toml")
    root = _harness(tmp_path, {"frieda/frieda.yaml": "task: frieda\n"})
    from suite.preflight import check_task_names
    (check,) = check_task_names(registry, "lmms-eval", ["pope"], root)
    assert not check.ok and "nowhere" in check.detail
