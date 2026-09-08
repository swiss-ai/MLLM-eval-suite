import json
import textwrap
from pathlib import Path

import pytest

from suite.tasks import benchmarks_dict, check_suite_lists, judge_tasks, load_registry, suite_list_text

MINI = textwrap.dedent('''
    version = 1
    [defaults]
    max_model_len = 131072

    [dashboard.gqa]
    cat = "General VQA & Perception"
    [dashboard.mmsi_bench]
    cat = "Spatial & Embodied"
    vk = "MMSIBench_wo_circular"
    vk_prefix = true
    [dashboard.mathvision]
    cat = "Math & Logic"
    cat_prefix = true
    [dashboard.mm_safetybench]
    cat = "Alignment"
    vk = "MMSafetyBench"
    headline = ["safety_rate"]

    [tasks.gqa]
    framework = "lmms-eval"
    harness_task = "gqa"
    card = true
    report = true
    [tasks.mathvision_test]
    framework = "lmms-eval"
    harness_task = "mathvision_test"
    card = true
    [tasks.mmsi_bench]
    framework = "VLMEvalKit"
    harness_task = "MMSIBench_wo_circular"
    lmms_task = "mmsi_bench"
    multi_image = true
    max_model_len = 262144
    card = true
    [tasks.mathvista_mini]
    framework = "VLMEvalKit"
    harness_task = "MathVista_MINI"
    judge = "openai"
    [tasks.mathverse]
    framework = "VLMEvalKit"
    harness_task = "MathVerse_MINI"
    judge = "openai"
    lmms_task = "mathverse"
    [tasks.frieda]
    framework = "lmms-eval"
    harness_task = "frieda"
    card = true
    [[tasks.frieda.assets]]
    env = "FRIEDA_DIR"
    relative = "images"
    source = "hf://datasets/knowledge-computing/FRIEDA/images.tar"
    extract = "tar"
    min_files = 1000
''')


@pytest.fixture
def reg(tmp_path):
    p = tmp_path / "tasks.toml"
    p.write_text(MINI)
    return load_registry(p)


def test_loads_fields_and_defaults(reg):
    t = reg.tasks["gqa"]
    assert t.framework == "lmms-eval" and t.max_model_len == 131072 and t.card and t.report
    m = reg.tasks["mmsi_bench"]
    assert m.max_model_len == 262144 and m.multi_image and m.lmms_task == "mmsi_bench"
    a = reg.tasks["frieda"].assets[0]
    assert (a.env, a.relative, a.extract, a.min_files) == ("FRIEDA_DIR", "images", "tar", 1000)
    assert reg.tasks["mathvista_mini"].judge == "openai" and reg.tasks["mathvista_mini"].judge_env == "OPENAI_API_KEY"


def test_category_by_row_or_family(reg):
    assert reg.category("gqa") == "General VQA & Perception"
    assert reg.category("mathvision_test") == "Math & Logic"
    assert reg.category("nope") is None


def test_resolve_by_harness_id(reg):
    assert reg.resolve("VLMEvalKit", "MMSIBench_wo_circular").name == "mmsi_bench"
    assert reg.resolve("lmms-eval", "mmsi_bench").name == "mmsi_bench"
    assert reg.resolve("lmms-eval", "gqa").name == "gqa"
    assert reg.resolve("lmms-eval", "MMSIBench_wo_circular") is None


def test_benchmarks_dict_shape(reg):
    b = benchmarks_dict(reg)
    assert b["gqa"] == {"cat": "General VQA & Perception"}
    assert b["mmsi_bench"] == {"cat": "Spatial & Embodied", "vk": "MMSIBench_wo_circular", "vk_prefix": True}
    assert b["mm_safetybench"]["headline"] == ("safety_rate",)


def test_judge_tasks_and_suite_list(reg):
    assert judge_tasks(reg, "VLMEvalKit") == ["MathVerse_MINI", "MathVista_MINI"]
    assert judge_tasks(reg, "lmms-eval") == ["mathverse"]
    assert suite_list_text(reg, "VLMEvalKit").strip().splitlines()[-1] == "MathVista_MINI"


def test_check_suite_lists_reports_drift(reg, tmp_path):
    d = tmp_path / "task_suites" / "VLMEvalKit"
    d.mkdir(parents=True)
    (d / "llm_judge.txt").write_text("# judged\nMMVet\n")
    msgs = check_suite_lists(reg, tmp_path)
    assert any("llm_judge.txt" in m and "MathVista_MINI" in m for m in msgs)


def test_unknown_field_rejected(tmp_path):
    p = tmp_path / "t.toml"
    p.write_text('version = 1\n[tasks.x]\nframework = "lmms-eval"\nharness_task = "x"\nbogus = 1\n')
    with pytest.raises(ValueError):
        load_registry(p)


def test_real_registry_matches_legacy_dashboard_dict():
    reg = load_registry()
    legacy = json.loads((Path(__file__).parent / "fixtures" / "legacy_benchmarks.json").read_text())
    got = {k: {f: (tuple(v) if f == "headline" else v) for f, v in e.items()} for k, e in benchmarks_dict(reg).items()}
    want = {k: {f: (tuple(v) if f == "headline" else v) for f, v in e.items()} for k, e in legacy.items()}
    assert got == want


def test_real_registry_sets():
    reg = load_registry()
    assert len([t for t in reg.tasks.values() if t.card]) == 33
    assert len([t for t in reg.tasks.values() if t.report]) == 19
    assert reg.tasks["mmsi_bench"].framework == "VLMEvalKit" and reg.tasks["frieda"].assets
    assert all(reg.category(t.name) for t in reg.tasks.values() if t.card), "every card task needs a dashboard category"


def test_cli_max_model_len(capsys):
    from suite.tasks import main
    assert main(["--framework", "VLMEvalKit", "--max-model-len", "ViewSpatialBench"]) == 0
    assert capsys.readouterr().out.strip() == "262144"
    assert main(["--framework", "lmms-eval", "--max-model-len", "pope"]) == 0
    assert capsys.readouterr().out.strip() == "131072"
    assert main(["--framework", "lmms-eval", "--max-model-len", "not_a_task"]) == 0
    assert capsys.readouterr().out.strip() == "131072"
