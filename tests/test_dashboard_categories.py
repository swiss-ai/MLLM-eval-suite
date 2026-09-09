"""Category compatibility captured before removing the dashboard taxonomy."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_dashboard as dashboard
from suite.tasks import Registry, load_registry


# All existing dashboard rows, including family inheritance and uncategorized
# rows. Keep literal expectations independent of either lookup implementation.
CATEGORY_TASKS = {
    "General VQA & Perception": "gqa mmbench_en_dev mme mme_cognition mme_perception mmerealworld mmstar mmvet realworldqa seedbench vqav2_val vstar_bench",
    "Robustness & Bias": "mmvp mmvp_pair vlms_are_biased vlmsareblind",
    "Spatial & Embodied": "3dsrbench cv_bench cv_bench_2d cv_bench_3d embspatial erqa mindcube mmsi_bench omnispatial omnispatial_manual_cot refspatial robospatial site_bench sparbench spatial_dise viewspatial vsibench_debiased",
    "Multi-Image": "blink muirbench",
    "Instruction Following": "mm_ifeval",
    "Counting & Grounding": "countbench pixmo_count refcoco",
    "Docs, Charts & OCR": "cc_ocr_doc_parsing cc_ocr_kie cc_ocr_multi_lan_ocr cc_ocr_multi_scene_ocr chartqa charxiv_descriptive charxiv_reasoning docvqa_val iconqa_val infovqa_val mtvqa ocrbench ocrbench_v2 omnidocbench seedbench_2_plus textvqa_val",
    "Math & Logic": "babyvision logicvista mathverse mathvision mathvista_mini visualpuzzles_direct visulogic",
    "STEM & Knowledge": "ai2d mmmu_pro_standard mmmu_pro_vision mmmu_val scienceqa",
    "Remote Sensing": "bigearth frieda geobench geobench_ref geobench_cap geobench_temporal geobench_single rsrcc vrsbench vrsbench_ref vrsbench_cap vrsbench_vqa",
    "Alignment": "hallusionbench mia_bench mm_safetybench pope",
    "Medical VQA": "medxpertqa_mm path_mmu path_mmu_test path_vqa pmc_vqa slake slake_en slake_zh vqa_rad",
    "Medical": "healthbench medmcqa medqa medxpertqa_text mmlu_medical pubmedqa",
    "Math (Text)": "aime24 aime25 gsm8k hmmt_feb_2025 math500_verify math_lvl5_verify",
    "Knowledge & Reasoning (Text)": "arc_challenge arc_easy gpqa_diamond_zeroshot hellaswag mmlu mmlu_pro truthfulqa_mc2 winogrande",
    "Instruction Following (Text)": "ifbench ifeval",
    None: "osworld screenspot screenspot_pro screenspot_v2 vsibench where2place",
}
PREFIX_CATEGORIES = {
    "3dsrbench": "Spatial & Embodied", "omnispatial": "Spatial & Embodied",
    "refcoco": "Counting & Grounding", "mathvision": "Math & Logic",
    "bigearth": "Remote Sensing", "geobench": "Remote Sensing",
    "rsrcc": "Remote Sensing", "vrsbench": "Remote Sensing",
    "path_mmu": "Medical VQA", "healthbench": "Medical",
}


@pytest.mark.parametrize("lookup", [load_registry().category, dashboard.category_for])
def test_all_current_category_assignments_and_case(lookup):
    for category, tasks in CATEGORY_TASKS.items():
        for task in tasks.split():
            assert lookup(task) == category, task
            assert lookup(task.upper()) == category, task.upper()
    for prefix, category in PREFIX_CATEGORIES.items():
        for suffix in ("_test", "_manual_cot"):
            assert lookup(prefix + suffix) == category
            assert lookup((prefix + suffix).upper()) == category
    assert lookup("unknown_task") is None


def test_category_exact_override_and_uncategorized_row_inheritance():
    registry = Registry(tasks={}, dashboard={
        "family": {"cat": "Family", "cat_prefix": True},
        "family_child": {"cat": "Child", "cat_prefix": True},
        "family_child_exact": {"cat": "Exact"},
        "family_child_inherited": {"headline": ("accuracy",)},
    })
    assert registry.category("FAMILY_CHILD_EXACT") == "Exact"
    assert registry.category("family_child_inherited") == "Family"
    assert registry.category("family_child_other") == "Family"
    # Category inheritance must not change metric-row lookup semantics.
    assert registry.dashboard_entry("family_child_inherited") == {"headline": ("accuracy",)}
