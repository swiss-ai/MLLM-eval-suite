import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from metric_selection import iter_headline_metrics, pick_headline_metric  # noqa: E402


def test_registry_headline_wins_over_json_order():
    metrics = {"coarse perception,none": 0.67, "average,none": 0.47, "average_stderr,none": 0.01}
    assert pick_headline_metric("mmstar", metrics, ("average",)) == ("average", 0.47)


def test_strict_headline_refuses_fallback():
    metrics = {"refcoco_ACC@0.1,none": 0.9}
    assert pick_headline_metric("refcoco_bbox_rec_val", metrics, ("refcoco_ACC@0.5",), strict=True) == (None, None)
    assert pick_headline_metric("refcoco_bbox_rec_val", {"refcoco_ACC@0.5,none": 0.5}, ("refcoco_ACC@0.5",), strict=True) == ("refcoco_ACC@0.5", 0.5)


def test_generic_fallback_and_secondary_rows():
    assert pick_headline_metric("gqa", {"alias": "gqa", "exact_match,none": 0.58}) == ("exact_match", 0.58)
    rows = iter_headline_metrics("mmvp", {"mmvp_accuracy,none": 0.7, "mmvp_pair_accuracy,none": 0.4}, ("mmvp_accuracy",))
    assert rows == [("mmvp", "mmvp_accuracy", 0.7), ("mmvp_pair", "mmvp_pair_accuracy", 0.4)]


def test_grader_failure_voids_score():
    assert pick_headline_metric("babyvision", {"babyvision_overall_accuracy,none": 0.3, "babyvision_grader_failure_rate,none": 0.5},
                                ("babyvision_overall_accuracy",)) == (None, None)
