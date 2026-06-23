"""Shared headline metric selection for Apertus eval reports."""

from __future__ import annotations

from typing import Any


def metric_display_name(metric: str) -> str:
    return metric.split(",", 1)[0]


def is_main_metric(metric: str) -> bool:
    if metric == "alias":
        return False
    lowered = metric.lower()
    return not ("stderr" in lowered or "_clt" in lowered or "_clustered" in lowered)


# Task-specific headline metrics. Order matters: more specific task names must
# precede broader substring matches such as seedbench and 3dsrbench.
TASK_METRIC_PRIORITY: list[tuple[str, tuple[str, ...]]] = [
    # EASI spatial-intelligence metrics.
    ("3dsrbench_circular", ("circular_accuracy", "vanilla_accuracy")),
    ("3dsrbench", ("circular_accuracy", "vanilla_accuracy")),
    ("site_bench", ("chance_adjusted_acc", "accuracy")),
    ("mmsi_bench", ("average",)),
    ("viewspatial", ("overall_accuracy",)),
    ("embspatial", ("embspatial_acc",)),
    ("mindcube", ("overall_accuracy",)),
    ("sparbench", ("sparbench_score",)),
    ("omnispatial", ("omnispatial",)),
    ("erqa", ("erqa_acc",)),
    ("blink", ("blink_acc",)),
    ("cv_bench", ("cv_bench_acc",)),
    # Other locally important multi-metric benchmarks.
    ("seedbench_2_plus", ("seedbench_2_plus_all",)),
    ("seedbench", ("seed_image", "seed_all")),  # image-only headline (skip video dims)
    ("mmstar", ("average",)),
    ("refcoco", ("refcoco_ACC@0.5",)),
    ("vstar", ("vstar_overall_acc",)),
    ("mmvp", ("mmvp_accuracy",)),
    ("vlms_are_biased", ("accuracy_by_topic.topic_mean", "accuracy")),
    ("vlmsareblind", ("accuracy_by_task.task_mean", "accuracy")),
]

STRICT_TASK_METRIC_PRIORITY = {"refcoco"}

ADDITIONAL_TASK_METRICS: dict[str, tuple[tuple[str, str], ...]] = {
    # Multi-headline benchmarks: the primary metric comes from
    # TASK_METRIC_PRIORITY (pick_headline_metric); these are the *second*
    # headline reported alongside it via iter_headline_metrics.
    #   mme   -> total (primary) + perception/cognition breakdown (here)
    #   mmvp  -> per-question accuracy (primary) + pair accuracy (here)
    "mme": (("mme_perception", "mme_perception_score"), ("mme_cognition", "mme_cognition_score")),
    "mmvp": (("mmvp_pair", "mmvp_pair_accuracy"),),
}


GLOBAL_METRIC_PRIORITY: tuple[str, ...] = (
    "exact_match",
    "accuracy",
    "acc",
    "relaxed_overall",
    "anls",
    "ocrbench_accuracy",
    "ocrbench_v2_accuracy",
    "mmmu_acc",
    "visulogic_acc",
    "close_accuracy",
    "overall_accuracy",
    "mathvision_standard_eval",
    "std_eval",
    "omnidocbench_overall",
    "pope_accuracy",
    "refcoco_ACC@0.5",
    "muirbench_score_overall",
    "mathvista_acc",
    "llm_as_judge_eval",
)


def _numeric_metric(metrics: dict[str, Any], wanted: str) -> tuple[str, float] | None:
    wanted_metric = wanted
    nested_key = ""
    if wanted.startswith(("accuracy_by_task.", "accuracy_by_topic.")):
        wanted_metric, _, nested_key = wanted.partition(".")
    for metric, value in metrics.items():
        if not is_main_metric(metric):
            continue
        display_metric = metric_display_name(metric)
        if display_metric != wanted_metric:
            continue
        if nested_key:
            if isinstance(value, dict):
                if isinstance(value.get(nested_key), (int, float)):
                    return wanted, float(value[nested_key])
                if nested_key in {"task_mean", "topic_mean"}:
                    values = [
                        float(nested_value)
                        for nested_name, nested_value in value.items()
                        if nested_name not in {"overall", nested_key}
                        and isinstance(nested_value, (int, float))
                    ]
                    if values:
                        return wanted, sum(values) / len(values)
            continue
        if isinstance(value, (int, float)):
            return metric_display_name(metric), float(value)
    return None


def pick_headline_metric(task: str, metrics: dict[str, Any]) -> tuple[str | None, float | None]:
    """Pick the benchmark headline metric.

    This intentionally uses task-aware policy instead of raw JSON order. Many
    result files contain category breakdowns before their official overall
    metric, and EASI spatial benchmarks define specific headline metrics.
    """
    task_lower = task.lower()

    # MME headline = full score (perception + cognition). A perception-only
    # headline drops the reasoning half and undersells thinking checkpoints.
    if task_lower == "mme":
        perception = _numeric_metric(metrics, "mme_perception_score")
        cognition = _numeric_metric(metrics, "mme_cognition_score")
        if perception is not None and cognition is not None:
            return "mme_total_score", perception[1] + cognition[1]
        if perception is not None:
            return perception
        if cognition is not None:
            return cognition

    for task_pattern, preferred_metrics in TASK_METRIC_PRIORITY:
        if task_pattern not in task_lower:
            continue
        for wanted in preferred_metrics:
            selected = _numeric_metric(metrics, wanted)
            if selected is not None:
                return selected
        if task_pattern in STRICT_TASK_METRIC_PRIORITY:
            return None, None

    for wanted in GLOBAL_METRIC_PRIORITY:
        selected = _numeric_metric(metrics, wanted)
        if selected is not None:
            return selected

    for metric, value in metrics.items():
        if not is_main_metric(metric) or not isinstance(value, (int, float)):
            continue
        return metric_display_name(metric), float(value)
    return None, None


def iter_headline_metrics(task: str, metrics: dict[str, Any]) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    metric, value = pick_headline_metric(task, metrics)
    if metric is not None and value is not None:
        rows.append((task, metric, value))

    seen = {(task, metric)}
    for task_pattern, extra_metrics in ADDITIONAL_TASK_METRICS.items():
        if task.lower() != task_pattern:
            continue
        for extra_task, wanted in extra_metrics:
            selected = _numeric_metric(metrics, wanted)
            if selected is None:
                continue
            extra_metric, extra_value = selected
            key = (extra_task, extra_metric)
            if key not in seen:
                rows.append((extra_task, extra_metric, extra_value))
                seen.add(key)
    return rows


def normalize_score(metric: str, value: float) -> float | None:
    lowered = metric.lower()
    if "mme_total" in lowered:
        return value / 2800.0
    if "mme_perception" in lowered:
        return value / 2000.0
    if "mme_cognition" in lowered:
        return value / 800.0
    if any(key in lowered for key in ("mathvision", "omnidocbench", "site_bench", "chance_adjusted_acc")):
        return value / 100.0
    if value > 1.0:
        return value / 100.0
    if value < 0:
        return None
    return value
