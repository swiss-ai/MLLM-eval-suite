"""Shared headline metric selection for Apertus eval reports."""

from __future__ import annotations

from typing import Any
import math


def metric_display_name(metric: str) -> str:
    return metric.split(",", 1)[0]


def is_main_metric(metric: str) -> bool:
    if metric == "alias":
        return False
    lowered = metric.lower()
    return not ("stderr" in lowered or "_clt" in lowered or "_clustered" in lowered)


# Fraction of failed grader calls above which a graded score is void.
MAX_GRADER_FAILURE = 0.05


# Which metric is a benchmark's headline is declared in suite/tasks.toml
# ([dashboard.<task>].headline, lmms_headline for a harness that only aliases
# the task, headline_strict to forbid fallbacks); callers pass it in. What stays
# here is computation and fallback: the MME total, the secondary rows below, and
# the generic priority for tasks the registry does not describe.
ADDITIONAL_TASK_METRICS: dict[str, tuple[tuple[str, str], ...]] = {
    # Multi-headline benchmarks: the primary metric comes from the registry
    # (pick_headline_metric); these are the *second*
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
    # A wanted name carrying a filter ("exact_match,flexible-extract") pins that
    # filter; otherwise filters share a display name and the winner would be
    # whichever the harness happened to serialize first.
    if "," in wanted_metric:
        for metric, value in metrics.items():
            if metric == wanted_metric and isinstance(value, (int, float)):
                return metric_display_name(metric), float(value)
        return None
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


def pick_headline_metric(task: str, metrics: dict[str, Any], headline: tuple[str, ...] = (),
                         strict: bool = False) -> tuple[str | None, float | None]:
    """Pick the benchmark headline metric.

    `headline` is the registry's ordered list of metric names for this task;
    with `strict`, no other metric may stand in when none of them is present.
    Raw JSON order is never trusted: many result files list category breakdowns
    before their official overall metric.
    """
    task_lower = task.lower()

    # A graded score is meaningless when the grader itself failed; same
    # contract as the VLMEvalKit judge-failure guard in derive_vlmeval_acc.
    for key, value in metrics.items():
        if (metric_display_name(key).endswith("grader_failure_rate") and is_main_metric(key)
                and isinstance(value, (int, float)) and value > MAX_GRADER_FAILURE):
            return None, None

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

    for wanted in headline:
        selected = _numeric_metric(metrics, wanted)
        if selected is not None:
            return selected
    if strict:
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


def iter_headline_metrics(task: str, metrics: dict[str, Any], headline: tuple[str, ...] = (),
                          strict: bool = False) -> list[tuple[str, str, float]]:
    rows: list[tuple[str, str, float]] = []
    metric, value = pick_headline_metric(task, metrics, headline, strict)
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
    if not math.isfinite(value):
        return None
    lowered = metric.lower()
    if "cider" in lowered:
        return value
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
