"""Task registry: one declarative table for what the suite runs and reports."""
from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from suite import REPO_ROOT

REGISTRY_PATH = REPO_ROOT / "suite" / "tasks.toml"
FRAMEWORKS = ("lmms-eval", "VLMEvalKit", "lm-eval")
JUDGE_ENV = {"openai": "OPENAI_API_KEY"}
SUITE_LIST_FILES = {"lmms-eval": "task_suites/lmms-eval/visual_llm_judge.txt",
                    "VLMEvalKit": "task_suites/VLMEvalKit/llm_judge.txt"}
TASK_FIELDS = {"framework", "harness_task", "judge", "judge_env", "lmms_task", "assets",
               "max_model_len", "multi_image", "card", "report"}
ASSET_FIELDS = {"env", "relative", "source", "extract", "min_files"}
DASHBOARD_FIELDS = {"cat", "cat_prefix", "vk", "vk_prefix", "headline", "lm_metric"}


@dataclass(frozen=True)
class Asset:
    env: str
    relative: str
    source: str
    extract: str = "none"
    min_files: int = 1


@dataclass(frozen=True)
class Task:
    name: str
    framework: str
    harness_task: str
    judge: str | None = None
    judge_env: str | None = None
    lmms_task: str | None = None
    assets: tuple[Asset, ...] = ()
    max_model_len: int = 131072
    multi_image: bool = False
    card: bool = False
    report: bool = False


@dataclass
class Registry:
    tasks: dict[str, Task]
    dashboard: dict[str, dict]
    defaults: dict = field(default_factory=dict)

    def by_framework(self, framework: str) -> list[Task]:
        return [t for t in self.tasks.values() if t.framework == framework]

    def resolve(self, framework: str, harness_id: str) -> Task | None:
        """The task a launcher means by a harness-level id, or None."""
        for t in self.tasks.values():
            if t.framework == framework and t.harness_task == harness_id:
                return t
        if framework == "lmms-eval":
            for t in self.tasks.values():
                if t.lmms_task == harness_id:
                    return t
        return None

    def dashboard_entry(self, task_name: str) -> dict | None:
        """Exact dashboard row, else the family whose cat_prefix covers the task."""
        if task_name in self.dashboard:
            return self.dashboard[task_name]
        for key, entry in self.dashboard.items():
            if entry.get("cat_prefix") and task_name.startswith(key):
                return entry
        return None

    def category(self, task_name: str) -> str | None:
        entry = self.dashboard_entry(task_name)
        return entry.get("cat") if entry else None


def _task(name: str, raw: dict, defaults: dict) -> Task:
    unknown = set(raw) - TASK_FIELDS
    if unknown:
        raise ValueError(f"task {name!r}: unknown fields {sorted(unknown)}")
    for required in ("framework", "harness_task"):
        if required not in raw:
            raise ValueError(f"task {name!r}: missing {required}")
    if raw["framework"] not in FRAMEWORKS:
        raise ValueError(f"task {name!r}: framework must be one of {FRAMEWORKS}")
    assets = []
    for a in raw.get("assets", []):
        bad = set(a) - ASSET_FIELDS
        if bad:
            raise ValueError(f"task {name!r}: unknown asset fields {sorted(bad)}")
        assets.append(Asset(**a))
    judge = raw.get("judge")
    judge_env = raw.get("judge_env") or (JUDGE_ENV.get(judge) if judge else None)
    return Task(name=name, framework=raw["framework"], harness_task=raw["harness_task"], judge=judge,
                judge_env=judge_env, lmms_task=raw.get("lmms_task"), assets=tuple(assets),
                max_model_len=int(raw.get("max_model_len", defaults.get("max_model_len", 131072))),
                multi_image=bool(raw.get("multi_image", False)), card=bool(raw.get("card", False)),
                report=bool(raw.get("report", False)))


def _dashboard(name: str, raw: dict) -> dict:
    unknown = set(raw) - DASHBOARD_FIELDS
    if unknown:
        raise ValueError(f"dashboard {name!r}: unknown fields {sorted(unknown)}")
    entry = dict(raw)
    if "headline" in entry:
        entry["headline"] = tuple(entry["headline"])
    return entry


def load_registry(path: Path | None = None) -> Registry:
    path = Path(path) if path else REGISTRY_PATH
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    if data.get("version") != 1:
        raise ValueError(f"{path}: unsupported registry version {data.get('version')!r}")
    defaults = data.get("defaults", {})
    tasks = {name: _task(name, raw, defaults) for name, raw in data.get("tasks", {}).items()}
    dashboard = {name: _dashboard(name, raw) for name, raw in data.get("dashboard", {}).items()}
    return Registry(tasks=tasks, dashboard=dashboard, defaults=defaults)


def benchmarks_dict(reg: Registry) -> dict[str, dict]:
    return {k: dict(v) for k, v in reg.dashboard.items()}


def judge_tasks(reg: Registry, framework: str) -> list[str]:
    ids = {t.harness_task for t in reg.by_framework(framework) if t.judge}
    if framework == "lmms-eval":
        ids |= {t.lmms_task for t in reg.tasks.values() if t.judge and t.lmms_task}
    return sorted(ids)


def suite_list_text(reg: Registry, framework: str) -> str:
    lines = ["# Generated from suite/tasks.toml by `python3 -m suite.tasks --write-suite-lists`; do not edit."]
    lines += judge_tasks(reg, framework)
    return "\n".join(lines) + "\n"


def _read_list(path: Path) -> list[str]:
    return sorted(l.strip() for l in path.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#"))


def check_suite_lists(reg: Registry, repo_root: Path | None = None) -> list[str]:
    root = Path(repo_root) if repo_root else REPO_ROOT
    msgs = []
    for framework, rel in SUITE_LIST_FILES.items():
        path = root / rel
        want = judge_tasks(reg, framework)
        have = _read_list(path) if path.exists() else []
        if want != have:
            msgs.append(f"{rel}: registry says {want}, file has {have}")
    return msgs


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Task registry utilities")
    p.add_argument("--check", action="store_true", help="verify generated suite lists match the registry")
    p.add_argument("--write-suite-lists", action="store_true", help="regenerate judge suite lists from the registry")
    p.add_argument("--list", choices=["card", "report", "all"], help="print task names in a set")
    p.add_argument("--harness-to-name", action="store_true", help="print '<harness_task>\\t<name>' for --framework")
    p.add_argument("--framework", choices=FRAMEWORKS)
    a = p.parse_args(argv)
    reg = load_registry()
    if a.write_suite_lists:
        for framework, rel in SUITE_LIST_FILES.items():
            (REPO_ROOT / rel).write_text(suite_list_text(reg, framework))
            print(f"wrote {rel}")
    if a.check:
        msgs = check_suite_lists(reg)
        for m in msgs:
            print("MISMATCH", m)
        return 1 if msgs else 0
    if a.harness_to_name:
        for t in reg.tasks.values():
            if not a.framework or t.framework == a.framework:
                print(f"{t.harness_task}\t{t.name}")
    if a.list:
        for t in reg.tasks.values():
            if a.framework and t.framework != a.framework:
                continue
            if a.list == "all" or getattr(t, a.list):
                print(t.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
