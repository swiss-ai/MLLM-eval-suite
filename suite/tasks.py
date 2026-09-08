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
               "max_model_len", "multi_image", "card", "report", "result_tasks"}
ASSET_FIELDS = {"env", "relative", "source", "extract", "min_files"}
DASHBOARD_FIELDS = {"cat", "cat_prefix", "vk", "vk_prefix", "headline", "lmms_headline", "headline_strict", "lm_metric", "lm_metric_unit"}


@dataclass(frozen=True)
class Asset:
    env: str
    relative: str
    source: str
    extract: str = "none"
    min_files: int = 1

    @property
    def default_root(self) -> str:
        """Directory under the datasets root that the env var points at by default."""
        return self.env.lower().removesuffix("_dir")


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
    result_tasks: tuple[str, ...] = ()

    def harness_id_for(self, framework: str) -> str | None:
        """The id this task has on a harness: its own id on the owning harness, the lmms-eval alias elsewhere."""
        if framework == self.framework:
            return self.harness_task
        if framework == "lmms-eval":
            return self.lmms_task
        return None


@dataclass
class Registry:
    tasks: dict[str, Task]
    dashboard: dict[str, dict]
    defaults: dict = field(default_factory=dict)

    def __post_init__(self):
        self._by_harness_id: dict[tuple[str, str], Task] = {}
        for t in self.tasks.values():
            for fw in FRAMEWORKS:
                hid = t.harness_id_for(fw)
                if hid is None:
                    continue
                other = self._by_harness_id.get((fw, hid))
                if other is not None:
                    raise ValueError(f"tasks {other.name!r} and {t.name!r} both claim {hid!r} on {fw}")
                self._by_harness_id[(fw, hid)] = t
        self._prefix_rows = [(k, e) for k, e in self.dashboard.items() if e.get("cat_prefix")]

    def by_framework(self, framework: str) -> list[Task]:
        return [t for t in self.tasks.values() if t.framework == framework]

    def resolve(self, framework: str, harness_id: str) -> Task | None:
        """The task a launcher means by a harness-level id, or None."""
        return self._by_harness_id.get((framework, harness_id))

    def lookup(self, framework: str, name: str) -> Task | None:
        """A task by registry name or by its id on the given harness."""
        return self.tasks.get(name) or self.resolve(framework, name)

    def dashboard_entry(self, task_name: str) -> dict | None:
        """Exact dashboard row, else the family whose cat_prefix covers the task."""
        if task_name in self.dashboard:
            return self.dashboard[task_name]
        for key, entry in self._prefix_rows:
            if task_name.startswith(key):
                return entry
        return None

    def category(self, task_name: str) -> str | None:
        entry = self.dashboard_entry(task_name)
        return entry.get("cat") if entry else None

    def headline_for(self, framework: str, task_name: str) -> tuple[tuple[str, ...], bool]:
        """Headline metric names to try in order for a result of this task on this harness, and whether
        nothing else may stand in when none is present. A harness that only aliases the task uses the
        lmms_headline field, since metric names differ between harnesses."""
        task = self.lookup(framework, task_name)
        entry = self.dashboard_entry(task.name if task else task_name)
        if not entry:
            return (), False
        key = "lmms_headline" if task and task.framework != framework else "headline"
        return tuple(entry.get(key, ())), bool(entry.get("headline_strict"))


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
                report=bool(raw.get("report", False)), result_tasks=tuple(raw.get("result_tasks", ())))


def _dashboard(name: str, raw: dict) -> dict:
    unknown = set(raw) - DASHBOARD_FIELDS
    if unknown:
        raise ValueError(f"dashboard {name!r}: unknown fields {sorted(unknown)}")
    entry = dict(raw)
    if "lm_metric_unit" in entry and entry["lm_metric_unit"] not in {"fraction", "percent"}:
        raise ValueError(f"dashboard {name!r}: lm_metric_unit must be fraction or percent")
    for key in ("headline", "lmms_headline"):
        if key in entry:
            entry[key] = tuple(entry[key])
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
    return sorted({hid for t in reg.tasks.values() if t.judge and (hid := t.harness_id_for(framework))})


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
    p.add_argument("--max-model-len", metavar="TASK", help="print the context length TASK needs on --framework")
    p.add_argument("--harness-id", metavar="TASK", help="print the id --framework runs TASK under (TASK itself if unregistered)")
    a = p.parse_args(argv)
    reg = load_registry()
    if a.harness_id:
        task = reg.lookup(a.framework, a.harness_id) if a.framework else reg.tasks.get(a.harness_id)
        print((task.harness_id_for(a.framework) if task and a.framework else None) or a.harness_id)
        return 0
    if a.max_model_len:
        task = reg.lookup(a.framework, a.max_model_len) if a.framework else reg.tasks.get(a.max_model_len)
        print(task.max_model_len if task else reg.defaults.get("max_model_len", 131072))
        return 0
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
