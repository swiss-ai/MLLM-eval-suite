"""Refuse work that cannot succeed, before it costs a node."""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from suite.tasks import Registry, load_registry

EXIT_PREFLIGHT = 2


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str = ""


def _readable_file(path: Path) -> bool:
    try:
        real = Path(path).resolve()
        return real.is_file() and os.access(real, os.R_OK)
    except OSError:
        return False


def _count_files(path: Path, limit: int) -> int:
    n = 0
    for _root, _dirs, files in os.walk(path):
        n += len(files)
        if n >= limit:
            return n
    return n


def check_model(model_path: Path) -> list[Check]:
    model_path = Path(model_path)
    out = [Check("model:dir", model_path.is_dir(), str(model_path))]
    if not model_path.is_dir():
        return out
    out.append(Check("model:config", _readable_file(model_path / "config.json"), "config.json"))
    shards = sorted(model_path.glob("*.safetensors"))
    bad = [s.name for s in shards if not _readable_file(s) or s.resolve().stat().st_size == 0]
    out.append(Check("model:weights", bool(shards) and not bad,
                     f"{len(shards)} shards" + (f", unreadable: {bad}" if bad else "")))
    return out


def check_tokenizer(tokenizer_path: Path | None) -> list[Check]:
    if tokenizer_path is None:
        return [Check("tokenizer", False, "no tokenizer path")]
    tokenizer_path = Path(tokenizer_path)
    return [Check("tokenizer:json", _readable_file(tokenizer_path / "tokenizer.json"), str(tokenizer_path)),
            Check("tokenizer:template", _readable_file(tokenizer_path / "chat_template.jinja"), "chat_template.jinja")]


def check_vision_tokenizer(vq_dir: Path | None) -> list[Check]:
    if vq_dir is None:
        return [Check("vision_tokenizer", False, "no vision tokenizer dir")]
    vq_dir = Path(vq_dir)
    ok = _readable_file(vq_dir / "config.yaml") and _readable_file(vq_dir / "model.ckpt")
    return [Check("vision_tokenizer", ok, str(vq_dir))]


def check_container(image: Path | None) -> list[Check]:
    return [Check("container:image", image is not None and _readable_file(Path(image)), str(image))]


def check_tasks(registry: Registry, framework: str, tasks: list[str], env: dict, max_model_len: int) -> list[Check]:
    out = []
    for name in tasks:
        task = registry.tasks.get(name) or registry.resolve(framework, name)
        if task is None:
            out.append(Check(f"task:{name}:registry", False, "not in suite/tasks.toml"))
            continue
        if task.framework != framework and not (framework == "lmms-eval" and task.lmms_task == name):
            out.append(Check(f"task:{name}:framework", False, f"registered for {task.framework}, launched on {framework}"))
        for asset in task.assets:
            base = env.get(asset.env, "")
            path = Path(base) / asset.relative if base else None
            n = _count_files(path, asset.min_files) if path and path.is_dir() else 0
            out.append(Check(f"task:{name}:assets", n >= asset.min_files,
                             f"{asset.env}={base or '<unset>'} {asset.relative}: {n} files, need {asset.min_files}; "
                             f"restore with `python3 -m suite.stage_datasets --task {task.name}`"))
        if task.judge:
            out.append(Check(f"task:{name}:judge", bool(env.get(task.judge_env or "")), f"{task.judge} judge needs {task.judge_env}"))
        if task.max_model_len > max_model_len:
            out.append(Check(f"task:{name}:context", False, f"needs max_model_len {task.max_model_len}, launch has {max_model_len}"))
    return out


def run_checks(*, registry: Registry, framework: str, model_path: Path, tasks: list[str], thinking: bool,
               tokenizer_path: Path | None, vision_tokenizer_dir: Path | None, container_image: Path | None,
               env: dict, max_model_len: int) -> list[Check]:
    checks = check_model(model_path) + check_tokenizer(tokenizer_path) + check_container(container_image)
    if framework in ("lmms-eval", "VLMEvalKit"):
        checks += check_vision_tokenizer(vision_tokenizer_dir)
    checks += check_tasks(registry, framework, tasks, env, max_model_len)
    return checks


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Preflight checks for an evaluation launch")
    p.add_argument("--registry")
    p.add_argument("--framework", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tasks", required=True, help="comma-separated task ids as the launcher names them")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--tokenizer")
    p.add_argument("--vision-tokenizer")
    p.add_argument("--container-image")
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--skip-model", action="store_true", help="foreign model served by name; no local checkpoint to check")
    a = p.parse_args(argv)
    registry = load_registry(a.registry) if a.registry else load_registry()
    tasks = [t for t in a.tasks.replace(" ", ",").split(",") if t]
    if a.skip_model:
        checks = check_container(Path(a.container_image) if a.container_image else None)
        checks += check_tasks(registry, a.framework, tasks, dict(os.environ), a.max_model_len)
    else:
        checks = run_checks(registry=registry, framework=a.framework, model_path=Path(a.model), tasks=tasks,
                            thinking=a.thinking, tokenizer_path=Path(a.tokenizer) if a.tokenizer else None,
                            vision_tokenizer_dir=Path(a.vision_tokenizer) if a.vision_tokenizer else None,
                            container_image=Path(a.container_image) if a.container_image else None,
                            env=dict(os.environ), max_model_len=a.max_model_len)
    failed = [c for c in checks if not c.ok]
    for c in checks:
        print(f"{'ok  ' if c.ok else 'FAIL'} {c.name}: {c.detail}")
    print(f"preflight: {len(checks) - len(failed)} ok, {len(failed)} failed")
    return EXIT_PREFLIGHT if failed else 0


if __name__ == "__main__":
    sys.exit(main())
