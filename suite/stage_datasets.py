"""Restore the local image trees that registry tasks declare, from their sources."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Callable

from suite import REPO_ROOT
from suite.fsutil import count_files
from suite.tasks import Task, load_registry

DEFAULT_ROOT = REPO_ROOT / "cache" / "rs_datasets"
def parse_source(source: str) -> tuple[str, str, str]:
    if source.startswith("hf://datasets/"):
        owner, repo, filename = source[len("hf://datasets/"):].split("/", 2)
        return "hf", f"{owner}/{repo}", filename
    if source.startswith("file://"):
        return "file", "", source[len("file://"):]
    raise ValueError(f"unsupported source {source!r}")


def default_downloader(source: str, dest_dir: Path) -> Path:
    kind, repo, filename = parse_source(source)
    if kind == "file":
        return Path(filename)
    from huggingface_hub import hf_hub_download
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    return Path(hf_hub_download(repo_id=repo, filename=filename, repo_type="dataset", local_dir=str(dest_dir)))


def _extract(archive: Path, kind: str, into: Path) -> None:
    into.mkdir(parents=True, exist_ok=True)
    if kind == "tar":
        with tarfile.open(archive) as tf:
            tf.extractall(into, filter="data")
    elif kind == "zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(into)
    elif kind == "none":
        shutil.copy2(archive, into / archive.name)
    else:
        raise ValueError(f"unknown extract kind {kind!r}")


def restore(task: Task, roots: dict[str, Path], downloader: Callable[[str, Path], Path] = default_downloader) -> list[Path]:
    restored = []
    for asset in task.assets:
        root = Path(roots[asset.env])
        target = root / asset.relative
        if target.is_dir() and count_files(target, asset.min_files) >= asset.min_files:
            continue
        downloads = root / "_downloads"
        downloads.mkdir(parents=True, exist_ok=True)
        archive = downloader(asset.source, downloads)
        _extract(archive, asset.extract, root)
        if count_files(target, asset.min_files) < asset.min_files:
            raise RuntimeError(f"{task.name}: {target} has fewer than {asset.min_files} files after extracting {archive}")
        restored.append(target)
    return restored


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Restore declared dataset assets")
    p.add_argument("--task", action="append", required=True)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    a = p.parse_args(argv)
    reg = load_registry()
    for name in a.task:
        task = reg.tasks[name]
        roots = {asset.env: Path(os.environ.get(asset.env) or Path(a.root) / asset.default_root) for asset in task.assets}
        restored = restore(task, roots)
        for path in restored:
            print(f"restored {name}: {path}")
        if not restored:
            print(f"{name}: assets already present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
