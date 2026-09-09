import io
import tarfile
import zipfile
from pathlib import Path

from suite.stage_datasets import parse_source, restore
from suite.tasks import Asset, Task


def _tar(dest: Path, names):
    with tarfile.open(dest, "w") as tf:
        for n in names:
            info = tarfile.TarInfo(n)
            info.size = 1
            tf.addfile(info, io.BytesIO(b"x"))


def test_parse_source():
    assert parse_source("hf://datasets/knowledge-computing/FRIEDA/images.tar") == ("hf", "knowledge-computing/FRIEDA", "images.tar")
    assert parse_source("hf://datasets/aialliance/GEOBench-VLM/Single.zip") == ("hf", "aialliance/GEOBench-VLM", "Single.zip")
    assert parse_source("file:///tmp/x.zip") == ("file", "", "/tmp/x.zip")


def test_restore_extracts_tar_into_env_root(tmp_path):
    archive = tmp_path / "images.tar"
    _tar(archive, ["images/a.png", "images/b.png"])
    task = Task(name="frieda", framework="lmms-eval", harness_task="frieda",
                assets=(Asset(env="FRIEDA_DIR", relative="images", source=f"file://{archive}", extract="tar", min_files=2),))
    root = tmp_path / "rs" / "frieda"
    done = restore(task, {"FRIEDA_DIR": root}, downloader=lambda src, dest: Path(src.removeprefix("file://")))
    assert done == [root / "images"]
    assert sorted(p.name for p in (root / "images").iterdir()) == ["a.png", "b.png"]


def test_restore_zip_and_skips_when_present(tmp_path):
    archive = tmp_path / "Images_val.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("Images_val/P0001.png", "x")
    task = Task(name="vrsbench_vqa", framework="lmms-eval", harness_task="vrsbench_vqa",
                assets=(Asset(env="VRSBENCH_DIR", relative="Images_val", source=f"file://{archive}", extract="zip", min_files=1),))
    root = tmp_path / "vrs"
    calls = []

    def dl(src, dest):
        calls.append(src)
        return Path(src.removeprefix("file://"))

    assert restore(task, {"VRSBENCH_DIR": root}, downloader=dl) == [root / "Images_val"]
    assert restore(task, {"VRSBENCH_DIR": root}, downloader=dl) == []
    assert len(calls) == 1
