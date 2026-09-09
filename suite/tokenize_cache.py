"""Fill the image-token cache for tasks without loading the language model.

Runs inside the evaluation container with the same cache environment the
lmms-eval job uses, so a later large-model job finds every image already
encoded and can run with the cache read-only.
"""
from __future__ import annotations

import argparse
import sys
import time

from PIL import Image

from suite import REPO_ROOT


def _images(visuals):
    out = []
    for v in visuals or []:
        if isinstance(v, Image.Image):
            out.append(v.convert("RGB"))
        elif isinstance(v, str):
            out.append(Image.open(v).convert("RGB"))
    return out


def iter_task_docs(task_name: str):
    from lmms_eval.tasks import TaskManager, get_task_dict

    manager = TaskManager()
    task_dict = get_task_dict([task_name], manager)
    for name, task in task_dict.items():
        if isinstance(task, dict):
            for sub_name, sub in task.items():
                if sub is None:
                    continue
                yield from _docs_of(sub_name, sub)
        elif task is not None:
            yield from _docs_of(name, task)


def _docs_of(name, task):
    if task.has_test_docs():
        docs = task.test_docs()
    elif task.has_validation_docs():
        docs = task.validation_docs()
    else:
        return
    for i, doc in enumerate(docs):
        yield name, i, task, doc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Tokenize-only image cache fill")
    p.add_argument("--tasks", required=True, help="comma-separated lmms-eval task names")
    p.add_argument("--shard", default="0/1", help="i/n: process docs with index % n == i")
    p.add_argument("--tokenizer", required=True, help="Apertus tokenizer directory")
    a = p.parse_args(argv)
    shard, nshards = (int(x) for x in a.shard.split("/"))
    sys.path.insert(0, str(REPO_ROOT / "shared"))
    from apertus_image_tokenizer import _shared_state
    from transformers import AutoTokenizer

    image_tokenizer, mm_kwargs = _shared_state()
    tokenizer = AutoTokenizer.from_pretrained(a.tokenizer, trust_remote_code=False)
    for task_name in [t for t in a.tasks.split(",") if t]:
        t0, n_docs, n_images = time.time(), 0, 0
        for sub_name, i, task, doc in iter_task_docs(task_name):
            if i % nshards != shard:
                continue
            images = _images(task.doc_to_visual(doc))
            if images:
                image_tokenizer.encode_images(images, tokenizer=tokenizer, mm_processor_kwargs=mm_kwargs)
            n_docs += 1
            n_images += len(images)
        print(f"tokenize_cache: task={task_name} shard={a.shard} docs={n_docs} images={n_images} seconds={time.time() - t0:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
