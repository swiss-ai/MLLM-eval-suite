#!/usr/bin/env python3
"""Stage xBD imagery into the GEOBench-VLM tree to restore skipped docs.

GEOBench-VLM does not redistribute its xBD (xView2) images: 210/3211
geobench_single and 1260/1713 geobench_temporal docs are skipped until they
are staged. The 630 required files (manifest alongside this script's output
tree) all come from the xView2 challenge *training* images archive.

Usage:
  1. Register at https://xview2.org and accept the challenge license.
  2. Download "Challenge training set" images and extract it anywhere.
  3. python3 scripts/stage_geobench_xbd.py --xbd-root /path/to/extracted/xbd

The geobench task filters are presence-based, so the next eval run picks the
restored docs up automatically; no code change needed.
"""

import argparse
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEOBENCH = os.path.join(REPO, "cache", "rs_datasets", "geobench")
MANIFEST = os.path.join(GEOBENCH, "xbd_needed_manifest.txt")


def index_xbd(root):
    files = {}
    for dirpath, _, names in os.walk(root):
        for n in names:
            if n.endswith(".png"):
                files.setdefault(n, os.path.join(dirpath, n))
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xbd-root", required=True, help="root of the extracted xView2 archive(s)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    needed = [l.strip() for l in open(MANIFEST) if l.strip()]
    available = index_xbd(args.xbd_root)

    staged, missing = 0, []
    for rel in needed:
        dst = os.path.join(GEOBENCH, rel)
        if os.path.exists(dst):
            continue
        original = os.path.basename(rel).removeprefix("xBD_")
        src = available.get(original)
        if src is None:
            missing.append(original)
            continue
        if not args.dry_run:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
        staged += 1

    print(f"staged {staged}/{len(needed)} files into {GEOBENCH}")
    if missing:
        print(f"NOT FOUND in --xbd-root ({len(missing)}), e.g. {missing[:3]}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
