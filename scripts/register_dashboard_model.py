#!/usr/bin/env python3
"""Register models onto the dashboard without editing scripts.

    list                          canonical keys found in results, with cell
                                  counts, flagged registered/unregistered
    add <key> <label> [--alias K] validate <key> against actual results, then
                                  append to dashboard_models.txt

Column order and labels live in scripts/dashboard_models.txt; this tool only
appends validated entries. Run scripts/refresh_dashboard.sh afterwards.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
sys.path.insert(0, str(HERE))
from make_dashboard import canonical_model_key  # noqa: E402

MANIFEST = HERE / "dashboard_models.txt"


def result_roots() -> list[Path]:
    roots = [
        Path(os.environ.get("RUNS_ROOT", "/capstor/store/cscs/swissai/infra01/users/xyixuan/apertus-1p5-eval/runs")),
        SUITE / "results" / "lmms-eval",
    ]
    vk = SUITE / "results" / "VLMEvalKit"
    if vk.is_dir():
        roots += [run for run in vk.iterdir() if run.is_dir()]
    outputs = Path(os.environ.get("VLMEVAL_OUTPUTS", "/capstor/store/cscs/swissai/infra01/vision-datasets/benchmark/VLMEval_Outputs"))
    roots.append(outputs)
    return [r for r in roots if r.is_dir()]


def scan() -> dict:
    found: dict[str, dict] = {}
    for root in result_roots():
        for d in root.iterdir():
            if not d.is_dir():
                continue
            key = canonical_model_key(d.name)
            entry = found.setdefault(key, {"dirs": set(), "cells": 0})
            entry["dirs"].add(d.name)
            entry["cells"] += sum(1 for _ in d.rglob("*results.json")) or sum(1 for _ in d.iterdir())
    return found


def registered() -> dict[str, str]:
    reg = {}
    for line in MANIFEST.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        keypart, label = line.split("=", 1)
        for k in keypart.split("|"):
            reg[k.strip()] = label.strip()
    return reg


def cmd_list(_args) -> int:
    reg = registered()
    found = scan()
    width = max((len(k) for k in found), default=20)
    for key in sorted(found):
        mark = f"registered as {reg[key]!r}" if key in reg else "UNREGISTERED"
        dirs = ", ".join(sorted(found[key]["dirs"])[:2])
        print(f"{key:{width}}  {found[key]['cells']:5d} artifacts  {mark}  ({dirs})")
    return 0


def cmd_add(args) -> int:
    reg = registered()
    found = scan()
    for k in [args.key, *args.alias]:
        if k in reg:
            print(f"error: {k!r} already registered as {reg[k]!r}", file=sys.stderr)
            return 2
        if k not in found:
            print(f"error: no results found for canonical key {k!r} — run `list` to see available keys", file=sys.stderr)
            return 2
    keypart = "|".join([args.key, *args.alias])
    with MANIFEST.open("a") as f:
        f.write(f"{keypart}={args.label}\n")
    print(f"registered: {keypart}={args.label}")
    print("now run: scripts/refresh_dashboard.sh")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="show canonical keys in results, registered or not")
    pa = sub.add_parser("add", help="validate and append a model to the manifest")
    pa.add_argument("key", help="canonical key (see `list`)")
    pa.add_argument("label", help="display label for the dashboard column")
    pa.add_argument("--alias", action="append", default=[], help="additional canonical keys to merge into this column")
    args = p.parse_args()
    return cmd_list(args) if args.cmd == "list" else cmd_add(args)


if __name__ == "__main__":
    sys.exit(main())
