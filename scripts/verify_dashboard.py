#!/usr/bin/env python3
"""Audit canonical-key collisions in the dashboard pipeline.

A collision is when more than one model directory maps to the same canonical
checkpoint key and its newest eligible scores disagree on a shared benchmark.
Use the dashboard collectors so status, units, task aliases and fallback agree.

Usage:
  python3 verify_dashboard.py --runs-root /path [--vlmeval-root /path]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_dashboard import audit_inventory, collect_inventory, parse_models_manifest


def audit(roots: list[Path], *, vlmeval_roots=(), vlmeval_results_roots=(), lm_eval_roots=(), manifest_roots=(),
          aliases=None, include_spatial=False, only_keys=None) -> int:
    def existing(paths):
        return [Path(path) for path in paths if Path(path).is_dir()]

    _manifests, collections = collect_inventory(
        existing(roots), vlmeval_roots=existing(vlmeval_roots), lm_eval_roots=existing(lm_eval_roots),
        vlmeval_results_roots=existing(vlmeval_results_roots),
        manifest_roots=manifest_roots, include_spatial=include_spatial)
    return audit_inventory(collections, aliases=aliases, only_keys=only_keys)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", required=True, type=Path, nargs="+")
    p.add_argument("--vlmeval-root", type=Path, nargs="+", default=[])
    p.add_argument("--vlmeval-results-root", type=Path, nargs="+", default=[],
                   help="suite run/model/benchmark sources; may join at most one --vlmeval-root")
    p.add_argument("--lm-eval-root", type=Path, nargs="+", default=[])
    p.add_argument("--models-file", type=Path)
    p.add_argument("--include-spatial", action="store_true")
    args = p.parse_args()
    if args.vlmeval_results_root and len(set(args.vlmeval_root)) > 1:
        p.error("suite result sources can join at most one shared VLMEvalKit root; audit independent shared roots separately")
    only_keys, aliases = None, {}
    if args.models_file:
        only_keys, _labels, aliases, _groups = parse_models_manifest(args.models_file)
    sys.exit(1 if audit(args.runs_root, vlmeval_roots=args.vlmeval_root, lm_eval_roots=args.lm_eval_root,
                        vlmeval_results_roots=args.vlmeval_results_root, aliases=aliases,
                        include_spatial=args.include_spatial, only_keys=only_keys) else 0)


if __name__ == "__main__":
    main()
