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
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_dashboard import (canonical_model_key, collect, collect_lm_eval,
                            collect_vlmeval, parse_models_manifest)
from suite.coverage import Manifests


def audit(roots: list[Path], *, vlmeval_roots=(), lm_eval_roots=(), manifest_roots=(),
          aliases=None, include_spatial=False, only_keys=None) -> int:
    lanes = [(collect, roots), (collect_vlmeval, vlmeval_roots), (collect_lm_eval, lm_eval_roots)]
    all_roots = [root for _collector, lane_roots in lanes for root in lane_roots]
    manifests = Manifests(all_roots + list(manifest_roots))
    keep = set(only_keys) if only_keys else None
    by_key = defaultdict(lambda: defaultdict(dict))
    for collector, lane_roots in lanes:
        for root in sorted({Path(root).resolve() for root in lane_roots if Path(root).is_dir()}):
            kwargs = {"include_spatial": include_spatial} if collector is collect else {}
            # Preserve each directory until after collection, so canonical or
            # user-defined aliases cannot hide conflicting candidate identities.
            _models, rows = collector(root, None, manifests, model_key=lambda name: name, **kwargs)
            for row in rows:
                for model, cell in row["cells"].items():
                    key = canonical_model_key(model)
                    key = (aliases or {}).get(key, key)
                    if keep is not None and key not in keep:
                        continue
                    by_key[key][str((root / model).resolve())][(row["task"], row["metric"])] = cell["v"]

    bad = 0
    for key, vals in sorted(by_key.items()):
        if len(vals) < 2:
            continue
        conflicts = []
        for t in sorted(set().union(*(v.keys() for v in vals.values()))):
            present = {n: v[t] for n, v in vals.items() if t in v}
            if len({round(x, 3) for x in present.values()}) > 1:
                conflicts.append((t, present))
        if conflicts:
            bad += 1
            print(f"COLLISION  key '{key}'  <-  {len(vals)} dirs:")
            for directory in vals:
                print(f"             {directory}")
            print(f"           {len(conflicts)} conflicting benchmark(s), e.g.:")
            for (task, metric), present in conflicts[:3]:
                shown = ", ".join(f"{n[-28:]}={x:.1f}" for n, x in present.items())
                print(f"             {task}: {shown}")
    print(f"\n{'PASS — no contaminating collisions' if not bad else f'FAIL — {bad} colliding key(s)'}")
    return bad


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-root", required=True, type=Path, nargs="+")
    p.add_argument("--vlmeval-root", type=Path, nargs="+", default=[])
    p.add_argument("--vlmeval-results-root", type=Path, nargs="+", default=[])
    p.add_argument("--lm-eval-root", type=Path, nargs="+", default=[])
    p.add_argument("--models-file", type=Path)
    p.add_argument("--include-spatial", action="store_true")
    args = p.parse_args()
    only_keys, aliases = None, {}
    if args.models_file:
        only_keys, _labels, aliases, _groups = parse_models_manifest(args.models_file)
    sys.exit(1 if audit(args.runs_root, vlmeval_roots=args.vlmeval_root, lm_eval_roots=args.lm_eval_root,
                        manifest_roots=args.vlmeval_results_root, aliases=aliases,
                        include_spatial=args.include_spatial, only_keys=only_keys) else 0)


if __name__ == "__main__":
    main()
