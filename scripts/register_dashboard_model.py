#!/usr/bin/env python3
"""Register models onto the dashboard without editing scripts.

    list                          canonical keys found in results, with cell
                                  counts, flagged registered/unregistered
    add <key> <label> [--alias K] [--group G]
                                  validate <key> against actual results, then
                                  insert into dashboard_models.txt under group G

Column order, labels and selector groups live in scripts/dashboard_models.txt;
this tool only inserts validated entries. Run scripts/refresh_dashboard.sh
afterwards.
"""
import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = HERE.parent
sys.path.insert(0, str(HERE))
from make_dashboard import canonical_model_key, parse_models_manifest  # noqa: E402

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
    found: dict[str, set] = {}
    for root in result_roots():
        for d in root.iterdir():
            if d.is_dir():
                found.setdefault(canonical_model_key(d.name), set()).add(d.name)
    return found


def registered() -> dict[str, str]:
    only, labels, aliases, _groups = parse_models_manifest(MANIFEST)
    label_of = dict(spec.split("=", 1) for spec in labels)
    reg = {key: label_of[key] for key in only}
    reg.update({alias: label_of[primary] for alias, primary in aliases.items()})
    return reg


def cmd_list(_args) -> int:
    reg = registered()
    found = scan()
    width = max((len(k) for k in found), default=20)
    for key in sorted(found):
        mark = f"registered as {reg[key]!r}" if key in reg else "UNREGISTERED"
        dirs = ", ".join(sorted(found[key])[:2])
        print(f"{key:{width}}  {mark}  ({dirs})")
    return 0


def group_headers(lines: list[str]) -> list[tuple[int, str]]:
    return [
        (i, line.split(":", 1)[1].strip())
        for i, line in enumerate(lines)
        if line.strip().lower().startswith("# group:")
    ]


def insert_entry(lines: list[str], entry: str, group: str | None) -> tuple[list[str], str]:
    """Place entry at the end of its group block, not the end of the file.

    Appending blindly lands the model in whichever group happens to be last,
    which silently files an Apertus checkpoint under Baselines.
    """
    headers = group_headers(lines)
    if not headers:
        lines.append(entry)
        return lines, "(ungrouped)"
    if group is None:
        start, name = headers[-1]
    else:
        match = [(i, n) for i, n in headers if n.lower() == group.lower()]
        if not match:
            raise KeyError(", ".join(n for _, n in headers))
        start, name = match[0]
    end = next((i for i, _ in headers if i > start), len(lines))
    while end - 1 > start and not lines[end - 1].strip():
        end -= 1
    lines.insert(end, entry)
    return lines, name


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
    entry = f"{keypart}={args.label}"
    lines = MANIFEST.read_text().splitlines()
    try:
        lines, group = insert_entry(lines, entry, args.group)
    except KeyError as exc:
        print(f"error: unknown --group {args.group!r}; manifest has: {exc.args[0]}", file=sys.stderr)
        return 2
    MANIFEST.write_text("\n".join(lines) + "\n")
    print(f"registered: {entry}")
    print(f"group: {group}" + ("" if args.group else "  (default: last group — pass --group to choose)"))
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
    pa.add_argument("--group", help="selector section to file it under (e.g. 'Apertus checkpoints', 'Baselines'); default is the last group in the manifest")
    args = p.parse_args()
    return cmd_list(args) if args.cmd == "list" else cmd_add(args)


if __name__ == "__main__":
    sys.exit(main())
