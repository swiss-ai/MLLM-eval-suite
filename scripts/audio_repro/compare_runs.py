"""Compare two repeated lmms-eval runs: scores and per-sample output identity.

    python scripts/audio_repro/compare_runs.py RUN_DIR_A RUN_DIR_B

RUN_DIR_* are run directories such as
results/lmms-eval/Apertus-v1.5-8B/<run-id>. For every task folder present in
both, prints each scored subtask's primary metric in both runs, how many
generated outputs are byte-identical, and how many outputs are runaways
(more than 1.5x the reference word count plus 20 words).
"""
import argparse
import glob
import json
import os

PRIMARY = ("wer", "bleu", "accuracy", "exact_match")


def latest(pattern):
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def scores(run_dir, folder):
    f = latest(f"{run_dir}/{folder}/*/*_results.json")
    out = {}
    if not f:
        return out
    for task, res in json.load(open(f))["results"].items():
        for key, value in res.items():
            name = key.split(",")[0]
            if name in PRIMARY and isinstance(value, (int, float)):
                out[task] = (name, value)
                break
    return out


def outputs(run_dir, folder):
    out = {}
    for f in glob.glob(f"{run_dir}/{folder}/*/*_samples_*.jsonl"):
        task = os.path.basename(f).split("_samples_", 1)[1][: -len(".jsonl")]
        for line in open(f):
            sample = json.loads(line)
            resp = sample.get("filtered_resps", sample.get("resps"))
            while isinstance(resp, list):
                resp = resp[0] if resp else ""
            out[(task, sample["doc_id"])] = (str(resp), len(str(sample.get("target", "")).split()))
    return out


def runaways(outs):
    return sum(len(text.split()) > 1.5 * ref + 20 for text, ref in outs.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_a")
    parser.add_argument("run_b")
    args = parser.parse_args()

    folders = sorted({os.path.basename(p) for p in glob.glob(f"{args.run_a}/*") if os.path.isdir(p)}
                     & {os.path.basename(p) for p in glob.glob(f"{args.run_b}/*") if os.path.isdir(p)})
    print("| Task | Metric | A | B | identical outputs | runaways A/B |")
    print("|---|---|---|---|---|---|")
    for folder in folders:
        sa, sb = scores(args.run_a, folder), scores(args.run_b, folder)
        oa, ob = outputs(args.run_a, folder), outputs(args.run_b, folder)
        common = oa.keys() & ob.keys()
        same = sum(oa[k][0] == ob[k][0] for k in common)
        ident = f"{same}/{len(common)}" if common else ""
        loops = f"{runaways(oa)}/{runaways(ob)}" if common else ""
        tasks = sorted(sa.keys() | sb.keys()) or [folder]
        for i, task in enumerate(tasks):
            metric = (sa.get(task) or sb.get(task) or ("",))[0]
            fmt = lambda s: f"{s[task][1]:.4g}" if task in s else "—"
            print(f"| {task} | {metric} | {fmt(sa)} | {fmt(sb)} | {ident if i == 0 else ''} | {loops if i == 0 else ''} |")


if __name__ == "__main__":
    main()
