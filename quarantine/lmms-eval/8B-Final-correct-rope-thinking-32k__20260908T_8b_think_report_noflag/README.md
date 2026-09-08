Apertus 1.5 8B thinking sweep, run id 20260908T_8b_think_report (2026-09-08) — DO NOT PUBLISH.

Launched with --thinking before the apertus_1p5_vllm wrapper fix: the base
constructor reset enable_thinking, so every job ran without deliberation while
being labelled as the thinking column (run_meta.json: enable_thinking=None, no
"thinking canary passed" line). The results are plain 8B numbers under a
thinking label. The rerun with the fixed wrapper landed under the same run id
in results/lmms-eval/8B-Final-correct-rope-thinking-32k/ and is the valid set.
