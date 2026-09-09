# EASI Spatial Intelligence Comparison

> Snapshot of 2026-07-02, default-prompt protocol. OmniSpatial has since moved to EASI's
> manual-CoT prompt on the dashboard (`omnispatial_manual_cot`); numbers here predate that.

This table compares Apertus against public
[EASI leaderboard](https://easi.lmms-lab.com/leaderboard/) rows on the five
spatial-intelligence benchmarks:

- `MMSI-Bench`
- `MindCube`
- `OmniSpatial`
- `SPAR-Bench`
- `ViewSpatialBench`


Scores are percentages. `avg5` is the unweighted mean of the five benchmarks
above. Public EASI snapshot: `lastUpdated=2026-05-14T17:33:41Z`.

## Headline

| model | MMSI | MindCube | OmniSpatial | SPAR | ViewSpatial | avg5 | rank note |
|---|---:|---:|---:|---:|---:|---:|---|
| Apertus-1.5-8B 256k-3600 online-DPO | 32.00 | 47.50 | 38.88 | 42.33 | 43.28 | **40.80** | best local run; would rank about **15 / 35** on avg5 |
| Apertus-1.5-8B SFT 256k-3600 | 31.80 | 46.06 | 37.70 | 41.11 | 41.89 | 39.71 | strong SFT baseline; would rank about **15 / 35** on avg5 |
| Apertus-1.5-8B 256k-4200 online-DPO | 31.40 | 46.54 | 37.77 | 41.28 | 41.63 | 39.72 | roughly tied with 3600 SFT |
| Apertus-1.5-8B SFT 256k-4200 | 33.70 | 46.35 | 31.31 | 39.93 | 40.04 | 38.27 | best local MMSI; lower avg5 due to OmniSpatial/ViewSpatial |

## Well-Known Models Behind Apertus

Rows with all five benchmarks use `avg5`. Rows marked `partial` are missing
OmniSpatial and SPAR on the public EASI leaderboard, so their comparison uses
the common three-benchmark average over MMSI, MindCube, and ViewSpatial.
For full rows, the Apertus reference is the SFT run,
`Apertus-1.5-8B SFT 256k-3600`, with `avg5=39.71`. For partial rows,
the Apertus reference is the same model over the common MMSI/MindCube/View set,
with `avg3=39.92`.

| model | coverage | MMSI | MindCube | OmniSpatial | SPAR | ViewSpatial | model avg | Apertus reference avg | Apertus lead | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Apertus-1.5-8B SFT 256k-3600 | local full 5/5 | 31.80 | 46.06 | 37.70 | 41.11 | 41.89 | **39.71** | **39.71** | reference | SFT reference run. |
| Apertus-1.5-8B 256k-3600 online-DPO | local full 5/5 | 32.00 | 47.50 | 38.88 | 42.33 | 43.28 | 40.80 | **39.71** | DPO +1.08 vs SFT | Online-DPO improves the 3600 SFT baseline by about 1.08 avg5. |
| Apertus-1.5-8B 256k-4200 online-DPO | local full 5/5 | 31.40 | 46.54 | 37.77 | 41.28 | 41.63 | 39.72 | **39.71** | DPO +0.01 vs 3600 SFT | 4200 online-DPO is essentially tied with the 3600 SFT reference on avg5. |
| Apertus-1.5-8B SFT 256k-4200 | local full 5/5 | 33.70 | 46.35 | 31.31 | 39.93 | 40.04 | 38.27 | **39.71** | -1.45 vs 3600 SFT | Stronger than 3600 SFT on MMSI, but lower overall due to OmniSpatial and ViewSpatial. |
| InternVL3_5-8B | full 5/5 | 29.00 | 40.19 | 47.36 | 38.19 | 39.99 | 38.95 | **39.71** | **+0.77** | Apertus is stronger on MMSI, MindCube, SPAR, and ViewSpatial; weaker on OmniSpatial. |
| InternVL3-8B | full 5/5 | 28.00 | 41.54 | 45.34 | 35.86 | 38.66 | 37.88 | **39.71** | **+1.83** | Apertus wins 4/5 benchmarks; OmniSpatial is the main deficit. |
| Qwen3-VL-8B-Instruct | full 5/5 | 31.10 | 29.42 | 46.97 | 39.62 | 42.20 | 37.86 | **39.71** | **+1.85** | Apertus is much stronger on MindCube and modestly ahead on MMSI/SPAR. |
| BAGEL-7B-MoT | full 5/5 | 31.00 | 34.71 | 41.68 | 39.10 | 41.32 | 37.56 | **39.71** | **+2.15** | Apertus wins MMSI, MindCube, SPAR, and ViewSpatial. |
| Qwen2.5-VL-7B-Instruct | full 5/5 | 26.80 | 36.05 | 37.38 | 33.83 | 36.85 | 34.18 | **39.71** | **+5.53** | Apertus is ahead across all five. |
| Qwen3-VL-2B-Instruct | full 5/5 | 28.90 | 34.52 | 34.64 | 33.92 | 36.97 | 33.79 | **39.71** | **+5.92** | Apertus is ahead across all five. |
| gemma-4-26B-A4B | partial 3/5 | 29.20 | 48.85 | -- | -- | 41.68 | 39.91 | **39.92** | **+0.01** | Partial comparison only; Gemma is missing OmniSpatial and SPAR. |
| Qwen3-VL-8B-Thinking | partial 3/5 | 28.60 | 43.17 | -- | -- | 47.25 | 39.67 | **39.92** | **+0.25** | Partial comparison only; Qwen Thinking is stronger on ViewSpatial. |
| Qwen3-VL-30B-A3B-Thinking | partial 3/5 | 29.40 | 40.87 | -- | -- | 47.37 | 39.21 | **39.92** | **+0.71** | Partial comparison only; Qwen Thinking is stronger on ViewSpatial. |

## Behavior Summary

| behavior | evidence |
|---|---|
| Strong spatial transformation and 3D reasoning | MindCube is high relative to Qwen3-VL-8B-Instruct: 46.06 vs 29.42. |
| Solid embodied-view and spatial task performance | ViewSpatial is competitive with Qwen3-VL-8B-Instruct and ahead of InternVL3-8B: 41.89 vs 42.20 and 38.66. |
| Competitive multi-image spatial reasoning | MMSI is above Qwen3-VL-8B-Instruct and InternVL3-8B: 31.80 vs 31.10 and 28.00. |
| Main weakness is OmniSpatial | Apertus trails Qwen3-VL-8B-Instruct and InternVL3-8B on OmniSpatial: 37.70 vs 46.97 and 45.34. |

## Context: Stronger Rows Still Ahead

| public model | avg5 |
|---|---:|
| Gemini 3 Pro | 56.86 |
| GPT-5 | 50.53 |
| Gemini 2.5 Pro | 49.80 |
| Grok 4 | 47.41 |
| Seed 1.6 | 46.32 |
| SenseNova-SI-1.1-BAGEL-7B-MoT | 43.37 |
| VST-7B-SFT | 41.78 |
| Apertus-1.5-8B 256k-3600 online-DPO | **40.80** |
| Apertus-1.5-8B SFT 256k-3600 | 39.71 |

Source:
[EASI leaderboard](https://easi.lmms-lab.com/leaderboard/) and
[EASI public leaderboard API](https://easi.lmms-lab.com/api/leaderboard).
