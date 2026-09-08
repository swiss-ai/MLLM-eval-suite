# September 8 correctness review

The PR stack improves the suite's operational controls: a shared task registry,
submission preflight, run manifests, thinking canaries, cache preparation, and
audited harness synchronization. This follow-up closes defects that could still
produce misleading scores after those controls were added.

## Result integrity

Dashboard collection and collision verification use the same eligible artifacts
across lmms-eval, VLMEvalKit, and lm-eval. Failed, invalid, running, malformed,
limited, known-partial, and unusable-score candidates are excluded before choosing
the newest result. Selection is deterministic across roots and model aliases.
Known rejected runs cannot be restored by importing an earlier dashboard snapshot.

Explicit percent units are preserved, valid zero scores remain visible, and judge
derivation shares the 5% failure threshold. Historical results without manifests
are labeled legacy rather than assigned inferred provenance.

Finalization checks fresh, parseable output for the requested task, including
nested task groups and judged CSVs; it compares available sample counts and
deduplicates resumed sample logs. Requested thinking requires an observed effective
flag and a passed canary, while missing token statistics remain unknown. Preflight
also checks every weight shard named by a safetensors index.

## Harness evidence

- lmms-eval `63dafbbe`: the visual-encoding loop no longer overwrites the task
  string with a Future. Regression tests exercise real wrapper methods and verify
  Molmo MMVP/ChartQA prefixes, generic prompts, and image order. Five focused
  prompt/sampling tests and two watchdog tests passed locally.
- VLMEvalKit `f6a2ac9`: Python 3.10 syntax and pinned style fixes preserve logging
  redaction and evaluation behavior. All pinned pre-commit hooks and eight focused
  tests passed; both GitHub lint runs passed at this commit.
- The synchronization audits pass with explicit reviewed reasons for equivalent
  formatting changes and the obsolete logger module. The audit's deletion path
  still prints removals without failing the gate; it is not a complete semantic
  equivalence check.

The earlier Apertus comparison in the design spec applies to its recorded commits.
It is not a rerun of every benchmark at the final PR heads. Historical manifests
occasionally name a stale checkout, so the saved result's embedded commit is
needed when reconstructing those comparisons.

## Stored-data audit and publication gate

The combined local gate passes 163 suite/image-contract tests, plus registry and
shell checks. A preview rebuild has 1,764 cells: all 1,761 previous scores are
unchanged, with RefSpatial's valid zero and two older usable MathVision scores
restored. All 1,764 are marked legacy because these historical artifacts lack run
manifests. Card/report coverage remains 784/1,102, with 318 missing cells.

The expanded [collision audit](collisions.txt) found seven conflicting text scores
hidden by two aliases. The SDPO alias is now split: the Mix and Less-Refuse
checkpoints have different `model.norm.weight` tensors (BF16, shape `[4096]`,
8,192 bytes each). SHA256 of those bytes is
`a58d715790893a135bb50880603e8fbdb8e25256a7e059bb3ef30c61a9015ad7`
for Mix and
`e7ddd4bedd23e64de02315212ad2d45ecbffb00b2be59fe3d4cde37809b7eeed`
for Less-Refuse. The short `SDPO-Mix-Less-Refuse-Feedback` path resolves to Mix.

The five-score 8B conflict remains unresolved: old `Apertus-v1.5-8B` text results
record no chat template, while `8B-Final-correct-rope` results record one; the old
textview directory is empty, preventing weight-identity verification. The retained
final textview also has RoPE overrides (factor 32, theta 4,000,000). A preference
for separate historical columns versus replacement full runs is pending.

`refresh_dashboard.sh` now runs the collision audit with the same selected models,
aliases, and all three result lanes before replacing dashboard artifacts. It also
stops on derivation errors. The preview above precedes the proven SDPO alias split;
published dashboard files are deliberately not regenerated while the 8B conflict
remains. Existing legacy snapshots can contain earlier alias conflation and are
not certified by the new run validation.

## GPU image probes

Two synthetic 224-by-224 red/blue images were sent through the actual simple VLLM
wrapper before and after the routing fix, sharing one loaded engine per model.
Generation used temperature 0, a 16-token cap, TP1, eager execution, and disabled
prefix caching. Both models ran on idle GH200 GPUs in coding allocation 3328238,
using production EDF `apertus-vllm-vision-eval-prod.toml` and vLLM
`0.26.1rc1.dev687+g324f452f6`.

| Model | Exact before/after matches | Nonempty outputs |
|---|---:|---:|
| Qwen3-VL-8B-Instruct | 2/2 | 2/2 |
| MedGemma 1.5 4B IT (Gemma 3 architecture) | 2/2 | 2/2 |

The probe loads the baseline method from `e8e1b2ac` and the candidate module from
`63dafbbe`. [Qwen output](qwen3.json), [MedGemma output](medgemma.json), and the
[probe source](probe_generic.py) record exact commits, image hashes, and responses.
Place a clone of the candidate lmms-eval next to the probe as `lmms/` to rerun it;
pass `--model`, `--gpu`, and `--output` for the local environment.

These checks validate a small image-generation path. They do not measure benchmark
accuracy, throughput, multi-GPU behavior, video/audio support, or the trial image.
Molmo inference was not run because its cached weights were incomplete.

## Remaining priorities

1. Re-run missing or semantically changed benchmarks, especially the corrected
   checkpoints' `slake_medevalkit` task, with fresh manifests and full sample counts.
2. Retain raw outputs and immutable model/data/image identities outside scratch,
   so historical evidence can be reconstructed without relying on legacy cells.
3. Add representative model-family and image/video/audio runtime gates, followed
   by controlled repeated performance comparisons before promoting the trial image.
4. Add paired uncertainty estimates and fixed dataset/prompt revisions to important
   model comparisons. More harness coverage alone does not establish scientific
   comparability or eliminate benchmark contamination.
