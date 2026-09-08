# Evaluation validity and harness sync follow-ups

> For agentic workers: use subagent-driven-development for independent harness and manifest tasks, with review before publication.

**Goal:** Close the concrete correctness gaps found in the September 8 PR review and update the existing PR stack with validated fixes.

**Architecture:** Validate results at finalization and again at collection. Select eligible artifacts deterministically across all roots and aliases, preserve explicit provenance or legacy status, and exercise model-specific formatting through the real harness methods. Keep the current harness boundaries and benchmark definitions.

**Tech stack:** Python, pytest/unittest, Bash, GitHub Actions; existing Slurm/Pyxis runtime for any subsequent model checks.

**Spec:** `docs/superpowers/specs/2026-09-08-eval-suite-hardening-design.md`, especially C2-C5 and C8-C9, plus the user-approved review findings in this conversation.

## Constraints

- Work in isolated clones under /tmp; preserve the canonical capstor checkout and damaged scratch checkout.
- Keep existing PR branches and update them by fast-forward commits; no main merge or production image promotion.
- Do not infer a throughput improvement or broad model compatibility from CPU regression checks.
- Preserve explicitly imported historical results as legacy; do not fabricate missing provenance.
- Record failing behavioral tests before fixing each defect. Run relevant existing tests and a final independent review.

## Task 1: Result collection and reporting

- [x] Add regression tests for invalid manifests, limited runs in every lane, older eligible fallback, symlinked result paths, cross-root/alias ordering and collisions, and percent units.
- [x] Index all manifest roots including lm-eval; resolve result paths consistently.
- [x] Filter candidates before newest selection. Reject failed/invalid/running or malformed manifests and explicit partial coverage; preserve manifestless historical results as legacy.
- [x] Merge rows by eligible artifact timestamp with deterministic ties across roots and aliases; make collision verification inspect the same candidate identities without basename loss.
- [x] Honor explicit percentage units; reject nonfinite/invalid scores.
- [x] Keep valid zero and RefSpatial scores visible; align judge failure thresholds and remove the retired scratch root from derivation.
- [x] Add a CPU-only suite CI gate and document evidence limitations. Keep dashboard artifacts separate from synthetic test results.

## Task 2: Manifest validity

- [x] Add tests for malformed, stale, wrong-task and incomplete result sets, unobserved thinking, and incomplete weight shards.
- [x] Validate parseability, task membership, available sample counts and freshness at finalization. Respect legitimate grouped tasks and resumed evaluations; snapshot preexisting artifacts at start where appropriate.
- [x] Require positive observed thinking evidence for requested thinking; missing evidence cannot silently become ok.
- [x] Validate all shard references in the weight index at preflight.
- [x] Run manifest/preflight tests and report supported formats and remaining runtime limits.

## Task 3: lmms-eval cross-model regression

- [x] Reproduce lost Molmo task prefixes through actual generate_until formatting with external backend mocked.
- [x] Preserve benchmark task identity through image encoding; test MCQ and task-specific prefixes plus generic wrapper behavior.
- [x] Add an automated lightweight regression gate suitable for CI.
- [x] Run focused harness tests, inspect fork delta, and commit on the existing sync branch.

## Task 4: VLMEvalKit sync CI

- [x] Reproduce current Python 3.10 syntax/style failures with the declared hooks.
- [x] Fix supported-Python syntax and scoped lint issues without altering evaluation semantics.
- [x] Run declared checks and focused existing tests; commit on the sync branch.

## Task 5: Integration and publication

- [x] Independently review each task diff and fix material findings.
- [x] Run all first-party tests, relevant harness regressions and shell syntax checks.
- [ ] Fast-forward push validated harness PR branches; update suite sync pointers and merge the new hardening commits into its stacked branch.
- [ ] Push suite PR branches, inspect fresh CI, and update concrete PR descriptions with validation and limitations.
- [ ] Report exact commits, tests and remaining GPU validation work. Do not merge PRs or promote images.
