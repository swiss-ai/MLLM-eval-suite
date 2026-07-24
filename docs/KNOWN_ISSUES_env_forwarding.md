# TEMPORARY WORKAROUND — EXTRA_PYTHONPATH not forwarded to VLMEvalKit jobs

## Symptom
Gemma-4 (model_type `gemma4_unified`, needs transformers 5) fails on VLMEvalKit with
"Transformers does not recognize this architecture", even when the caller exports
`EXTRA_PYTHONPATH=.../cache/pylibs/transformers5`.

## Root cause (verified 2026-07-24, job 2886939)
The staged tf5 overlay works fine — with it on PYTHONPATH the container loads
transformers 5.13.1 and reads the gemma4_unified config cleanly. The problem is
delivery: `launchers/VLMEvalKit/eval.sh` submits via `sbatch` with no `--export`,
and under the Pyxis `--environment` container custom env vars from the submitting
shell do not propagate into the job. The slurm template reads `${EXTRA_PYTHONPATH}`
(eval_job.slurm:142) but it arrives empty. (lmms-eval is unaffected only by luck of
its own submission wrapper.)

Earlier hypotheses that were WRONG and should not be repeated:
 - "the eval image was rebuilt" — image mtime is unchanged since Jun 11.
 - "import ordering binds the container transformers first" — refuted; the overlay
   wins whenever it is actually on the path.

## Temporary workaround (in use now)
Prepend `SBATCH_EXPORT` when invoking the VK launcher:

    export SBATCH_EXPORT="ALL,EXTRA_PYTHONPATH=$PWD/cache/pylibs/transformers5"
    EXTRA_PYTHONPATH="$PWD/cache/pylibs/transformers5" \
      bash launchers/VLMEvalKit/eval.sh --model Gemma4-12B-it --data SparBench ...

This forces sbatch to carry the variable into the job.

## Optimal fix (TODO — not yet applied)
Decide the clean version, e.g. one of:
 - add `--export=ALL,EXTRA_PYTHONPATH` to the sbatch line in launchers/VLMEvalKit/eval.sh
   (and mirror in lmms-eval for parity), OR
 - have the launcher write EXTRA_PYTHONPATH into the job via the SBATCH_OVERRIDES array, OR
 - bake transformers 5 into the eval container image so no overlay is needed at all
   (removes this whole class of env-forwarding fragility).
