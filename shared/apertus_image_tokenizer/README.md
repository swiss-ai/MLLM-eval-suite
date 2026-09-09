# apertus_image_tokenizer

Harness-side image tokenization for Apertus: images become framed
`<|visual token N|>` text (Emu3.5 VQ) spliced into the prompt before the
engine, which only ever sees token ids. Both harness wrappers call the single
entry point `splice_frames()`.

## Image-token cache: scope and safety contract

**Apertus-only.** The cache memoizes the deterministic image → VQ-token
conversion, which is identical for every Apertus checkpoint (same
`APERTUS_VQ_HUB` → same tokens, so 8B and 70B share entries). Foreign
continuous-encoder models never touch it — they encode images inside the model
and have nothing to amortize. Both launchers gate it accordingly
(`launchers/lmms-eval/eval.sh` on `MODEL_BACKEND == apertus*`,
`launchers/VLMEvalKit/eval.sh` on the foreign-model classification).

**Fill is effectively one-time.** The first Apertus run over a task writes
that task's entries; every later run — any checkpoint, either harness — only
reads. Steady state is a read-only cache; concurrent cross-node writes are a
transient of the initial fill, not an ongoing workload.

**Why concurrent fill is safe enough.** SQLite WAL locking is not trustworthy
across nodes on Lustre, so the design does not rely on it for correctness:

- Writes are `INSERT OR IGNORE` keyed by image content hash — first writer
  wins, and every racing writer would insert identical bytes (the conversion
  is deterministic). The worst race outcome is wasted encode work.
- `PRAGMA busy_timeout` retries are counted and logged; the optional
  collision guard stores enough source data to detect a hash collision.
- Per-task cache files bound the blast radius: one corrupted file costs one
  task's fill, never the campaign.

Observed across the full 9-model campaign, including real cross-node fills:
zero busy-timeout warnings, zero guard trips.

**Discipline for large fan-outs:** let one run fill first, then launch the
fleet with `IMAGE_TOKEN_CACHE_MODE=readonly` (optionally with preload). This
avoids even the benign race and is the cheapest possible schedule anyway.
