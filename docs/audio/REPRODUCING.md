# Audio evaluation of the Apertus 1.5 release checkpoints

This branch runs the audio table of the Apertus 1.5 report: the released checkpoints `swiss-ai/Apertus-v1.5-8B` and `swiss-ai/Apertus-v1.5-70B`, and the open audio models Qwen2-Audio, Qwen2.5-Omni and Kimi-Audio, on 16 tasks with one fixed generation setting. Every input is pinned, every change against `main` is listed below, and the scripts refuse to submit from a checkout with local changes.

## Pinned inputs

| Input | Pin |
|---|---|
| Suite | this branch, on top of `main` 101fba0 |
| lmms-eval | `swiss-ai/lmms-eval` branch `ahadinia/audio-eval-final` @ 794ab50e, on top of a0650005 (the commit `main` pins) |
| Apertus image | `apertus-vllm-release-eval.sqsh`, sha256 `578ee90b642833c21509fa857e8581247fc89b6a218a26f82b142192478dcb4c`, built from `dockerfiles/Dockerfile.vllm-apertus-release-eval` on `ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64`; that base tag moves, so a rebuild can give a different hash |
| Peer images | `toml/shared/apertus-vllm-vision-eval-prod.toml` (Qwen2-Audio, Qwen2.5-Omni), `toml/shared/apertus-vllm-vision-eval-2026-05-torch210.toml` (Kimi-Audio) |
| Apertus weights | `swiss-ai/Apertus-v1.5-8B` @ a411d838, `swiss-ai/Apertus-v1.5-70B` @ 59e744e3 |
| Peer weights | `Qwen/Qwen2-Audio-7B-Instruct` @ 0a095220, `Qwen/Qwen2.5-Omni-7B` @ ae9e1690, `moonshotai/Kimi-Audio-7B-Instruct` @ 9a82a84c |
| Peer overlays | Qwen2.5-Omni: `qwen-omni-utils==0.0.8`, `audioread==3.0.1`; Kimi-Audio: `scripts/audio_repro/build_kimi_overlay.sbatch` (Kimi-Audio 349251e1, flash-attn 2.7.4.post1) |

## Changes against `main`

### Suite

| Commit | Change | Why | Origin |
|---|---|---|---|
| 7a5ffd1 | Release-vLLM eval image recipe and its toml (prod with only `image =` changed) | The released checkpoints use the Transformers 5.14 layout with `lm_head` pruned to the 131,072 text ids; the prod image's vLLM cannot load them | swiss-ai/MLLM-eval-suite#14 |
| 9c7008f | No reservation by default | `SD-69241-apertus-1-5-0` has expired and Slurm rejects submissions that name it | #14 |
| b19bf4d | `scripts/audio_repro/`: `submit_apertus.sh`, `submit_peers.sh`, `compare_runs.py`, `build_kimi_overlay.sbatch` | One command per model; settings below | #14 and #15, rewritten for `main`'s launcher |
| 489b138 | lmms-eval pinned to `ahadinia/audio-eval-final` | The lmms-eval changes below | this branch |
| 858120d | `LIMIT=N` in the submit scripts | Short runs to check the setup before a full run | this branch |

### lmms-eval (on top of a0650005)

| Commit | Change | Why | Origin (author) |
|---|---|---|---|
| db424cde | The Apertus wrapper accepts audio, through the inherited chat renderer | At a0650005 the wrapper rejects audio content | 00a3ae16 on `codex/pr2-preserve-audio` (Alvorecer721) |
| 07239e89 | Per-language FLEURS tasks (`fleurs_en_us`, `de_de`, `fr_fr`, `it_it`, `es_419`, `pl_pl`, `uk_ua`, …) | Not on a0650005 | swiss-ai/lmms-eval#12 (Anunay Yadav) |
| c99a2252 | CoVoST2 en-zh loads the dataset's `default` config | `lmms-lab-audio/covost2_en-zh` has only `default`; `en_zh` fails to load | swiss-ai/lmms-eval#27 |
| 1f0957e7 | A task's `max_new_tokens` is used as given | The vLLM backend took `max(task cap, 4096)`, so one looping sample could add thousands of insertions to a WER (70B FLEURS Italian 16.2 against 7.2) | #27 |
| 01a4ac61 | Qwen2.5-Omni imports moviepy only for video; Qwen2-Audio downmixes multi-channel audio | Audio-only tasks failed in these backends | #26 / #28 |
| c3542980 | Qwen2-Audio and Kimi-Audio look up each request's own task in a batch | A batch can span the subtasks of a group | #26 / #28 |
| b8fe150d | An out-of-range MMAU answer letter scores as wrong | It raised and stopped the run | #26 / #28 |
| dd7be24e | Chat backends answer every request; Qwen2.5-Omni decodes dataset audio to mono 16 kHz | Batch size above 1 failed; dataset audio objects were not accepted | #26 / #28 |
| 1807d8b1 | Qwen2-Audio and Kimi-Audio warn when a generation error leaves answers empty | At batch size 512 Qwen2-Audio ran out of memory and returned empty answers without a visible error | #26 / #28 |
| 794ab50e | TED-LIUM loads without a Hugging Face login | The dataset is public; the other audio tasks still need a login | #28 |

Not ported: de31accd (keep `enable_thinking` across vLLM initialization), because a0650005 already carries 8fd62f0f, which does the same.

## Configuration

### Generation

Greedy decoding for every model, and each task's output cap passed explicitly to each job, because `main`'s launcher otherwise sets `max_new_tokens=16384` on every job. The scripts submit one job per task.

| Task | Splits scored | max_new_tokens |
|---|---|---|
| `librispeech` | test-clean, test-other (dev splits also run) | 256 |
| `open_asr_voxpopuli`, `open_asr_spgispeech` | test | 4096 |
| `tedlium_long_form` | val (8 talks, 3 to 22 minutes) | 4096 (the task declares 256, which truncates the transcripts) |
| `fleurs_en_us`, `de_de`, `fr_fr`, `it_it`, `es_419`, `pl_pl`, `uk_ua` | test | 256 |
| `covost2` | en-zh test | 256 |
| `mmau` | test-mini | 128 |
| `clotho_aqa` | test | 8 |
| `muchomusic`, `vocalsound_test` | test | no task cap: 4096 for Apertus and Qwen2.5-Omni, 256 for Qwen2-Audio and Kimi-Audio (each backend's default) |

### Apertus

| Setting | 8B | 70B |
|---|---|---|
| Backend | `apertus_1p5_vllm` | same |
| Tokenizer and chat template | the checkpoint's own (`TOKENIZER_PATH`, `CHAT_TEMPLATE`); the suite's default tokenizer renders audio requests with an extra newline after the audio | same |
| Workers | 4 data-parallel, one GPU each | 1, `tensor_parallel_size=4`, CUDA graphs with `fuse_allreduce_rms` disabled (it fails graph capture at ≤128 tokens on this vLLM build) |
| `gpu_memory_utilization` | 0.6 (job default) | 0.75 (`main`'s 70b profile) |
| `max_num_batched_tokens` | 49152; 65536 for TED-LIUM long form | same |
| Other | batch size 512, `max_model_len` 131072, prefix caching on, model-level `max_new_tokens=4096`, `VLLM_MAX_AUDIO_DECODE_DURATION_S=3600` | same |

### Peers

| Setting | Qwen2-Audio | Qwen2.5-Omni | Kimi-Audio |
|---|---|---|---|
| Backend | `qwen2_audio` | `qwen2_5_omni` (chat) | `kimi_audio` |
| Batch size | 8 (1 on TED-LIUM long form) | 1 (the backend joins a batch into one conversation) | 1 |
| System prompt | none | "You are a speech recognition model." (speech recognition), "You are an audio understanding model." (MMAU, MuChoMusic, Clotho-AQA, CoVoST2), "You are a vocal sound classification model." (VocalSound) | none |
| Attention | default | eager; `sdpa` on TED-LIUM long form, where eager runs out of memory on the longest talks | default |
| Image | prod | prod with the qwen-omni-utils overlay | archived 2026-05 image with the Kimi overlay |

## Differences from the runs behind the current report

The final run on this branch replaces earlier numbers that came from older code. What differs:

- **lmms-eval base.** The earlier runs used lmms-eval 649a28e2 (Apertus) and 9aee58f2 (peers). This branch sits on a0650005, which adds an upstream merge: a rebuilt vLLM wrapper and changes to the evaluator. The task definitions, scorers and peer backends used here are identical to the tested ones apart from formatting; the Apertus vLLM path and the evaluator are not, so Apertus numbers can move.
- **70B memory.** `gpu_memory_utilization` 0.75 on every task; the earlier non-TED 70B runs used 0.85.
- **Qwen2.5-Omni.** One consistent setup. The results page's Omni column mixed runs: FLEURS English and Ukrainian and MuChoMusic ran with Omni's default system prompt, and the other FLEURS languages through the earlier `google_fleurs` task.
- **Tasks.** Only the table's 16 tasks; VoiceBench and MMSU are not run.

## Steps

```bash
git clone --recurse-submodules -b ahadinia/audio-eval-final https://github.com/swiss-ai/MLLM-eval-suite
cd MLLM-eval-suite

# 1. Image (once): build it and check the sha256 above. Point a copy of
#    toml/shared/apertus-vllm-release-eval.toml outside the checkout at it
#    (editing it in place makes the checkout dirty and the scripts refuse to run).
sbatch --account=infra01 --nodes=1 --exclusive --time=04:00:00 \
  dockerfiles/build_release_eval_image.sh "$PWD" "$PWD/cache/image-builds/release-eval"
export EVAL_ENVIRONMENT=/path/to/your/apertus-vllm-release-eval.toml

# 2. Host Python >= 3.8 for the launcher; a Hugging Face login for the datasets.
uv venv --python 3.12 ~/venvs/mllm-eval && source ~/venvs/mllm-eval/bin/activate
export HF_TOKEN=<token>

# 3. Check the setup: print every job's arguments, then run the first 32 samples.
export CKPT_DIR=/path/with/Apertus-v1.5-8B-and-70B PEER_MODEL_DIR=/path/with/peer/snapshots
DRY_RUN=1 bash scripts/audio_repro/submit_apertus.sh 8b check
LIMIT=32 bash scripts/audio_repro/submit_apertus.sh 8b check librispeech

# 4. Two runs per model (one job per task).
for r in r1 r2; do
  bash scripts/audio_repro/submit_apertus.sh 8b  apertus_8b_$r
  bash scripts/audio_repro/submit_apertus.sh 70b apertus_70b_$r
  bash scripts/audio_repro/submit_peers.sh qwen2_audio peers_qwen2_audio_$r
  EXTRA_PYTHONPATH=/path/to/qwen-omni-overlay bash scripts/audio_repro/submit_peers.sh qwen2_5_omni peers_qwen2_5_omni_$r
  EXTRA_PYTHONPATH=/path/to/kimi-overlay bash scripts/audio_repro/submit_peers.sh kimi_audio peers_kimi_audio_$r
done

# 5. Compare the two runs of a model.
python scripts/audio_repro/compare_runs.py \
  results/lmms-eval/Apertus-v1.5-8B/apertus_8b_r1 results/lmms-eval/Apertus-v1.5-8B/apertus_8b_r2
```

To resubmit only some tasks (for example after a preemption), pass them as the third argument: `bash scripts/audio_repro/submit_apertus.sh 8b apertus_8b_r1 covost2,mmau`.

## Known limits

- Qwen2-Audio's CoVoST2 runs have failed every time (an audio decoding error, or one data-parallel rank stalling), and Qwen2.5-Omni's CoVoST2 has not completed in our runs, so neither has a CoVoST2 score yet.
- `toml/shared/apertus-vllm-release-eval.toml` points at a copy of the image in personal scratch; it needs a shared location before this merges.
