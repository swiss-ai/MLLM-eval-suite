# Audio evaluation of the Apertus 1.5 release checkpoints

This branch runs the audio table of the Apertus 1.5 report: the released checkpoints `swiss-ai/Apertus-v1.5-8B` and `swiss-ai/Apertus-v1.5-70B`, and the open audio models Qwen2-Audio, Qwen2.5-Omni and Kimi-Audio, on 16 tasks with one fixed generation setting. Every input is pinned, every change against `main` is listed below, and the scripts refuse to submit from a checkout with local changes.

## Pinned inputs

| Input | Pin |
|---|---|
| Suite | this branch, on top of `main` 101fba0 |
| lmms-eval | `swiss-ai/lmms-eval` branch `ahadinia/audio-eval-final` @ 255cda3a, on top of a0650005 (the commit `main` pins) |
| Apertus image | `apertus-vllm-release-eval.sqsh`, sha256 `578ee90b642833c21509fa857e8581247fc89b6a218a26f82b142192478dcb4c`, built from `dockerfiles/Dockerfile.vllm-apertus-release-eval` on `ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64` (the vLLM image of the model card), adding the lmms-eval runtime; that base tag moves, so a rebuild can give a different hash |
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
| 747870c | MMAU runs as `mmau_test_mini` | The `mmau` group also runs `mmau_test`, which has no public answers; `main`'s run check rejects a group with an unscored member, so every MMAU job ended FAILED | this branch |

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
| 7d38e311 | Qwen2-Audio answers an undecodable clip empty | One CoVoST2 en-zh clip makes the audio decoder raise; Qwen2-Audio read audio outside its error handling, so the rank that drew it left the evaluation and the job hung until its time limit. Clips that decode are handled as before | this branch |
| 255cda3a | Qwen2.5-Omni's text output follows `max_new_tokens` | Omni's `generate()` in `transformers` caps the text model with `thinker_max_new_tokens` (default 1024) and passes a plain `max_new_tokens` only to its speech model, without a warning. Both Omni backends passed only `max_new_tokens`, so every Omni task ran with 1024 text tokens: TED-LIUM long form transcripts stopped at 853 to 967 words on talks of 1,108 to 3,219 words, and short-answer tasks could run past their caps. Qwen's cookbooks pass `thinker_max_new_tokens`; upstream lmms-eval main has the same bug | this branch |

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
| `mmau_test_mini` | test-mini | 128 (run as `mmau_test_mini`, not the `mmau` group: its test split has no public answers, and `main`'s run check rejects a group with an unscored member) |
| `clotho_aqa` | test | 8 |
| `muchomusic`, `vocalsound_test` | test | 4096 for every model (the tasks declare no cap; earlier runs used each backend's default, 256 for Qwen2-Audio and Kimi-Audio, but their answers there are at most 46 words, so the cap never applied) |

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
| System prompt | none | "You are a speech recognition model." (speech recognition), "You are a speech translation model." (CoVoST2), "You are an audio understanding model." (MMAU, MuChoMusic, Clotho-AQA), "You are a vocal sound classification model." (VocalSound) | none |
| Attention | default | eager; `sdpa` on TED-LIUM long form, where eager runs out of memory on the longest talks | default |
| Image | prod | prod with the qwen-omni-utils overlay | archived 2026-05 image with the Kimi overlay |

System prompts against each model's own material: Qwen2-Audio's template default ("You are a helpful assistant.") is what Qwen2-Audio's README and its official chat evaluation use (its published ASR, translation and VocalSound numbers come from the base model with no system prompt). Kimi-Audio takes no system prompt; its inference code rejects the role. For Qwen2.5-Omni, the speech-recognition, speech-translation and vocal-sound prompts are those of Qwen's `cookbooks/universal_audio_understanding.ipynb`. Qwen publishes no prompt for audio question answering; "You are an audio understanding model." follows the audio results page. Qwen's chat examples (for example `omni_chatting_for_music.ipynb`) use Omni's default prompt ("You are Qwen, a virtual human ..."); with it, Omni adds chat to short answers ("Yes. If you have any other questions, feel free to ask"), and its scores move both ways (two runs each, before the cap fix: MMAU 66.3 against 67.8, MuChoMusic 65.9 against 62.5, Clotho-AQA 78.1 against 87.0), so it is not used. The speech-translation prompt replaced the understanding prompt on CoVoST2 with the same score (43.9 against 43.7 BLEU, before the cap fix). The task prompts are the same for every model; Kimi-Audio's own evaluation toolkit lists the classes in its VocalSound prompt and scores classification with an LLM judge, so its published numbers are not directly comparable.

## Differences from the runs behind the current report

The final run on this branch replaces earlier numbers that came from older code. What differs:

- **lmms-eval base.** The earlier runs used lmms-eval 649a28e2 (Apertus) and 9aee58f2 (peers). This branch sits on a0650005, which adds an upstream merge: a rebuilt vLLM wrapper and changes to the evaluator. The task definitions, scorers and peer backends used here are identical to the tested ones apart from formatting; the Apertus vLLM path and the evaluator are not, so Apertus numbers can move.
- **70B memory.** `gpu_memory_utilization` 0.75 on every task; the earlier non-TED 70B runs used 0.85.
- **Qwen2.5-Omni.** One consistent setup, with task caps that take effect (255cda3a; every earlier Omni run, including the results page's, had 1024 text tokens on every task). The results page's Omni column mixed runs: FLEURS English and Ukrainian and MuChoMusic ran with Omni's default system prompt, and the other FLEURS languages through the earlier `google_fleurs` task.
- **Caps on MuChoMusic and VocalSound.** 4096 for every model; the earlier Qwen2-Audio and Kimi-Audio runs used their backend default of 256, which their answers (at most 46 words) never reached.
- **Tasks.** Only the table's 16 tasks; VoiceBench and MMSU are not run.

## Validation

Before the final run, every task ran for every model from a fresh clone of this branch with `LIMIT=32` (the first 32 samples of each task and split; TED-LIUM long form has 8 talks and ran in full), 85 jobs on the high-priority queue. Every job log shows lmms-eval 794ab50e, the task's cap and the batch size above, and no output is empty. Outputs were compared with the same samples of the runs behind the current report:

| Model | Outputs identical to the earlier run | Where they differ |
|---|---|---|
| Kimi-Audio | every task | none |
| Qwen2.5-Omni | every task except TED-LIUM long form (5/8) | `sdpa` is not deterministic on the long talks |
| Qwen2-Audio | most tasks; FLEURS Polish 20/32, Ukrainian 22/32 | batches of 8 are padded differently in a 32-sample run; its Polish and Ukrainian outputs are garbled anyway |
| Apertus 8B | most tasks; CoVoST2 102/128, FLEURS Polish 24/32, TED-LIUM 5/8 | late divergence on long outputs (TED-LIUM transcripts match for about 180 words), single characters in CoVoST2, and different paths on garbled Polish |
| Apertus 70B | most tasks; CoVoST2 92/128, VocalSound 15/32, TED-LIUM 0/8 | as for 8B, plus its free-text VocalSound answers; the earlier 70B runs (TP=4) were not deterministic either |

The Apertus differences come from small numeric changes (batch composition, the rebuilt vLLM wrapper) and do not shift scores: on the same 32 samples, FLEURS WER differs by -1.15 to +0.96 between old and new across the seven languages and both models, in both directions. MMAU runs as `mmau_test_mini` since this validation: the `mmau` group jobs scored test-mini but ended FAILED (see Changes), and the `mmau_test_mini` jobs completed with all outputs identical for every model. Qwen2-Audio's and Qwen2.5-Omni's CoVoST2 completed on these 32 samples; earlier full runs of Qwen2-Audio's CoVoST2 failed, so that job may still fail on the full set.

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

# 3. Weights at the pinned revisions, and the two peer overlays.
export CKPT_DIR=/path/for/apertus PEER_MODEL_DIR=/path/for/peers
hf download swiss-ai/Apertus-v1.5-8B  --revision a411d838600baf0e3635a3daf66fb7c55fc97bb6 --local-dir $CKPT_DIR/Apertus-v1.5-8B
hf download swiss-ai/Apertus-v1.5-70B --revision 59e744e313e967811aefabde3732609b64201acc --local-dir $CKPT_DIR/Apertus-v1.5-70B
hf download Qwen/Qwen2-Audio-7B-Instruct      --revision 0a095220c30b7b31434169c3086508ef3ea5bf0a --local-dir $PEER_MODEL_DIR/Qwen2-Audio-7B-Instruct
hf download Qwen/Qwen2.5-Omni-7B              --revision ae9e1690543ffd5c0221dc27f79834d0294cba00 --local-dir $PEER_MODEL_DIR/Qwen2.5-Omni-7B
hf download moonshotai/Kimi-Audio-7B-Instruct --revision 9a82a84c37ad9eb1307fb6ed8d7b397862ef9e6b --local-dir $PEER_MODEL_DIR/Kimi-Audio-7B-Instruct
uv pip install --target /path/to/qwen-omni-overlay --no-deps qwen-omni-utils==0.0.8 audioread==3.0.1
sbatch --account=infra01 --environment=$PWD/toml/shared/apertus-vllm-vision-eval-2026-05-torch210.toml \
  scripts/audio_repro/build_kimi_overlay.sbatch /path/to/kimi-overlay

# 4. Check the setup: print every job's arguments, then run the first 32 samples.
DRY_RUN=1 bash scripts/audio_repro/submit_apertus.sh 8b check
LIMIT=32 bash scripts/audio_repro/submit_apertus.sh 8b check librispeech

# 5. Two runs per model (one job per task).
for r in r1 r2; do
  bash scripts/audio_repro/submit_apertus.sh 8b  apertus_8b_$r
  bash scripts/audio_repro/submit_apertus.sh 70b apertus_70b_$r
  bash scripts/audio_repro/submit_peers.sh qwen2_audio peers_qwen2_audio_$r
  EXTRA_PYTHONPATH=/path/to/qwen-omni-overlay bash scripts/audio_repro/submit_peers.sh qwen2_5_omni peers_qwen2_5_omni_$r
  EXTRA_PYTHONPATH=/path/to/kimi-overlay bash scripts/audio_repro/submit_peers.sh kimi_audio peers_kimi_audio_$r
done

# 6. Compare the two runs of a model.
python scripts/audio_repro/compare_runs.py \
  results/lmms-eval/Apertus-v1.5-8B/apertus_8b_r1 results/lmms-eval/Apertus-v1.5-8B/apertus_8b_r2
```

To resubmit only some tasks (for example after a preemption), pass them as the third argument: `bash scripts/audio_repro/submit_apertus.sh 8b apertus_8b_r1 covost2,mmau`.

## Known limits

- Qwen2-Audio's CoVoST2 runs have failed every time (an audio decoding error, or one data-parallel rank stalling), and Qwen2.5-Omni's CoVoST2 has not completed in our runs, so neither has a CoVoST2 score yet.
