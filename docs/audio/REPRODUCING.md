# Reproducing the audio evaluation of the Apertus 1.5 release checkpoints

This branch evaluates the public release checkpoints `swiss-ai/Apertus-v1.5-8B` and `swiss-ai/Apertus-v1.5-70B` on 16 audio benchmarks (speech recognition, speech translation, audio understanding, audio question answering and sound classification), with every input pinned and one fixed generation setting. It is based on `audio-results` (8ac19a5).

## Pinned inputs

| Input | Pin |
|---|---|
| Suite | this branch |
| lmms-eval | `swiss-ai/lmms-eval` branch `ahadinia/audio-release-ckpt-peers` @ 585073ca (swiss-ai/lmms-eval#28, on top of swiss-ai/lmms-eval#27 at 649a28e2, which produced the Apertus results; #28 changes only the peer backends, the chat-backend mixin they share with other chat models, MMAU's handling of out-of-range answer letters, and the login flag of the two TED-LIUM tasks) |
| Image | `apertus-vllm-release-eval.sqsh`, sha256 `578ee90b642833c21509fa857e8581247fc89b6a218a26f82b142192478dcb4c`, built from `dockerfiles/Dockerfile.vllm-apertus-release-eval` on `ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64`; that base tag moves, so a rebuild can give a different hash |
| Weights | `swiss-ai/Apertus-v1.5-8B` @ a411d838, `swiss-ai/Apertus-v1.5-70B` @ 59e744e3; every file's sha256 matches the Hub |
| Backend | `apertus_1p5_vllm` (suite default) with the tokenizer and chat template shipped with each checkpoint |

lmms-eval commits on top of `main`:

1. de31accd, 00a3ae16: Apertus thinking mode and audio chat in the `apertus_1p5_vllm` wrapper (from `codex/pr2-preserve-audio`).
2. 255f9d18: the `google_fluers` per-language FLEURS tasks (from swiss-ai/lmms-eval#12); this evaluation uses `fleurs_en_us`, `de_de`, `fr_fr`, `it_it`, `es_419`, `pl_pl` and `uk_ua`.
3. 37e5ae90: CoVoST2 en-zh loads the dataset's `default` config (`lmms-lab-audio/covost2_en-zh` no longer has `en_zh`).
4. 6bd2671b: a task's `max_new_tokens` is used as given. The vLLM backend took `max(task cap, 4096)`, so every cap of 256 or less silently became 4096.

The results below come from two runs per model (2026-10-06) from one fresh clone of this branch at f06baa9 (lmms-eval 649a28e2). Its caches (datasets, vLLM compilation and image tokens) were empty for the first runs; the second runs reused what the first ones cached.

## Configuration

These are the settings of the reported runs, as recorded in their result files and logs; the environment variable is set by `submit_apertus.sh`.

### Tasks and generation

All tasks decode greedily (temperature 0, no sampling, one beam). Tasks that declare no temperature get 0 from the `apertus_1p5_vllm` wrapper.

| Task (lmms-eval) | Dataset | Split(s) | max_new_tokens |
|---|---|---|---|
| `librispeech` | `lmms-lab-audio/librispeech` | dev-clean, dev-other, test-clean, test-other | 256 |
| `open_asr_voxpopuli` | `hf-audio/esb-datasets-test-only-sorted` | test | 4096 |
| `open_asr_spgispeech` | `hf-audio/esb-datasets-test-only-sorted` | test | 4096 |
| `tedlium_long_form` | `lmms-lab-audio/tedlium` | val | 4096 (passed with `--gen-kwargs`; the task declares 256) |
| `fleurs_en_us`, `fleurs_de_de`, `fleurs_fr_fr`, `fleurs_it_it`, `fleurs_es_419`, `fleurs_pl_pl`, `fleurs_uk_ua` | `google/fleurs` | test | 256 |
| `covost2` | `lmms-lab-audio/covost2_en-zh` (en-zh), `lmms-lab-audio/covost2` (zh-en) | dev, test | 256 |
| `mmau` | `lmms-lab-audio/mmau` | test, test_mini (only test_mini is scored) | 128 |
| `muchomusic` | `lmms-lab-audio/muchomusic` | test | 4096 (no task cap; model-level cap, see Engine) |
| `clotho_aqa` | `lmms-lab-audio/ClothoAQA` | val, test (filtered) | 8 |
| `vocalsound_test` | `lmms-lab-audio/vocalsound` | test | 4096 (no task cap; model-level cap, see Engine) |

### Engine

| Setting | 8B | 70B |
|---|---|---|
| Backend | `apertus_1p5_vllm` | `apertus_1p5_vllm` |
| Tokenizer and chat template | The checkpoint's own (`Apertus-v1.5-8B/` and its `chat_template.jinja`), set in `submit_apertus.sh` through `TOKENIZER_PATH` and `CHAT_TEMPLATE`. The launcher's default, the suite's internal Apertus tokenizer, renders audio requests with an extra newline after the audio. | The checkpoint's own (`Apertus-v1.5-70B/`) |
| Workers | 4 data-parallel processes, one GPU each (`tensor_parallel_size=1`) | 1 process, `tensor_parallel_size=4` |
| CUDA graphs | on (`enforce_eager=false`) | on, with `compilation_config={"pass_config":{"fuse_allreduce_rms":false}}` |
| `gpu_memory_utilization` | 0.6 | 0.85 (TED-LIUM long-form: 0.75) |
| `max_model_len` | 131072, with `hf_overrides={"max_position_embeddings":131072}` | same |
| `max_num_batched_tokens` | 49152 (TED-LIUM long-form: 65536) | 49152 (TED-LIUM long-form: 65536) |
| `enable_prefix_caching` | true | true |
| Model-level `max_new_tokens` | 4096, set in `submit_apertus.sh` (`--extra-model-args max_new_tokens=4096`). It applies only to tasks that declare no cap (MuChoMusic, VocalSound); a task's own cap takes precedence. | same |
| lmms-eval `--batch_size` | 512 | 512 |
| Seed | vLLM engine 1; lmms-eval (Python, NumPy, PyTorch) 1234 | same |
| Environment | `VLLM_MAX_AUDIO_DECODE_DURATION_S=3600` | same |

`scripts/audio_repro/submit_apertus.sh` sets the model-specific values and the TED-LIUM overrides; the launcher defaults supply the rest (backend, `max_model_len`, batch size, seed).

## Why these settings

- **Image.** The release checkpoints use the Transformers 5.14 layout (`Apertus1p5ForConditionalGeneration`) with `lm_head` pruned to the 131,072 text ids. The prod image's vLLM cannot load them (vocab-size assertion in `vocab_parallel_embedding`); the release vLLM can.
- **Generation.** Each task's declared cap, temperature 0. TED-LIUM long-form gets 4096, because its task cap of 256 truncates the transcripts of its talks (3 to 22 minutes, up to 4,164 words); tasks that declare no cap (MuChoMusic, VocalSound) use the model-level cap of 4096 set in `submit_apertus.sh` (see Engine). With the old `max()` rule, one looping sample added about 10 WER to 70B FLEURS Italian (16.2 against 7.2).
- **70B.** TP=4 with CUDA graphs. On this vLLM build the fused all-reduce + RMSNorm pass (`fuse_allreduce_rms`) fails graph capture at batch sizes up to 128 tokens with an illegal memory access, so only that pass is disabled. Running eager instead gives the same scores within about 1 point.
- **TED-LIUM long-form.** The eight talks run 3 to 22 minutes, and the longest exceed the encoder cache, which follows `max_num_batched_tokens` (launcher default 49152): `VLLM_MAX_AUDIO_DECODE_DURATION_S=3600` (vLLM's default limit is 600 s), `max_num_batched_tokens=65536`, and on 70B `gpu_memory_utilization=0.75` (at 0.85 the longest talk runs out of memory).

## Steps

```bash
git clone --recurse-submodules -b ahadinia/audio-release-ckpt https://github.com/swiss-ai/MLLM-eval-suite
cd MLLM-eval-suite

# 1. Image (once): build it and check the sha256 above. Point a copy of
#    toml/shared/apertus-vllm-release-eval.toml outside the checkout at it (editing the
#    file in place makes the checkout dirty, and the submit script refuses to run).
#    export EVAL_ENVIRONMENT=/path/to/your/apertus-vllm-release-eval.toml
sbatch --account=infra01 --nodes=1 --exclusive --time=04:00:00 \
  dockerfiles/build_release_eval_image.sh "$PWD" "$PWD/cache/image-builds/release-eval"

# 2. Host Python: the launcher needs python3 >= 3.8 (login nodes may default to 3.6).
uv venv --python 3.12 ~/venvs/mllm-eval && source ~/venvs/mllm-eval/bin/activate

# 3. Submit twice (the script refuses to run from a checkout with local changes).
export CKPT_DIR=/path/with/Apertus-v1.5-8B-and-70B
for size in 8b 70b; do
  bash scripts/audio_repro/submit_apertus.sh $size apertus_${size}_r1
  bash scripts/audio_repro/submit_apertus.sh $size apertus_${size}_r2
done

# 4. Compare the two runs.
python scripts/audio_repro/compare_runs.py \
  results/lmms-eval/Apertus-v1.5-8B/apertus_8b_r1 results/lmms-eval/Apertus-v1.5-8B/apertus_8b_r2
```

To compare against the old cap rule, submit from a second clone whose lmms-eval submodule is checked out at 37e5ae90 (the commit before the cap fix) and committed, since the submit script refuses a checkout with local changes.

## Reproducibility check

Each model ran twice from empty caches; `compare_runs.py` compares the generated text sample by sample.

| Model | Identical outputs, run 1 vs run 2 | Largest score difference |
|---|---|---|
| 8B | 100% on every task | none; all scores identical |
| 70B | 57-99% per task (TED-LIUM long form 2 of 8 talks) | TED-LIUM long form 5.0; all other tasks within 0.4 |

The 8B runs are byte-identical. With TP=4 the 70B outputs also depend on batch timing and kernel autotuning, so they differ on borderline samples, which moves scores by a few tenths. TED-LIUM long form has only 8 talks, so one talk that changes moves the score by several points.

## Results

WER lower is better; BLEU and accuracies higher is better (accuracies ×100). Both runs are shown.

| Benchmark | Metric | 8B | 70B |
|---|---|---|---|
| LibriSpeech test-clean | WER | 6.6 / 6.6 | 6.1 / 6.2 |
| LibriSpeech test-other | WER | 19.9 / 19.9 | 19.4 / 19.4 |
| VoxPopuli | WER | 11.2 / 11.2 | 10.3 / 10.3 |
| SPGISpeech | WER | 18.8 / 18.8 | 12.6 / 12.6 |
| TED-LIUM long form | WER | 35.0 / 35.0 | 42.8 / 47.8 ¹ |
| FLEURS English | WER | 15.1 / 15.1 | 14.1 / 14.1 |
| FLEURS German | WER | 16.1 / 16.1 | 17.8 / 17.8 |
| FLEURS French | WER | 22.4 / 22.4 | 21.1 / 21.1 |
| FLEURS Italian | WER | 7.6 / 7.6 | 7.3 / 7.2 |
| FLEURS Spanish | WER | 8.7 / 8.7 | 7.8 / 7.9 |
| FLEURS Polish | WER | 43.8 / 43.8 | 44.2 / 44.5 |
| FLEURS Ukrainian | WER | 34.9 / 34.9 | 31.5 / 31.7 |
| CoVoST2 en-zh test | BLEU | 15.3 / 15.3 | 13.1 / 13.1 |
| MMAU test-mini | accuracy | 55.2 / 55.2 | 55.8 / 55.9 |
| MuChoMusic | accuracy | 51.3 / 51.3 | 57.5 / 57.3 |
| Clotho-AQA test | exact match | 52.8 / 52.8 | 57.3 / 57.4 |
| VocalSound | accuracy | 16.1 / 16.1 | 5.0 / 5.3 ² |

¹ 8 talks. On the four longest talks the 70B stops after 40-71% of the transcript in both runs; in run 2 one talk also repeats a passage, which accounts for most of the difference.

² The task prompt does not list the six classes. The 70B mostly answers as if no audio were given (asks for a recording, or classifies the prompt text), so it scores below chance (16.7%); the score reflects the prompt more than audio recognition.

## Peer baselines

`scripts/audio_repro/submit_peers.sh` runs the open audio models of the comparison with the same generation setting as the Apertus runs (each task's declared cap, TED-LIUM long form 4096, greedy). They run on Hugging Face generate, not vLLM.

| Model | Revision | Backend and settings |
|---|---|---|
| `Qwen/Qwen2-Audio-7B-Instruct` | `0a095220` | `qwen2_audio`, batch size 8 (TED-LIUM long form 1), prod eval image |
| `Qwen/Qwen2.5-Omni-7B` | `ae9e1690` | `qwen2_5_omni`, batch size 1, the per-category system prompts of the audio results page, `attn_implementation=sdpa` on TED-LIUM long form, prod eval image with a `qwen-omni-utils==0.0.8` and `audioread==3.0.1` overlay |
| `moonshotai/Kimi-Audio-7B-Instruct` | `9a82a84c` | `kimi_audio`, batch size 1, archived 2026-05 image with the overlay from `build_kimi_overlay.sbatch` (Kimi-Audio `349251e1`) |

```bash
export PEER_MODEL_DIR=/path/with/hf-download-snapshots
bash scripts/audio_repro/submit_peers.sh qwen2_audio peers_qwen2_audio_r1
EXTRA_PYTHONPATH=/path/to/qwen-omni-overlay bash scripts/audio_repro/submit_peers.sh qwen2_5_omni peers_qwen2_5_omni_r1
EXTRA_PYTHONPATH=/path/to/kimi-overlay bash scripts/audio_repro/submit_peers.sh kimi_audio peers_kimi_audio_r1
```

Why the backend settings differ from the Apertus runs:

- **Batch size.** The launcher's default of 512 is a vLLM setting. At 512 Qwen2-Audio runs out of GPU memory and its backend returns empty answers for the whole batch; at 8 its outputs are identical to batch size 1 on 64 LibriSpeech samples.
- **Attention on TED-LIUM long form.** With the backend's default eager attention, Qwen2.5-Omni runs out of GPU memory on the four longest talks and returns empty transcripts. `sdpa` computes the same attention without materializing the full matrix.

The report's peer numbers come from the audio results page, except TED-LIUM long form, where the page ran the peers with the task's 256-token cap. TED-LIUM long form was rerun with these settings (lmms-eval 9aee58f2 from swiss-ai/lmms-eval#26, whose peer code matches the pinned commit apart from the TED-LIUM login flag):

| Model | TED-LIUM long form WER, every run |
|---|---|
| Qwen2-Audio | 99.6 in all 6 runs |
| Qwen2.5-Omni | 76.2, 76.2, 76.2, 76.2, 75.6 |
| Kimi-Audio | 61.8 in all 4 runs |

Kimi-Audio produces identical transcripts in every run. Qwen2-Audio's differ only on one talk, where it outputs a short garbage string in two variants with the same score. Qwen2.5-Omni with `sdpa` is not deterministic: 3 to 6 of the 8 transcripts match between runs, and in one run a talk continues further before stopping.

Qwen2-Audio accepts inputs of up to about 30 s and answers each 20-minute talk with a sentence or less; Kimi-Audio and Qwen2.5-Omni stop early on the longest talks.

Known limits: Qwen2-Audio stalls on one data-parallel rank on CoVoST2, so its CoVoST2 score is not available. In earlier runs (lmms-eval 0745eae6), Kimi-Audio's VoiceBench and MMSU answers came back empty; they are not used.

## Notes

CoVoST2 zh-en BLEU is near zero for both models (8B 1.1 in both runs, 70B 1.5 and 1.4): the translations are mostly unrelated to the references. It is not reported.
