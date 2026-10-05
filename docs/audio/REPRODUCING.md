# Reproducing the audio evaluation of the Apertus 1.5 release checkpoints

This branch evaluates the public release checkpoints `swiss-ai/Apertus-v1.5-8B` and `swiss-ai/Apertus-v1.5-70B` on 16 audio benchmarks (speech recognition, speech translation, audio understanding, audio question answering and sound classification), with every input pinned and one fixed generation setting. It is based on `audio-results` (8ac19a5).

## Pinned inputs

| Input | Pin |
|---|---|
| Suite | this branch |
| lmms-eval | `swiss-ai/lmms-eval` branch `ahadinia/audio-release-ckpt` @ 649a28e2 (swiss-ai/lmms-eval#27, on `main`; below) |
| Image | `apertus-vllm-release-eval.sqsh`, sha256 `578ee90b642833c21509fa857e8581247fc89b6a218a26f82b142192478dcb4c`, built from `dockerfiles/Dockerfile.vllm-apertus-release-eval` on `ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64` |
| Weights | `swiss-ai/Apertus-v1.5-8B` @ a411d838, `swiss-ai/Apertus-v1.5-70B` @ 59e744e3; every file's sha256 matches the Hub |
| Backend | `apertus_1p5_vllm` (suite default) with the suite's Apertus tokenizer and chat template |

lmms-eval commits on top of `main`:

1. de31accd, 00a3ae16: Apertus thinking mode and audio chat in the `apertus_1p5_vllm` wrapper (from `codex/pr2-preserve-audio`).
2. 255f9d18: the `google_fluers` per-language FLEURS tasks (from swiss-ai/lmms-eval#12); this evaluation uses `fleurs_en_us`, `de_de`, `fr_fr`, `it_it`, `es_419`, `pl_pl` and `uk_ua`.
3. 37e5ae90: CoVoST2 en-zh loads the dataset's `default` config (`lmms-lab-audio/covost2_en-zh` no longer has `en_zh`).
4. 6bd2671b: a task's `max_new_tokens` is used as given. The vLLM backend took `max(task cap, 4096)`, so every cap of 256 or less silently became 4096.

The results below come from two runs per model (2026-10-05) from a fresh clone of this branch at 530b67d (lmms-eval 649a28e2), with all caches empty: datasets, vLLM compilation and image tokens.

## Configuration

These are the settings recorded in the result files of the reported runs.

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
| Tokenizer and chat template | `/capstor/store/cscs/swissai/infra01/MLLM/tokenizer/apertus_emu3.5_wavtok_instruct_thinking_token_fixed` (and its `chat_template.jinja`) | same |
| Workers | 4 data-parallel processes, one GPU each (`tensor_parallel_size=1`) | 1 process, `tensor_parallel_size=4` |
| CUDA graphs | on (`enforce_eager=false`) | on, with `compilation_config={"pass_config":{"fuse_allreduce_rms":false}}` |
| `gpu_memory_utilization` | 0.6 | 0.85 (TED-LIUM long-form: 0.75) |
| `max_model_len` | 131072, with `hf_overrides={"max_position_embeddings":131072}` | same |
| `max_num_batched_tokens` | 49152 (TED-LIUM long-form: 65536) | 49152 (TED-LIUM long-form: 65536) |
| `enable_prefix_caching` | true | true |
| Model-level `max_new_tokens` | 4096, set in `submit_apertus.sh` (`--extra-model-args max_new_tokens=4096`). It applies only to tasks that declare no cap (MuChoMusic, VocalSound); a task's own cap takes precedence. | same |
| lmms-eval `--batch_size` | 512 | 512 |
| Seed | 1 | 1 |
| Environment | `VLLM_MAX_AUDIO_DECODE_DURATION_S=3600` | same |

`scripts/audio_repro/submit_apertus.sh` sets the model-specific values and the TED-LIUM overrides; the launcher defaults supply the rest (backend, tokenizer, `max_model_len`, batch size, seed).

## Why these settings

- **Image.** The release checkpoints use the Transformers 5.14 layout (`Apertus1p5ForConditionalGeneration`) with `lm_head` pruned to the 131,072 text ids. The prod image's vLLM cannot load them (vocab-size assertion in `vocab_parallel_embedding`); the release vLLM can.
- **Generation.** Each task's declared cap, temperature 0. TED-LIUM long-form gets 4096, because its task cap of 256 truncates the transcripts of its 20-minute talks (about 3,000 words); tasks that declare no cap (MuChoMusic, VocalSound) use the model-level cap of 4096 set in `submit_apertus.sh` (see Engine). With the old `max()` rule, one looping sample added about 10 WER to 70B FLEURS Italian (16.2 against 7.2).
- **70B.** TP=4 with CUDA graphs. On this vLLM build the fused all-reduce + RMSNorm pass (`fuse_allreduce_rms`) fails graph capture at batch sizes up to 128 tokens with an illegal memory access, so only that pass is disabled. Running eager instead gives the same scores within about 1 point.
- **TED-LIUM long-form.** Talks run about 22 minutes, 36k-52k audio tokens each: `VLLM_MAX_AUDIO_DECODE_DURATION_S=3600`, `max_num_batched_tokens=65536` (the encoder cache follows it), and on 70B `gpu_memory_utilization=0.75` (CUDA graphs need memory outside vLLM's share).

## Steps

```bash
git clone --recurse-submodules -b ahadinia/audio-release-ckpt https://github.com/swiss-ai/MLLM-eval-suite
cd MLLM-eval-suite

# 1. Image (once): build it, check the sha256 above, and point `image =` in toml/shared/apertus-vllm-release-eval.toml at it.
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

To compare against the old cap rule, check out lmms-eval 37e5ae90 (the commit before the cap fix) and submit again.

## Reproducibility check

Each model ran twice; `compare_runs.py` compares the generated text sample by sample.

| Model | Identical outputs, run 1 vs run 2 | Largest score difference |
|---|---|---|
| 8B | 75-100% per task | 0.2, on MMAU test-mini; most tasks within 0.05 |
| 70B | 38-99% per task | TED-LIUM long-form 0.8 and SPGISpeech 0.4; all other tasks within 0.3 |

Each job compiles its own vLLM kernels from an empty cache, and kernel autotuning differs slightly between jobs, so outputs are not byte-identical. They differ only on borderline samples and move scores by a few tenths at most. 70B with TP=4 also depends on batch timing. Two earlier runs per model (on cce67fa7, the same audio code with a warm shared cache) agree with these within 0.7 on every task except TED-LIUM long-form (8B 31.2 against 32.3; 70B 63.3 to 72.7 across the four runs), which has only 8 talks.

## Results

WER lower is better; BLEU and accuracies higher is better (accuracies ×100). Both runs are shown.

| Benchmark | Metric | 8B | 70B |
|---|---|---|---|
| LibriSpeech test-clean | WER | 6.4 / 6.3 | 6.0 / 6.0 |
| LibriSpeech test-other | WER | 19.4 / 19.3 | 18.8 / 18.9 |
| VoxPopuli | WER | 20.4 / 20.3 ¹ | 10.4 / 10.4 |
| SPGISpeech | WER | 17.2 / 17.2 | 12.8 / 12.4 |
| TED-LIUM long form | WER | 31.2 / 31.2 | 72.7 / 71.9 ² |
| FLEURS English | WER | 14.9 / 14.9 | 14.3 / 14.2 |
| FLEURS German | WER | 15.6 / 15.7 | 17.9 / 17.8 |
| FLEURS French | WER | 22.1 / 22.1 | 21.3 / 21.3 |
| FLEURS Italian | WER | 7.6 / 7.6 | 7.2 / 7.2 |
| FLEURS Spanish | WER | 8.4 / 8.5 | 8.0 / 8.0 |
| FLEURS Polish | WER | 44.6 / 44.6 | 45.0 / 45.0 |
| FLEURS Ukrainian | WER | 34.5 / 34.5 | 31.7 / 31.9 |
| CoVoST2 en-zh test | BLEU | 16.5 / 16.5 | 13.3 / 13.3 |
| MMAU test-mini | accuracy | 53.5 / 53.7 | 55.7 / 55.6 |
| MuChoMusic | accuracy | 51.8 / 51.8 | 58.3 / 58.6 |
| Clotho-AQA test | exact match | 54.2 / 54.2 | 58.6 / 58.6 |
| VocalSound | accuracy | 16.4 / 16.4 | 14.0 / 14.1 ³ |

¹ One sample loops to VoxPopuli's own 4096-token cap in both runs, adding about 9 WER; without it the score is about 11.2.

² 8 talks; the output for several talks differs between runs, and the score ranges from 63.3 to 72.7 across four runs. Report as a range.

³ The task prompt does not list the six classes; the score reflects the prompt more than audio recognition.

## Notes

CoVoST2 zh-en BLEU is near zero for both models (8B 1.6): the translations are fluent but mostly unrelated to the references, while en-zh and Mandarin ASR work. This points to the zh-en data and is not reported.
