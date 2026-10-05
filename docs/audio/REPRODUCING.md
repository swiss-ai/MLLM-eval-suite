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

The results below were produced on cce67fa7, the same changes on top of f30dc97 (`codex/pr2-preserve-audio`). Against 649a28e2 (6bd2671b plus the repository's automatic black/isort commit), the code the audio tasks use (the seven FLEURS tasks, CoVoST2, the cap rule, the Apertus wrapper) is byte-identical; the branches differ only in `main`'s mtvqa changes, #12's other FLEURS languages and splits, test formatting, and `main`'s `emu3p5.py`.

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
| `muchomusic` | `lmms-lab-audio/muchomusic` | test | 4096 (no task cap; backend fallback) |
| `clotho_aqa` | `lmms-lab-audio/ClothoAQA` | val, test (filtered) | 8 |
| `vocalsound_test` | `lmms-lab-audio/vocalsound` | test | 4096 (no task cap; backend fallback) |

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
| lmms-eval `--batch_size` | 512 | 512 |
| Seed | 1 | 1 |
| Environment | `VLLM_MAX_AUDIO_DECODE_DURATION_S=3600` | same |

`scripts/audio_repro/submit_apertus.sh` sets the model-specific values and the TED-LIUM overrides; the launcher defaults supply the rest (backend, tokenizer, `max_model_len`, batch size, seed).

## Why these settings

- **Image.** The release checkpoints use the Transformers 5.14 layout (`Apertus1p5ForConditionalGeneration`) with `lm_head` pruned to the 131,072 text ids. The prod image's vLLM cannot load them (vocab-size assertion in `vocab_parallel_embedding`); the release vLLM can.
- **Generation.** Each task's declared cap, temperature 0. TED-LIUM long-form gets 4096, because its task cap of 256 truncates the transcripts of its 20-minute talks (about 3,000 words); tasks that declare no cap (MuChoMusic, VocalSound) use the backend fallback of 4096. With the old `max()` rule, one looping sample added about 10 WER to 70B FLEURS Italian (16.2 against 7.2).
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

## Reproducibility check (2026-10-05)

Each configuration ran twice; `compare_runs.py` compares the generated text sample by sample.

| Model | Identical outputs, run 1 vs run 2 | Score difference |
|---|---|---|
| 8B | 100% on all 16 tasks | none |
| 70B | 67-99% per task | within 0.5, except TED-LIUM long-form (69.3 / 63.3) |

8B is deterministic. 70B with TP=4 depends slightly on batch timing; scores agree except on TED-LIUM long-form, which has only 8 talks.

## Results

WER lower is better; BLEU and accuracies higher is better (accuracies ×100). 70B shows both runs.

| Benchmark | Metric | 8B | 70B |
|---|---|---|---|
| LibriSpeech test-clean | WER | 6.3 | 6.0 / 6.0 |
| LibriSpeech test-other | WER | 19.3 | 18.8 / 18.8 |
| VoxPopuli | WER | 20.3 ¹ | 10.5 / 10.4 |
| SPGISpeech | WER | 16.7 | 12.4 / 12.4 |
| TED-LIUM long form | WER | 32.3 | 69.3 / 63.3 ² |
| FLEURS English | WER | 14.8 | 14.5 / 14.2 |
| FLEURS German | WER | 15.7 | 18.0 / 17.9 |
| FLEURS French | WER | 22.2 | 21.1 / 21.2 |
| FLEURS Italian | WER | 7.6 | 7.2 / 7.2 |
| FLEURS Spanish | WER | 8.4 | 8.0 / 8.0 |
| FLEURS Polish | WER | 44.0 | 44.4 / 44.9 |
| FLEURS Ukrainian | WER | 34.2 | 32.0 / 31.6 |
| CoVoST2 en-zh test | BLEU | 16.4 | 13.3 / 13.3 |
| MMAU test-mini | accuracy | 53.5 | 55.3 / 55.7 |
| MuChoMusic | accuracy | 51.5 | 58.5 / 58.7 |
| Clotho-AQA test | exact match | 54.2 | 58.7 / 58.7 |
| VocalSound | accuracy | 16.4 | 14.0 / 13.9 ³ |

¹ One sample loops to VoxPopuli's own 4096-token cap in both runs, adding about 9 WER; without it the score is about 11.2.

² 8 talks; the output for some talks differs between runs. Report as a range.

³ The task prompt does not list the six classes; the score reflects the prompt more than audio recognition.

## Notes

CoVoST2 zh-en BLEU is near zero for both models (8B 1.6): the translations are fluent but mostly unrelated to the references, while en-zh and Mandarin ASR work. This points to the zh-en data and is not reported.
