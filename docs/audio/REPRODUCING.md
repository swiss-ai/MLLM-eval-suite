# Reproducing the Apertus 1.5 audio evaluation

This branch evaluates the public release checkpoints `swiss-ai/Apertus-v1.5-8B`
and `swiss-ai/Apertus-v1.5-70B` (and the audio peer baselines) on the audio
benchmarks of `docs/audio/audio_benchmark_results.html`, with every input
pinned and one fixed generation setting. It is based on `audio-results`
(8ac19a5), the code the results page was produced from.

## Pinned inputs

| Input | Pin |
|---|---|
| Suite | this branch |
| lmms-eval | `swiss-ai/lmms-eval` branch `ahadinia/audio-release-repro` @ abe6c324 (three commits on f30dc97, below) |
| Image | `apertus-vllm-release-eval.sqsh`, sha256 `578ee90b642833c21509fa857e8581247fc89b6a218a26f82b142192478dcb4c`, built from `dockerfiles/Dockerfile.vllm-apertus-release-eval` on `ghcr.io/swiss-ai/vllm_apertus_1.5_release:latest-arm64` |
| Weights | `swiss-ai/Apertus-v1.5-8B` @ a411d838, `swiss-ai/Apertus-v1.5-70B` @ 59e744e3; every file's sha256 matches the Hub |
| Peers | `Qwen/Qwen2-Audio-7B-Instruct` @ 0a095220, `Qwen/Qwen2.5-Omni-7B` @ ae9e1690, run in the default prod image |
| Backend | `apertus_1p5_vllm` (suite default) with the suite's Apertus tokenizer and chat template |

lmms-eval changes on top of f30dc97:

1. 74a74363: per-language FLEURS tasks (`fleurs_en_us`, `de_de`, `fr_fr`,
   `it_it`, `es_419`, `pl_pl`, `uk_ua`, copied from the unmerged
   `google-fluers` branch), and `dataset_name: default` for CoVoST2 en-zh
   (`lmms-lab-audio/covost2_en-zh` no longer has an `en_zh` config).
2. cce67fa7: a task's `max_new_tokens` is used as given. The vLLM backend took
   `max(task cap, 4096)`, so every cap of 256 or less silently became 4096.
3. abe6c324: peer backends run audio-only tasks (Qwen2.5-Omni imported
   moviepy at load; Qwen2-Audio got `(channels, samples)` audio from
   VoiceBench/MMSU and now downmixes to mono).

## Why these settings

- **Image.** The release checkpoints use the Transformers 5.14 layout
  (`Apertus1p5ForConditionalGeneration`) with `lm_head` pruned to the 131,072
  text ids. The prod image's vLLM cannot load them (vocab-size assertion in
  `vocab_parallel_embedding`); the release vLLM can.
- **Generation.** Each task's declared cap, `temperature 0`. TED-LIUM long-form
  gets 4096, the cap the results page states for it; tasks that declare no cap
  (MuChoMusic, VocalSound) use the backend fallback of 4096. With the old
  `max()` rule, one looping sample added about 10 WER to 70B FLEURS Italian
  (16.2 against 7.2).
- **70B.** TP=4 with CUDA graphs. On this vLLM build the fused all-reduce +
  RMSNorm pass (`fuse_allreduce_rms`) fails graph capture at batch sizes up to
  128 tokens with an illegal memory access, so only that pass is disabled.
  Running eager instead gives the same scores within about 1 point.
- **TED-LIUM long-form.** Talks run about 22 minutes, 36k-52k audio tokens
  each: `VLLM_MAX_AUDIO_DECODE_DURATION_S=3600`, `max_num_batched_tokens=65536`
  (the encoder cache follows it), and on 70B `gpu_memory_utilization=0.75`
  (CUDA graphs need memory outside vLLM's share).
- **VoiceBench.** AlpacaEval, CommonEval, WildVoice and SD-QA are scored by a
  GPT judge and need `API_TYPE=openai` and `OPENAI_API_KEY`; the scripts run
  the judge-free subsets (AdvBench, BBH, IFEval, MMSU, OpenBookQA) and MMSU.

## Steps

```bash
git clone --recurse-submodules -b ahadinia/audio-release-repro https://github.com/swiss-ai/MLLM-eval-suite
cd MLLM-eval-suite

# 1. Image (once): build it, check the sha256 above, and point `image =` in
#    toml/shared/apertus-vllm-release-eval.toml at it.
sbatch --account=infra01 --nodes=1 --exclusive --time=04:00:00 \
  dockerfiles/build_release_eval_image.sh "$PWD" "$PWD/cache/image-builds/release-eval"

# 2. Host Python: the launcher needs python3 >= 3.8 (login nodes may default to 3.6).
uv venv --python 3.12 ~/venvs/mllm-eval && source ~/venvs/mllm-eval/bin/activate

# 3. Submit (the scripts refuse to run from a checkout with local changes).
export CKPT_DIR=/path/with/Apertus-v1.5-8B-and-70B
for size in 8b 70b; do
  bash scripts/audio_repro/submit_apertus.sh $size core  apertus_${size}_r1
  bash scripts/audio_repro/submit_apertus.sh $size voice apertus_${size}_voice_r1
done
export PEER_MODEL_DIR=/path/with/peer/snapshots
bash scripts/audio_repro/submit_peers.sh qwen2_audio  peers_qwen2_audio_r1
bash scripts/audio_repro/submit_peers.sh qwen2_5_omni peers_qwen2_5_omni_r1

# 4. Repeat with a second run id, then compare the two runs.
python scripts/audio_repro/compare_runs.py \
  results/lmms-eval/Apertus-v1.5-8B/apertus_8b_r1 results/lmms-eval/Apertus-v1.5-8B/apertus_8b_r2
```

To compare against the old cap behaviour, check out lmms-eval 74a74363 (the
commit before the cap fix) and submit again.

## Reproducibility check (2026-10-05)

Each configuration ran twice; `compare_runs.py` compares the generated text
sample by sample.

| Model | Identical outputs, run 1 vs run 2 | Score difference |
|---|---|---|
| 8B | 100% on all 16 tasks | none |
| 70B | 67-99% per task | within 0.5, except TED-LIUM long-form (69.3 / 63.3) |

8B is deterministic. 70B with TP=4 depends slightly on batch timing; scores
agree except on TED-LIUM long-form, which has only 8 talks.

## Results

WER lower is better; BLEU and accuracies higher is better (accuracies ×100).
70B shows both runs.

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

¹ One sample loops to VoxPopuli's own 4096-token cap in both runs, adding about
9 WER; without it the score is about 11.2.
² 8 talks; the output for some talks differs between runs. Report as a range.
³ The task prompt does not list the six classes; the score reflects the prompt
more than audio recognition.

## Differences from the results page

28 of the 34 Apertus values above are within 2 points of the page's
`swiss-ai/Apertus-v1.5-8B` and `-70B` columns. Larger differences: 8B
VoxPopuli (+8.7, the one looping sample), 70B TED-LIUM long-form (+14 to +20,
unstable), and 70B VocalSound (14.0 against 4.9). The page does not record how
those columns were run.

The page author's 2026-09-23 "audio-official" runs used a different method:
the prod image, an unpushed lmms-eval commit (5d05caf3, branch
`codex/audio-review-20260923`), a rewritten model config
(`ApertusAudioReleaseForCausalLM`, `output_vocab_size` 131072) over the same
weight files, and client-side audio tokenization. They agree with the results
above within about 1.5 points except 70B VocalSound (5.1 official). That
method cannot be rerun from public code until the commit and its audio
tokenizer codebase are published.

CoVoST2 zh-en BLEU is near zero for both models (8B 1.6): the translations are
fluent but mostly unrelated to the references, while en-zh and Mandarin ASR
work. This points to the zh-en data and is not reported.
