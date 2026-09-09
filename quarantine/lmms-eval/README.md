Molmo-7B-D-0924 lmms-eval smoke run (2026-07-22) — DO NOT PUBLISH.

Ran via the generic vLLM backend with max_model_len=4096 (Molmo-1's native
context). Mechanically clean: no crash, 0% empty responses, 87% well-formed
single-letter answers. But the scores are implausible against this model's
trusted VLMEvalKit cells (CV-Bench-2D 72.9, HallusionBench 62.9, MathVista 49.5):

  mmstar   30.9   (4-choice chance floor is 25)
  chartqa  13.5

Answer distribution is skewed to "A" (38% of responses vs 29% of targets),
which points at a prompt/template mismatch rather than model capability.
Molmo-1 ships a chat_template in tokenizer_config.json; confirm the lmms-eval
vLLM path applies it before trusting any rerun.


EuroVLM-9B-Preview ocrbench_v2 + omnidocbench (2026-07-24) — DO NOT PUBLISH.

Non-termination, not incapability. 99.0% of ocrbench_v2 samples and 98.0% of
omnidocbench samples hit the 16384 max_new_tokens cap (median output = 16384,
EuroVLM's median on every working task is 2-6 tokens), and 91%/86% of the
predictions parse to empty:

  ocrbench_v2    1.45   (en 1.4 / cn 1.6; the few non-empty preds are "_", "-", "T")
  omnidocbench   0.01   (text_edit 1.00, table_teds 0.00, formula_edit 1.00)

The same model reads text fine in the same run — ocrbench (v1) 67.7, TextVQA
74.0, DocVQA 71.3, ChartQA 71.2 — and other models do not run away on the same
task (Gemma4-12B 0.9% capped, Apertus-70B 0.6%). Only these two long-form /
structured-output OCR tasks are affected, so the wrapper never emits a stop
token when no short answer format constrains it. Fix the stop-token/chat-template
handling before rerunning.


gemma-4-12b-it vqav2_val (2026-07-24) — DO NOT PUBLISH.

exact_match 0.0 over the full 214354 samples with 0% empty responses. The model
answers correctly; the wrapper leaves special tokens on the string, so every
comparison misses:

  pred='down<|im_end|><|endoftext|>'   pred='no<|im_end|><|endoftext|>'
  pred='1<|im_end|><|endoftext|>'      pred='tired<|im_end|>'

Rescoring the saved predictions with the harness metric after stripping the
tokens gives 80.37, which sits where this model belongs (Molmo2 77.7, Qwen3
81.8). Rerun with the fixed wrapper for a native cell.
