Engine-independent runtime modules, importable by both harness checkouts
(both job scripts put this directory on PYTHONPATH).

- `apertus_image_tokenizer/` — harness-side Apertus image tokenization:
  Emu3.5 VQ encode + frame layout + sqlite token cache, vendored from the
  archived May engine image (vLLM-internal imports shimmed out). Wrappers call
  `splice_frames(prompt, images, tokenizer)`; the engine only sees token ids.
  Vendored counterpart may still exist inside old engine images — the sqlite
  cache format is shared, so schema changes must stay compatible.
