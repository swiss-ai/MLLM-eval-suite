"""Extract the text backbone from an Apertus 1.5 multimodal HF release.

The Hub releases (swiss-ai/Apertus-v1.5-8B, -70B) package the discrete-
early-fusion backbone as Apertus1p5ForConditionalGeneration: the LM weights
live under model.language_model.*, the wavtokenizer is bundled as
model.audio_tokenizer.*, and the config nests the real LM config under
text_config. No inference engine needs any of that for text evaluation, so
this writes a standard ApertusForCausalLM checkpoint next to the HF cache:
renamed LM tensors, a flat apertus config, and the tokenizer files copied
through. Deterministic; safe to re-run (skips when the output looks complete).

Usage:
  python scripts/extract_text_backbone.py swiss-ai/Apertus-v1.5-8B <out_dir>
"""

import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

LM_PREFIX = "model.language_model."
KEEP_FLAT = ("lm_head.",)
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)


def main(repo_id: str, out_dir: str) -> None:
    out = Path(out_dir)
    if (out / "config.json").exists() and list(out.glob("*.safetensors")):
        print(f"{out} already extracted; skipping")
        return
    out.mkdir(parents=True, exist_ok=True)

    # The index is JSON, so fetch metadata first and pull only the shards that
    # actually carry text-backbone weights (the vision and audio tokenizer
    # shards contribute nothing and are over a gigabyte between them).
    src = Path(snapshot_download(repo_id, allow_patterns=["*.json", "*.jinja"]))
    index = json.loads((src / "model.safetensors.index.json").read_text())
    shards = sorted(
        {
            shard
            for key, shard in index["weight_map"].items()
            if key.startswith(LM_PREFIX) or key.startswith(KEEP_FLAT)
        }
    )
    src = Path(
        snapshot_download(repo_id, allow_patterns=["*.json", "*.jinja", *shards])
    )

    full = json.loads((src / "config.json").read_text())

    # The release understands the full multimodal vocab but generates text
    # only: embed_tokens covers ~267k ids while lm_head outputs the 131k text
    # ids. Softmax is already computed over the head's rows, so the exact text
    # view truncates the embedding to match; text prompts never use ids above
    # the text vocab (chat special tokens sit at the bottom of the table).
    head_shard = index["weight_map"]["lm_head.weight"]
    with safe_open(src / head_shard, framework="pt") as f:
        text_vocab = f.get_slice("lm_head.weight").get_shape()[0]
    print(f"text vocab (lm_head rows): {text_vocab}")

    cfg = dict(full["text_config"])
    cfg["architectures"] = ["ApertusForCausalLM"]
    cfg["model_type"] = "apertus"
    cfg["vocab_size"] = text_vocab
    cfg.setdefault("tie_word_embeddings", full.get("tie_word_embeddings", False))
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    weight_map = {}
    for shard in shards:
        kept = {}
        with safe_open(src / shard, framework="pt") as f:
            for key in f.keys():
                if key.startswith(LM_PREFIX):
                    new_key = "model." + key[len(LM_PREFIX):]
                    if new_key == "model.embed_tokens.weight":
                        # Slice on the file so the discarded multimodal rows are
                        # never read, let alone kept alive by the saved view.
                        kept[new_key] = f.get_slice(key)[:text_vocab]
                    else:
                        kept[new_key] = f.get_tensor(key)
                elif key.startswith(KEEP_FLAT):
                    kept[key] = f.get_tensor(key)
        if kept:
            save_file(kept, str(out / shard), metadata={"format": "pt"})
            weight_map.update({k: shard for k in kept})
            print(f"{shard}: kept {len(kept)} tensors")

    (out / "model.safetensors.index.json").write_text(
        json.dumps(
            {"metadata": {"total_size": sum((out / s).stat().st_size for s in set(weight_map.values()))},
             "weight_map": weight_map},
            indent=2,
        )
        + "\n"
    )

    for name in TOKENIZER_FILES:
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    print(f"text backbone written to {out}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
