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

    src = Path(
        snapshot_download(
            repo_id,
            allow_patterns=["*.json", "*.safetensors", "*.jinja"],
        )
    )

    full = json.loads((src / "config.json").read_text())
    cfg = dict(full["text_config"])
    cfg["architectures"] = ["ApertusForCausalLM"]
    cfg["model_type"] = "apertus"
    cfg.setdefault("tie_word_embeddings", full.get("tie_word_embeddings", False))
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")

    index = json.loads((src / "model.safetensors.index.json").read_text())
    weight_map = {}
    for shard in sorted(set(index["weight_map"].values())):
        kept = {}
        with safe_open(src / shard, framework="pt") as f:
            for key in f.keys():
                if key.startswith(LM_PREFIX):
                    kept["model." + key[len(LM_PREFIX):]] = f.get_tensor(key)
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
