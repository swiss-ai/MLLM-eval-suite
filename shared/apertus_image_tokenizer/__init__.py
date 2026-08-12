import os

from .tokenizer import ApertusImageTokenizer

_shared_tokenizer = None


def default_mm_kwargs():
    kwargs = {}
    hub = os.environ.get("APERTUS_VQ_HUB")
    if not hub:
        for env in ("VLLM_APERTUS_MODELS_CACHE", "LMMS_EVAL_MODELS_CACHE"):
            cache = os.environ.get(env)
            if cache and os.path.isdir(os.path.join(cache, "BAAI/Emu3.5-VisionTokenizer")):
                hub = os.path.join(cache, "BAAI/Emu3.5-VisionTokenizer")
                break
    if hub:
        kwargs["apertus_vq_hub"] = hub
    return kwargs


def splice_frames(prompt, images, tokenizer, mm_processor_kwargs=None):
    """Replace each image placeholder in `prompt` with its framed visual-token
    text from the Emu3.5 VQ tokenizer, so the engine only ever sees token ids.
    Fails loudly on placeholder/image count mismatch."""
    global _shared_tokenizer
    if _shared_tokenizer is None:
        _shared_tokenizer = ApertusImageTokenizer()
    mm_kwargs = default_mm_kwargs() if mm_processor_kwargs is None else mm_processor_kwargs
    frames = _shared_tokenizer.encode_images(images, tokenizer=tokenizer, mm_processor_kwargs=mm_kwargs)
    aliases = _shared_tokenizer.placeholder_aliases(tokenizer, mm_kwargs)
    placeholder = next((a for a in aliases if a in prompt), None)
    if placeholder is None or prompt.count(placeholder) != len(frames):
        raise ValueError(
            f"image placeholder mismatch: aliases={aliases} "
            f"count={None if placeholder is None else prompt.count(placeholder)} images={len(frames)}"
        )
    for frame in frames:
        prompt = prompt.replace(placeholder, frame, 1)
    return prompt
