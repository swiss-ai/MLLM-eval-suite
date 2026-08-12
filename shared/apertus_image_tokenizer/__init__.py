import os
import threading
from functools import lru_cache

from .tokenizer import ApertusImageTokenizer

_init_lock = threading.Lock()
_warm = False


@lru_cache(maxsize=1)
def _mm_kwargs():
    # Only an explicit override travels from here; otherwise the tokenizer's
    # own DEFAULT_VQ_HUB + ensure_local_emu35_weights resolve (and validate)
    # the cached weights, so there is exactly one resolution path.
    hub = os.environ.get("APERTUS_VQ_HUB")
    return {"apertus_vq_hub": hub} if hub else {}


@lru_cache(maxsize=1)
def _shared_state():
    return ApertusImageTokenizer(), _mm_kwargs()


@lru_cache(maxsize=4)
def _aliases_for(tokenizer_id):
    tokenizer, mm_kwargs = _TOKENIZERS[tokenizer_id], _mm_kwargs()
    return ApertusImageTokenizer.placeholder_aliases(tokenizer, mm_kwargs)


_TOKENIZERS = {}


def splice_frames(prompt, images, tokenizer):
    """Replace each image placeholder in `prompt` with its framed visual-token
    text from the Emu3.5 VQ tokenizer, so the engine only ever sees token ids.
    Fails loudly on placeholder/image count mismatch."""
    global _warm
    image_tokenizer, mm_kwargs = _shared_state()
    # The VQ tokenizer lazy-loads on first encode via a module-import dance
    # that is not safe under concurrent first calls; serialize until warm.
    if not _warm:
        with _init_lock:
            frames = image_tokenizer.encode_images(images, tokenizer=tokenizer, mm_processor_kwargs=mm_kwargs)
            _warm = True
    else:
        frames = image_tokenizer.encode_images(images, tokenizer=tokenizer, mm_processor_kwargs=mm_kwargs)
    _TOKENIZERS[id(tokenizer)] = tokenizer
    aliases = _aliases_for(id(tokenizer))
    placeholder = next((a for a in aliases if a in prompt), None)
    parts = prompt.split(placeholder) if placeholder is not None else [prompt]
    if len(parts) - 1 != len(frames):
        raise ValueError(f"image placeholder mismatch: aliases={aliases} count={len(parts) - 1} images={len(frames)}")
    out = [parts[0]]
    for frame, part in zip(frames, parts[1:]):
        out.append(frame)
        out.append(part)
    return "".join(out)
