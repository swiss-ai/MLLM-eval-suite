# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
from collections.abc import Sequence
from typing import Any

from PIL import Image


def hash_image_payload(
    mode: str,
    size: tuple[int, int],
    pixel_bytes: bytes,
) -> str:
    hasher = hashlib.sha256()
    hasher.update(mode.encode("utf-8"))
    hasher.update(str(size[0]).encode("utf-8"))
    hasher.update(b"x")
    hasher.update(str(size[1]).encode("utf-8"))
    hasher.update(pixel_bytes)
    return hasher.hexdigest()


def hash_pil_image(image: Image.Image) -> str:
    # The tokenizer receives PIL RGB pixels; hashing mode+size+bytes captures
    # the exact effective tokenizer input for cache keying.
    return hash_image_payload(image.mode, image.size, image.tobytes())


def logical_key_payload(logical_key: Sequence[Any]) -> str:
    return json.dumps(
        list(logical_key),
        sort_keys=False,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def stable_cache_key(logical_key: Sequence[Any]) -> tuple[str, str]:
    payload = logical_key_payload(logical_key)
    key = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return key, payload
