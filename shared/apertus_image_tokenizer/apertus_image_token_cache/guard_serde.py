# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import base64
import json
from typing import Any

from .records import CollisionGuardCacheRecord

_SERDE_VERSION = "apertus-collision-guard-v2"


def _require_str(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Missing or invalid string field: {key}")
    return value


def _require_image_meta(
    payload: dict[str, Any],
    *,
    key: str,
    include_bytes: bool,
) -> tuple[str, tuple[int, int], str, bytes | None]:
    meta = payload.get(key)
    if not isinstance(meta, dict):
        raise ValueError(f"Missing or invalid object field: {key}")

    mode = _require_str(meta, "mode")
    sha256 = _require_str(meta, "sha256")
    size = meta.get("size")
    if (
        not isinstance(size, list)
        or len(size) != 2
        or not all(isinstance(dim, int) for dim in size)
    ):
        raise ValueError(f"Missing or invalid size field: {key}.size")

    decoded_bytes: bytes | None = None
    if include_bytes:
        bytes_b64 = _require_str(meta, "bytes_b64")
        decoded_bytes = base64.b64decode(bytes_b64.encode("ascii"), validate=True)

    return mode, (size[0], size[1]), sha256, decoded_bytes


def serialize_collision_guard_record(record: CollisionGuardCacheRecord) -> bytes:
    payload = {
        "version": _SERDE_VERSION,
        "prompt": record.prompt,
        "fingerprint_payload": record.fingerprint_payload,
        "raw_image": {
            "mode": record.raw_image_mode,
            "size": [record.raw_image_size[0], record.raw_image_size[1]],
            "sha256": record.raw_image_sha256,
            "bytes_b64": base64.b64encode(record.raw_image_bytes).decode("ascii"),
        },
        "resized_image": {
            "mode": record.resized_image_mode,
            "size": [record.resized_image_size[0], record.resized_image_size[1]],
            "sha256": record.resized_image_sha256,
        },
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def deserialize_collision_guard_record(value: bytes) -> CollisionGuardCacheRecord:
    payload = json.loads(value.decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Collision guard payload must be a JSON object.")

    version = _require_str(payload, "version")
    if version != _SERDE_VERSION:
        raise ValueError(f"Unsupported collision guard payload version: {version}")

    raw_mode, raw_size, raw_sha, raw_bytes = _require_image_meta(
        payload, key="raw_image", include_bytes=True
    )
    assert raw_bytes is not None
    resized_mode, resized_size, resized_sha, _ = _require_image_meta(
        payload, key="resized_image", include_bytes=False
    )

    return CollisionGuardCacheRecord(
        prompt=_require_str(payload, "prompt"),
        fingerprint_payload=_require_str(payload, "fingerprint_payload"),
        raw_image_mode=raw_mode,
        raw_image_size=raw_size,
        raw_image_bytes=raw_bytes,
        raw_image_sha256=raw_sha,
        resized_image_mode=resized_mode,
        resized_image_size=resized_size,
        resized_image_sha256=resized_sha,
    )
