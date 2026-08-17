# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass(frozen=True)
class CollisionGuardVerificationData:
    fingerprint_payload: str
    raw_image_mode: str
    raw_image_size: tuple[int, int]
    raw_image_bytes: bytes
    raw_image_sha256: str
    resized_image_mode: str
    resized_image_size: tuple[int, int]
    resized_image_sha256: str


@dataclass(frozen=True)
class CollisionGuardCacheRecord:
    prompt: str
    fingerprint_payload: str
    raw_image_mode: str
    raw_image_size: tuple[int, int]
    raw_image_bytes: bytes
    raw_image_sha256: str
    resized_image_mode: str
    resized_image_size: tuple[int, int]
    resized_image_sha256: str
