# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass
from pathlib import Path


def _coerce_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
    return default


def _coerce_int(value: object, *, default: int, min_value: int) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    if parsed < min_value:
        return default
    return parsed


@dataclass(frozen=True)
class ApertusImageTokenCacheConfig:
    cache_dir: Path | None
    collision_guard: bool
    memory_cache_size: int
    sqlite_busy_timeout_ms: int
    sqlite_mmap_size: int
    debug_logging: bool
    disabled_reason: str | None
    preload: bool = False
    readonly: bool = False
    write_misses: bool = True

    CACHE_DIR_ENV = "VLLM_APERTUS_IMAGE_TOKEN_CACHE_DIR"
    COLLISION_GUARD_ENV = "VLLM_APERTUS_IMAGE_TOKEN_CACHE_COLLISION_GUARD"
    MEMORY_SIZE_ENV = "VLLM_APERTUS_IMAGE_TOKEN_MEMORY_CACHE_SIZE"
    SQLITE_BUSY_TIMEOUT_ENV = "VLLM_APERTUS_IMAGE_TOKEN_SQLITE_BUSY_TIMEOUT_MS"
    SQLITE_MMAP_SIZE_ENV = "VLLM_APERTUS_IMAGE_TOKEN_SQLITE_MMAP_SIZE"
    DEBUG_ENV = "VLLM_APERTUS_IMAGE_TOKEN_CACHE_DEBUG"
    PRELOAD_ENV = "VLLM_APERTUS_IMAGE_TOKEN_CACHE_PRELOAD"
    READONLY_ENV = "VLLM_APERTUS_IMAGE_TOKEN_CACHE_READONLY"
    WRITE_MISSES_ENV = "VLLM_APERTUS_IMAGE_TOKEN_CACHE_WRITE_MISSES"

    DEFAULT_MEMORY_CACHE_SIZE = 131072
    DEFAULT_SQLITE_BUSY_TIMEOUT_MS = 5000
    DEFAULT_SQLITE_MMAP_SIZE = 1024 * 1024 * 1024
    DEFAULT_SQLITE_SUBDIR = "image_tokens"
    DEFAULT_SQLITE_DB_FILENAME = "apertus_image_token_cache.sqlite3"

    @property
    def enabled(self) -> bool:
        return self.cache_dir is not None

    @property
    def sqlite_db_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        return (
            self.cache_dir
            / self.DEFAULT_SQLITE_SUBDIR
            / self.DEFAULT_SQLITE_DB_FILENAME
        )

    @classmethod
    def from_env(cls) -> "ApertusImageTokenCacheConfig":
        memory_cache_size = _coerce_int(
            os.getenv(cls.MEMORY_SIZE_ENV),
            default=cls.DEFAULT_MEMORY_CACHE_SIZE,
            min_value=1,
        )
        sqlite_busy_timeout_ms = _coerce_int(
            os.getenv(cls.SQLITE_BUSY_TIMEOUT_ENV),
            default=cls.DEFAULT_SQLITE_BUSY_TIMEOUT_MS,
            min_value=1,
        )
        sqlite_mmap_size = _coerce_int(
            os.getenv(cls.SQLITE_MMAP_SIZE_ENV),
            default=cls.DEFAULT_SQLITE_MMAP_SIZE,
            min_value=0,
        )
        collision_guard = _coerce_bool(
            os.getenv(cls.COLLISION_GUARD_ENV),
            default=False,
        )
        debug_logging = _coerce_bool(
            os.getenv(cls.DEBUG_ENV),
            default=False,
        )
        preload = _coerce_bool(
            os.getenv(cls.PRELOAD_ENV),
            default=False,
        )
        readonly = _coerce_bool(
            os.getenv(cls.READONLY_ENV),
            default=False,
        )
        write_misses = _coerce_bool(
            os.getenv(cls.WRITE_MISSES_ENV),
            default=True,
        )

        cache_dir_value = os.getenv(cls.CACHE_DIR_ENV)
        if not isinstance(cache_dir_value, str) or not cache_dir_value.strip():
            return cls(
                cache_dir=None,
                collision_guard=collision_guard,
                memory_cache_size=memory_cache_size,
                sqlite_busy_timeout_ms=sqlite_busy_timeout_ms,
                sqlite_mmap_size=sqlite_mmap_size,
                debug_logging=debug_logging,
                disabled_reason=f"{cls.CACHE_DIR_ENV} is unset or empty.",
                preload=preload,
                readonly=readonly,
                write_misses=write_misses,
            )

        cache_dir = Path(os.path.expandvars(cache_dir_value.strip())).expanduser()
        if not cache_dir.is_dir():
            return cls(
                cache_dir=None,
                collision_guard=collision_guard,
                memory_cache_size=memory_cache_size,
                sqlite_busy_timeout_ms=sqlite_busy_timeout_ms,
                sqlite_mmap_size=sqlite_mmap_size,
                debug_logging=debug_logging,
                disabled_reason=(
                    f"{cls.CACHE_DIR_ENV} points to a non-existing or non-directory "
                    f"path: {cache_dir}"
                ),
                preload=preload,
                readonly=readonly,
                write_misses=write_misses,
            )

        return cls(
            cache_dir=cache_dir,
            collision_guard=collision_guard,
            memory_cache_size=memory_cache_size,
            sqlite_busy_timeout_ms=sqlite_busy_timeout_ms,
            sqlite_mmap_size=sqlite_mmap_size,
            debug_logging=debug_logging,
            disabled_reason=None,
            preload=preload,
            readonly=readonly,
            write_misses=write_misses,
        )
