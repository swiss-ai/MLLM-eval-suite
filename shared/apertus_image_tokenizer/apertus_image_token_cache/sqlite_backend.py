# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sqlite3
import threading
from pathlib import Path

SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS image_token_cache (
    mode TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    value BLOB NOT NULL,
    PRIMARY KEY (mode, cache_key)
);
"""


class SQLiteImagePromptBackend:
    def __init__(
        self,
        *,
        db_path: Path,
        busy_timeout_ms: int,
        mmap_size: int,
        readonly: bool = False,
    ) -> None:
        self._db_path = db_path.resolve()
        self._busy_timeout_ms = max(1, int(busy_timeout_ms))
        self._mmap_size = max(0, int(mmap_size))
        self._readonly = bool(readonly)
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.Lock()

    @property
    def db_path(self) -> Path:
        return self._db_path

    @property
    def readonly(self) -> bool:
        return self._readonly

    def _ensure_connection(self) -> sqlite3.Connection:
        with self._lock:
            if self._connection is not None:
                return self._connection

            connect_target = str(self._db_path)
            connect_kwargs: dict[str, object] = {
                "timeout": self._busy_timeout_ms / 1000.0,
                "isolation_level": None,
                "check_same_thread": False,
            }
            if self._readonly:
                connect_target = f"file:{self._db_path}?mode=ro"
                connect_kwargs["uri"] = True

            self._connection = sqlite3.connect(connect_target, **connect_kwargs)
            self._connection.execute(
                f"PRAGMA busy_timeout={self._busy_timeout_ms};"
            ).close()
            self._connection.execute("PRAGMA temp_store=MEMORY;").close()
            self._connection.execute(f"PRAGMA mmap_size={self._mmap_size};").close()
            if self._readonly:
                self._connection.execute("PRAGMA query_only=ON;").close()
            else:
                self._connection.execute("PRAGMA journal_mode=WAL;").fetchone()
                self._connection.execute("PRAGMA synchronous=NORMAL;").close()
                self._connection.execute(SQLITE_SCHEMA).close()
            return self._connection

    def get(self, *, mode: str, key: str) -> bytes | None:
        connection = self._ensure_connection()
        with self._lock:
            cursor = connection.execute(
                "SELECT value FROM image_token_cache WHERE mode=? AND cache_key=?;",
                (mode, key),
            )
            row = cursor.fetchone()
            cursor.close()
        if row is None:
            return None
        return bytes(row[0])

    def put_if_absent(self, *, mode: str, key: str, value: bytes) -> bool:
        if self._readonly:
            return False
        connection = self._ensure_connection()
        with self._lock:
            cursor = connection.execute(
                "INSERT OR IGNORE INTO image_token_cache (mode, cache_key, value) "
                "VALUES (?, ?, ?);",
                (mode, key, value),
            )
            inserted = cursor.rowcount > 0
            cursor.close()
            return inserted

    def preload_rows(
        self,
        *,
        mode: str,
        max_entries: int | None = None,
    ) -> list[tuple[str, bytes]]:
        if max_entries is not None and max_entries <= 0:
            return []

        connection = self._ensure_connection()
        with self._lock:
            if max_entries is None:
                cursor = connection.execute(
                    "SELECT cache_key, value FROM image_token_cache WHERE mode=?;",
                    (mode,),
                )
            else:
                cursor = connection.execute(
                    "SELECT cache_key, value FROM image_token_cache "
                    "WHERE mode=? LIMIT ?;",
                    (mode, max_entries),
                )
            rows = cursor.fetchall()
            cursor.close()

        return [(str(row[0]), bytes(row[1])) for row in rows]

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None
