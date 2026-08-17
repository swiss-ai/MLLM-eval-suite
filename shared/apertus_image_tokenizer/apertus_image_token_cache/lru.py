# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections import OrderedDict
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class ThreadSafeLRUCache(Generic[K, V]):
    def __init__(self, max_entries: int) -> None:
        self._max_entries = int(max_entries)
        self._bounded = self._max_entries > 0
        self._lock = threading.Lock()
        self._items: OrderedDict[K, V] = OrderedDict()

    def get(self, key: K) -> V | None:
        with self._lock:
            if key not in self._items:
                return None
            value = self._items.pop(key)
            self._items[key] = value
            return value

    def put(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._items:
                self._items.pop(key)
            self._items[key] = value
            while self._bounded and len(self._items) > self._max_entries:
                self._items.popitem(last=False)

    def remove(self, key: K) -> None:
        with self._lock:
            self._items.pop(key, None)
