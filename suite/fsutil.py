import hashlib
import os
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def file_stat(path: Path) -> dict:
    st = Path(path).stat()
    return {"size": st.st_size, "mtime": int(st.st_mtime)}


def count_files(path: Path, limit: int) -> int:
    """Number of files under path, stopping early once limit is reached."""
    n = 0
    for _root, _dirs, files in os.walk(path):
        n += len(files)
        if n >= limit:
            break
    return n
