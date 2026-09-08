import hashlib
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
