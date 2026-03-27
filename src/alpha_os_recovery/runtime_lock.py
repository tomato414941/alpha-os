"""Advisory file locks for runtime entrypoints."""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .config import DATA_DIR


@dataclass(frozen=True)
class RuntimeLockBusy(Exception):
    """Raised when another runtime process already holds the lock."""

    path: Path

    def __str__(self) -> str:
        return f"lock already held: {self.path}"


def runtime_lock_path(runtime_name: str, asset_list: list[str]) -> Path:
    assets = "-".join(sorted(asset.upper() for asset in asset_list))
    lock_dir = DATA_DIR / "locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    return lock_dir / f"{runtime_name}_{assets}.lock"


@contextmanager
def hold_runtime_lock(path: Path) -> Iterator[None]:
    handle = path.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeLockBusy(path) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"pid={os.getpid()}\n")
        handle.flush()
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
