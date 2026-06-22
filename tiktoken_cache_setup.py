"""
Persistent tiktoken cache (avoids macOS temp-dir cleanup and repeat downloads).

Set TIKTOKEN_CACHE_DIR before any langchain / OpenAI import that uses tiktoken.
Optional override via environment or .env (TIKTOKEN_CACHE_DIR=...).
"""

from __future__ import annotations

import hashlib
import os
import shutil
from pathlib import Path
from typing import Iterable, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CACHE_DIR = _SCRIPT_DIR / "tiktoken_cache"

_CL100K_BLOB = (
    "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
)
_CL100K_SHA256 = (
    "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
)
_CL100K_CACHE_KEY = hashlib.sha1(_CL100K_BLOB.encode()).hexdigest()


def _cache_path(cache_dir: Path) -> Path:
    return cache_dir / _CL100K_CACHE_KEY


def _hash_ok(data: bytes) -> bool:
    return hashlib.sha256(data).hexdigest() == _CL100K_SHA256


def _candidate_seed_files() -> Iterable[Path]:
    home = Path.home()
    globs = [
        ".vscode/extensions/github.copilot-chat-*/dist/cl100k_base.tiktoken",
        ".vscode/extensions/github.copilot-*/dist/resources/cl100k_base.tiktoken.noindex",
    ]
    for pattern in globs:
        for path in home.glob(pattern):
            if path.is_file():
                yield path


def _try_seed_from_local(cache_path: Path) -> bool:
    for src in _candidate_seed_files():
        try:
            data = src.read_bytes()
        except OSError:
            continue
        if not _hash_ok(data):
            continue
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, cache_path)
        return True
    return False


def _try_download(cache_path: Path) -> bool:
    try:
        import requests
    except ImportError:
        return False
    try:
        resp = requests.get(_CL100K_BLOB, timeout=60)
        resp.raise_for_status()
        data = resp.content
    except Exception:
        return False
    if not _hash_ok(data):
        return False
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(data)
    return True


def ensure_cl100k_encoding_cached(cache_dir: Optional[Path] = None) -> bool:
    """Return True when cl100k_base is available offline in the cache directory."""
    root = cache_dir or Path(os.environ.get("TIKTOKEN_CACHE_DIR", _DEFAULT_CACHE_DIR))
    target = _cache_path(root)
    if target.is_file():
        try:
            if _hash_ok(target.read_bytes()):
                return True
        except OSError:
            pass

    if _try_seed_from_local(target):
        return True
    return _try_download(target)


def configure_tiktoken_cache(base_dir: Optional[str] = None) -> str:
    """
    Point tiktoken at a stable project cache directory.
    Does not raise if the encoding file is missing (tiktoken may still try online).
    """
    if not os.environ.get("TIKTOKEN_CACHE_DIR"):
        cache_dir = (
            Path(base_dir) / "tiktoken_cache"
            if base_dir
            else _DEFAULT_CACHE_DIR
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["TIKTOKEN_CACHE_DIR"] = str(cache_dir)
    cache_dir_str = os.environ["TIKTOKEN_CACHE_DIR"]
    ensure_cl100k_encoding_cached(Path(cache_dir_str))
    return cache_dir_str


configure_tiktoken_cache()
