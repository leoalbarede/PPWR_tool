"""
Persistent FlashRank reranker model cache (default flashrank uses /tmp).

The LangChain FlashrankRerank wrapper does not expose cache_dir; use
get_flashrank_ranker() from analyzer_docling instead of bare FlashrankRerank().
"""

from __future__ import annotations

import os
import zipfile
from pathlib import Path
from typing import Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
_DEFAULT_CACHE_DIR = _SCRIPT_DIR / "flashrank_cache"

DEFAULT_FLASHRANK_MODEL = "ms-marco-MultiBERT-L-12"
_MODEL_ONNX = "flashrank-MultiBERT-L12_Q.onnx"
_MODEL_URL = (
    "https://huggingface.co/prithivida/flashrank/resolve/main/"
    f"{DEFAULT_FLASHRANK_MODEL}.zip"
)


def flashrank_cache_dir(base_dir: Optional[str] = None) -> Path:
    env = os.environ.get("FLASHRANK_CACHE_DIR")
    if env:
        return Path(env)
    if base_dir:
        return Path(base_dir) / "flashrank_cache"
    return _DEFAULT_CACHE_DIR


def _model_ready(cache_dir: Path, model_name: str = DEFAULT_FLASHRANK_MODEL) -> bool:
    onnx = cache_dir / model_name / _MODEL_ONNX
    return onnx.is_file()


def ensure_flashrank_model_cached(
    cache_dir: Optional[Path] = None,
    model_name: str = DEFAULT_FLASHRANK_MODEL,
) -> bool:
    """Return True when the ONNX reranker is present locally."""
    root = cache_dir or flashrank_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    if _model_ready(root, model_name):
        return True

    zip_path = root / f"{model_name}.zip"
    try:
        import requests
    except ImportError:
        return False

    try:
        with requests.get(_MODEL_URL, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(zip_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(root)
        zip_path.unlink(missing_ok=True)
        return _model_ready(root, model_name)
    except Exception:
        zip_path.unlink(missing_ok=True)
        return False


def get_flashrank_ranker(model_name: str = DEFAULT_FLASHRANK_MODEL):
    """Ranker wired to the project cache directory (no /tmp dependency)."""
    from flashrank import Ranker

    cache = flashrank_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    ensure_flashrank_model_cached(cache, model_name)
    return Ranker(model_name=model_name, cache_dir=str(cache))


ensure_flashrank_model_cached()
