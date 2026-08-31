"""Helpers for recording the locked re_arc dependency version."""
from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


RE_ARC_LOCK_PACKAGE = "arc-agi-3-local"


def _find_uv_lock() -> Path | None:
    seen: set[Path] = set()
    for anchor in (Path(__file__).resolve(), Path.cwd()):
        current = anchor if anchor.is_dir() else anchor.parent
        for parent in (current, *current.parents):
            path = parent / "uv.lock"
            if path in seen:
                continue
            seen.add(path)
            if path.exists():
                return path
    return None


def _commit_from_source(source: Any) -> str | None:
    if not isinstance(source, dict):
        return None
    raw_commit = source.get("commit")
    if raw_commit not in (None, ""):
        return str(raw_commit).strip() or None
    raw_git = source.get("git")
    if raw_git in (None, ""):
        return None
    return urlparse(str(raw_git)).fragment.strip() or None


def read_locked_re_arc_commit(lock_path: str | Path | None = None) -> str | None:
    """Return the Git commit pinned for re_arc in uv.lock, if available."""
    resolved_path = Path(lock_path) if lock_path is not None else _find_uv_lock()
    if resolved_path is None or not resolved_path.exists():
        return None
    try:
        payload = tomllib.loads(resolved_path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return None

    packages = payload.get("package")
    if not isinstance(packages, list):
        return None
    for package in packages:
        if not isinstance(package, dict) or package.get("name") != RE_ARC_LOCK_PACKAGE:
            continue
        return _commit_from_source(package.get("source"))
    return None
