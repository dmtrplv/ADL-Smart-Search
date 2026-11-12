"""Манифест релиза и дельта-архивы для обновления приложения.

Манифест описывает только программную часть установки (exe, ``_internal``,
``models``, ``sertificates`` и т.п.). Пользовательские ``data/``, ``config/``,
``.env`` и логи в него не попадают.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import zipfile
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
DEFAULT_MANAGED_ROOTS = (
    "ADL-Smart-Search.exe",
    "ADL-Smart-Search-Index.exe",
    "ADL-Smart-Search-Updater.exe",
    "version.json",
    "_internal",
    "models",
    "sertificates",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _posix(path: str) -> str:
    return path.replace("\\", "/")


def _safe_rel(path: str) -> bool:
    rel = _posix(path).strip()
    if not rel or rel.startswith("/") or re.match(r"^[A-Za-z]:", rel):
        return False
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    return bool(parts) and all(p != ".." for p in parts)


def _iter_root_files(root: str, rel_root: str):
    path = os.path.join(root, rel_root)
    if os.path.isfile(path):
        yield _posix(rel_root), path
        return
    if not os.path.isdir(path):
        return
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in (".git", "__pycache__", ".pytest_cache"))
        for name in sorted(filenames):
            if name.endswith((".pyc", ".pyo")):
                continue
            full = os.path.join(dirpath, name)
            rel = _posix(os.path.relpath(full, root))
            yield rel, full


def collect_program_files(root: str, managed_roots: tuple[str, ...] | list[str] = DEFAULT_MANAGED_ROOTS):
    """Возвращает список (relative_path, size, sha256) для программных файлов."""
    root = os.path.abspath(root)
    seen: set[str] = set()
    out: list[tuple[str, int, str]] = []
    for rel_root in managed_roots:
        for rel, full in _iter_root_files(root, str(rel_root)):
            if rel in seen or not _safe_rel(rel):
                continue
            seen.add(rel)
            try:
                size = os.path.getsize(full)
            except OSError:
                continue
            out.append((rel, size, file_sha256(full)))
    out.sort(key=lambda x: x[0])
    return out


def build_manifest(
    root: str,
    *,
    version: str,
    channel: str = "stable",
    notes: str = "",
    assets: dict | None = None,
    managed_roots: tuple[str, ...] | list[str] = DEFAULT_MANAGED_ROOTS,
    published_at: str | None = None,
) -> dict:
    files = [
        {"path": rel, "size": size, "sha256": sha}
        for rel, size, sha in collect_program_files(root, managed_roots)
    ]
    present_roots = [r for r in managed_roots if os.path.exists(os.path.join(root, r))]
    return {
        "schema": SCHEMA_VERSION,
        "version": str(version),
        "channel": str(channel or "stable"),
        "published_at": published_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "notes": str(notes or ""),
        "managed_roots": [_posix(r) for r in present_roots],
        "files": files,
        "assets": assets or {},
    }


def manifest_files(manifest: dict) -> dict[str, dict]:
    return {str(item.get("path")): item for item in (manifest.get("files") or []) if item.get("path")}


def validate_manifest(manifest: dict) -> None:
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if int(manifest.get("schema") or 0) != SCHEMA_VERSION:
        raise ValueError(f"unsupported manifest schema: {manifest.get('schema')!r}")
    if not str(manifest.get("version") or "").strip():
        raise ValueError("manifest.version is empty")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest.files must be a non-empty list")
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("manifest.files entries must be objects")
        path = str(item.get("path") or "")
        if not _safe_rel(path):
            raise ValueError(f"unsafe manifest path: {path!r}")
        if not isinstance(item.get("size"), int) or int(item["size"]) < 0:
            raise ValueError(f"bad size for {path}")
        if not _SHA256_RE.match(str(item.get("sha256") or "").lower()):
            raise ValueError(f"bad sha256 for {path}")
    for root in manifest.get("managed_roots") or []:
        if not _safe_rel(str(root)):
            raise ValueError(f"unsafe managed root: {root!r}")


def load_manifest(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    validate_manifest(data)
    return data


def save_manifest(path: str, manifest: dict) -> None:
    validate_manifest(manifest)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def compare_manifests(old: dict, new: dict) -> tuple[set[str], set[str]]:
    """Возвращает (изменённые/новые файлы, удалённые файлы)."""
    old_map = manifest_files(old)
    new_map = manifest_files(new)
    changed = {
        path
        for path, item in new_map.items()
        if path not in old_map or old_map[path].get("sha256") != item.get("sha256")
    }
    deleted = {path for path in old_map if path not in new_map}
    return changed, deleted


def create_delta_zip(
    new_root: str,
    old_manifest: dict,
    new_manifest: dict,
    out_path: str,
    *,
    compresslevel: int = 1,
) -> tuple[int, int]:
    """Создаёт zip только с изменёнными/новыми файлами.

    Возвращает (число файлов, размер zip в байтах).
    """
    changed, _deleted = compare_manifests(old_manifest, new_manifest)
    root = os.path.abspath(new_root)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tmp = f"{out_path}.tmp"
    count = 0
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel) as zf:
        for rel in sorted(changed):
            full = os.path.join(root, rel.replace("/", os.sep))
            if not os.path.isfile(full):
                raise FileNotFoundError(f"delta source file is missing: {rel}")
            zf.write(full, arcname=rel)
            count += 1
    os.replace(tmp, out_path)
    return count, os.path.getsize(out_path)


def version_parts(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for chunk in re.split(r"[.+-]", str(version or "")):
        if chunk.isdigit():
            parts.append(int(chunk))
        else:
            break
    return tuple(parts) if parts else (0,)


def is_newer(candidate: str, current: str) -> bool:
    a = version_parts(candidate)
    b = version_parts(current)
    length = max(len(a), len(b))
    return a + (0,) * (length - len(a)) > b + (0,) * (length - len(b))
