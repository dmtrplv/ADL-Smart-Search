"""Метаданные сборки индексов: дата и совместимость.

Даты у ``local`` и ``ckm`` (а также у архетипов и полей) могут различаться,
поэтому они хранятся в одном небольшом ``data/build_info.json`` с секциями:

    {
      "schema": 1,
      "local": {
        "archetype": {...},
        "field": {...}
      },
      "ckm": {...}
    }

Если файла нет (старые данные), дата берётся из mtime индексного файла —
этот fallback нужен только для совместимости.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime

from adl_smart_search.index_files import index_filenames

logger = logging.getLogger(__name__)

BUILD_INFO_FILENAME = "build_info.json"
SCHEMA_VERSION = 1


def build_info_path(index_root: str) -> str:
    return os.path.join(index_root or ".", BUILD_INFO_FILENAME)


def load_build_info(index_root: str) -> dict:
    path = build_info_path(index_root)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        logger.exception("build_info: cannot read %s", path)
        return {}


def kind_entry(index_root: str, source: str, kind: str) -> dict:
    data = load_build_info(index_root)
    entry = data.get(source) or {}
    if not isinstance(entry, dict):
        return {}
    kind_data = entry.get(kind) or {}
    return kind_data if isinstance(kind_data, dict) else {}


def _parse_built_at(value) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text).timestamp()
    except (TypeError, ValueError):
        logger.debug("build_info: bad built_at: %r", value)
        return None


def index_built_at(index_root: str, source: str, kind: str) -> float | None:
    """Время сборки индекса: из build_info.json, иначе mtime файла."""
    entry = kind_entry(index_root, source, kind)
    ts = _parse_built_at(entry.get("built_at"))
    if ts is not None:
        return ts

    docs_fn, _ = index_filenames(kind)
    path = os.path.join(index_root or ".", source, docs_fn)
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def format_index_date(ts: float | None) -> str:
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(ts).strftime("%d.%m.%Y")
    except Exception:
        logger.debug("build_info: bad timestamp: %r", ts)
        return ""


def format_index_datetime(ts: float | None) -> str:
    if ts is None:
        return ""
    try:
        return datetime.fromtimestamp(ts).strftime("%d.%m.%Y %H:%M")
    except Exception:
        logger.debug("build_info: bad timestamp: %r", ts)
        return ""


def index_built_label(index_root: str, source: str, kind: str) -> str:
    ts = index_built_at(index_root, source, kind)
    label = format_index_datetime(ts)
    return label or "ещё не индексировалось"


def save_kind_build_info(
    index_root: str,
    source: str,
    kind: str,
    *,
    parser_version: int,
    index_format_version: int,
    embed_model: str,
    embed_dim: int,
    built_at: str | None = None,
) -> None:
    """Записывает/обновляет дату сборки одного индекса, не трогая остальные."""
    path = build_info_path(index_root)
    data = load_build_info(index_root)
    data["schema"] = SCHEMA_VERSION
    source_data = data.get(source)
    if not isinstance(source_data, dict):
        source_data = {}
        data[source] = source_data
    source_data[kind] = {
        "built_at": built_at or datetime.now().astimezone().isoformat(timespec="seconds"),
        "parser_version": int(parser_version or 0),
        "index_format_version": int(index_format_version or 0),
        "embed_model": str(embed_model or ""),
        "embed_dim": int(embed_dim or 0),
    }
    try:
        os.makedirs(index_root or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        logger.exception("build_info: cannot write %s", path)
