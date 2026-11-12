"""Каталог корпоративных моделей.

Каталог читается из локального JSON-файла (путь из ENV ADL_CORP_CATALOG,
по умолчанию <repo-root>/config/corp_catalog.json; для совместимости проверяются
data/corp_catalog.json и старый ./corp_catalog.json). Структура файла — JSON-список записей:
    [{"model_name": "...", "provider": "...", "model": "...", "model_kwargs": {...}}, ...]
"""

import json
import logging
import os

logger = logging.getLogger(__name__)


def load_catalog_json(path: str | None) -> list[dict] | None:
    """Читает каталог из JSON-файла. Returns None, если путь пустой/не читается/структура неверная."""
    raw_path = (path or "").strip()
    if not raw_path:
        return None
    try:
        with open(raw_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        logger.exception("load_catalog_json: cannot read JSON catalog: %s", raw_path)
        return None
    if not isinstance(data, list):
        return None
    rows = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if not str(item.get("model_name") or "").strip():
            continue
        rows.append(item)
    return rows


def resolve_corp_catalog(path: str | None) -> list[dict]:
    """Возвращает каталог из JSON-файла; при пустом пути/нечитаемом файле — пустой список."""
    loaded = load_catalog_json(path)
    if loaded is not None:
        return loaded
    if (path or "").strip():
        logger.warning("corp_catalog: cannot load catalog from '%s', using empty list", path)
    return []


def resolve_corp_catalog_from_paths(*paths: str) -> list[dict]:
    """Первый успешно прочитанный каталог из путей в порядке приоритета.

    Отсутствующие и нечитаемые файлы пропускаются без traceback: это обычная
    ситуация при поиске каталога по нескольким местам (например, новый
    ``data/corp_catalog.json`` и старый ``./corp_catalog.json``).
    """
    tried: list[str] = []
    seen: set[str] = set()
    for path in paths:
        raw = (path or "").strip()
        if not raw:
            continue
        key = os.path.normcase(os.path.abspath(raw))
        if key in seen:
            continue
        seen.add(key)
        tried.append(raw)
        if not os.path.isfile(raw):
            continue
        loaded = load_catalog_json(raw)
        if loaded:
            return loaded
    if tried:
        logger.warning("corp_catalog: no catalog found in %s", ", ".join(tried))
    return []


def corp_catalog_by_name(catalog: list[dict]) -> dict[str, dict]:
    return {str(x.get("model_name")): x for x in (catalog or []) if x.get("model_name")}


def catalog_providers(catalog: list[dict]) -> list[str]:
    """Уникальные провайдеры из каталога (в порядке появления)."""
    seen: list[str] = []
    for x in catalog or []:
        p = str(x.get("provider") or "").strip()
        if p and p not in seen:
            seen.append(p)
    return seen