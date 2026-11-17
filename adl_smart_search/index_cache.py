"""Общий кэш загруженных индексов для всех сессий Streamlit.

Индексы и нормализованные векторы не зависят от пользовательской сессии.
Раньше они лежали в ``st.session_state`` и копировались при каждом открытии
вкладки/переподключении. Здесь они разделяются между сессиями, а старый набор
вытесняется при пересборке индекса (``max_entries=1``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import streamlit as st

from .index_files import index_filenames
from .rag import build_token_vocab
from .sources import SourceSpec, load_all

logger = logging.getLogger(__name__)


@dataclass
class IndexResources:
    docs: list[dict]
    vectors: np.ndarray
    field_docs: list[dict]
    field_vectors: np.ndarray
    arch_token_vocab: set[str]
    field_token_vocab: set[str]
    bundles: list


def _normalize_rows(mat):
    if mat is None:
        return mat
    try:
        if not hasattr(mat, "shape") or mat.size == 0:
            return mat
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype(np.float32)
    except Exception:
        logger.debug("normalize_rows: skipping normalization")
        return mat


def _stat_key(path: str) -> tuple[int, int]:
    try:
        stt = os.stat(path)
        return int(stt.st_mtime_ns), int(stt.st_size)
    except OSError:
        return 0, 0


def index_signature(sources: list[SourceSpec]) -> tuple:
    """Ключ кэша: меняется при пересборке любого индекса."""
    parts: list[tuple] = []
    for spec in sources:
        for kind in ("archetype", "field"):
            for filename in index_filenames(kind):
                path = os.path.join(spec.index_dir, filename)
                parts.append((spec.name, filename, *_stat_key(path)))
        build_info = os.path.join(
            os.path.dirname(os.path.normpath(spec.index_dir)), "build_info.json"
        )
        parts.append((spec.name, "build_info.json", *_stat_key(build_info)))
    return tuple(parts)


def get_index_resources(sources: list[SourceSpec]) -> IndexResources:
    """Возвращает общий (кэшированный) набор индексов и векторов."""
    specs = list(sources or [])
    signature = index_signature(specs)
    sources_data = tuple(
        (s.name, s.label, s.adl_dir, s.index_dir) for s in specs
    )
    return _load_merged_cached(signature, sources_data)


def clear_index_cache() -> None:
    """Сбросить кэш после пересборки индекса."""
    _load_merged_cached.clear()


@st.cache_resource(max_entries=1, show_spinner="Загрузка индексов...")
def _load_merged_cached(
    signature: tuple, sources_data: tuple
) -> IndexResources:
    # signature — часть ключа кэша, в теле не используется.
    _ = signature
    specs = [SourceSpec(*item) for item in sources_data]
    merged = load_all(None, specs)
    docs = merged.docs
    field_docs = merged.field_docs
    return IndexResources(
        docs=docs,
        vectors=_normalize_rows(merged.vectors),
        field_docs=field_docs,
        field_vectors=_normalize_rows(merged.field_vectors),
        arch_token_vocab=build_token_vocab(docs, field="anchor"),
        field_token_vocab=build_token_vocab(field_docs, field="anchor"),
        bundles=merged.bundles,
    )
