"""Сборка/загрузка индексов в UI (с прогресс-барами)."""

import logging
import os

import numpy as np
import streamlit as st

from adl_smart_search.index_files import ALL_INDEX_FILE_NAMES, LOCAL_SOURCE, index_filenames, source_index_dir
from adl_smart_search.index_store import (
    build_or_load_field_index_incremental,
    build_or_load_index_incremental,
    load_archetype_index,
    load_field_index,
)
from adl_smart_search.ui.llm_provider import get_embedder

logger = logging.getLogger(__name__)


def maybe_move_legacy_indexes_to_data(index_dir: str):
    """
    Миграция: если старые индексы лежали в корне проекта, переносим их в
    папку локальной базы (<index_dir>/local).
    """
    target_dir = source_index_dir(index_dir or "./data", LOCAL_SOURCE)
    os.makedirs(target_dir, exist_ok=True)
    legacy_files = list(ALL_INDEX_FILE_NAMES)
    for fn in legacy_files:
        src = os.path.join(".", fn)
        dst = os.path.join(target_dir, fn)
        try:
            if os.path.exists(src) and not os.path.exists(dst):
                os.replace(src, dst)
        except Exception:
            # не критично
            logger.exception("maybe_move_legacy_indexes_to_data: cannot move %s -> %s", src, dst)


def build_archetype_index_with_progress(
    adl_dir: str,
    preferred_lang: str,
    index_dir: str,
    embed_model_name: str,
    source: str = LOCAL_SOURCE,
):
    embedder = get_embedder(embed_model_name)
    bar = st.progress(0, text="Подготовка...")
    status = st.empty()

    def cb(done, total, phase):
        p = 0.0 if total == 0 else (done / total)
        value = min(100, max(0, int(round(p * 100))))
        bar.progress(value, text=f"{phase}: {done}/{total}")
        status.caption(f"{phase}: {done}/{total}")

    docs, vectors = build_or_load_index_incremental(
        adl_dir=adl_dir,
        embedder=embedder,
        preferred_lang=preferred_lang,
        index_dir=index_dir,
        source=source,
        progress_cb=cb,
    )
    bar.progress(100, text="Готово")
    return docs, vectors


def build_field_index_with_progress(
    adl_dir: str,
    preferred_lang: str,
    index_dir: str,
    embed_model_name: str,
    source: str = LOCAL_SOURCE,
):
    embedder = get_embedder(embed_model_name)
    bar = st.progress(0, text="Подготовка...")
    status = st.empty()

    def cb(done, total, phase):
        p = 0.0 if total == 0 else (done / total)
        value = min(100, max(0, int(round(p * 100))))
        bar.progress(value, text=f"{phase}: {done}/{total}")
        status.caption(f"{phase}: {done}/{total}")

    field_docs, field_vectors = build_or_load_field_index_incremental(
        adl_dir=adl_dir,
        embedder=embedder,
        preferred_lang=preferred_lang,
        index_dir=index_dir,
        source=source,
        progress_cb=cb,
    )
    bar.progress(100, text="Готово")
    return field_docs, field_vectors


def delete_archetype_index_files(index_dir: str):
    for fn in index_filenames("archetype"):
        try:
            os.remove(os.path.join(index_dir, fn))
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception("delete_archetype_index_files: cannot remove %s", os.path.join(index_dir, fn))


def delete_field_index_files(index_dir: str):
    for fn in index_filenames("field"):
        try:
            os.remove(os.path.join(index_dir, fn))
        except FileNotFoundError:
            pass
        except Exception:
            logger.exception("delete_field_index_files: cannot remove %s", os.path.join(index_dir, fn))


def load_archetype_index_safe(index_dir: str):
    try:
        return load_archetype_index(index_dir=index_dir)
    except FileNotFoundError:
        # Индекс ещё не построен — это штатная ситуация.
        logger.debug("load_archetype_index_safe: index not built yet in %s", index_dir)
    except Exception:
        logger.exception("load_archetype_index_safe: load failed from %s", index_dir)
    return None, None


def load_field_index_safe(index_dir: str):
    try:
        return load_field_index(index_dir=index_dir)
    except FileNotFoundError:
        # Индекс ещё не построен — это штатная ситуация.
        logger.debug("load_field_index_safe: index not built yet in %s", index_dir)
    except Exception:
        logger.exception("load_field_index_safe: load failed from %s", index_dir)
    return None, None


def norm_rows(mat):
    if mat is None:
        return None
    try:
        if not hasattr(mat, "shape") or mat.size == 0:
            return mat
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (mat / norms).astype(np.float32)
    except Exception:
        logger.debug("norm_rows: skipping normalization")
        return mat