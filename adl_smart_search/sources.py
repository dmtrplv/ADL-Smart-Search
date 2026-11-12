"""Мульти-источники архетипов: локальная база + CKM (и, в перспективе, другие зеркала).

Каждый источник хранит собственные индексы в своём index_dir. При загрузке
индексы всех включённых источников склеиваются в единый список документов,
а каждый документ помечается источником (doc['source']), чтобы UI мог
фильтровать и показывать бейдж источника.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from adl_smart_search.config import AppConfig
from adl_smart_search.index_files import (
    CKM_SOURCE,
    LOCAL_SOURCE,
    index_files_exist,
    source_index_dir,
)
from adl_smart_search.ui.indexing import load_archetype_index_safe, load_field_index_safe

logger = logging.getLogger(__name__)

__all__ = [
    "CKM_SOURCE",
    "LOCAL_SOURCE",
    "MergedSources",
    "SourceBundle",
    "SourceSpec",
    "discover_sources",
    "discover_sources_from_values",
    "load_all",
    "resolve_local_index_dir",
]


@dataclass(frozen=True)
class SourceSpec:
    """Описание одного источника архетипов."""

    name: str
    label: str
    adl_dir: str
    index_dir: str

    @property
    def enabled(self) -> bool:
        """Источник доступен поиску: есть исходная папка .adl либо готовый индекс."""
        if self.adl_dir and os.path.isdir(self.adl_dir):
            return True
        return index_files_exist(self.index_dir)


@dataclass
class SourceBundle:
    """Загруженные индексы одного источника (могут быть не построены → None)."""

    spec: SourceSpec
    docs: list | None
    vectors: np.ndarray | None
    field_docs: list | None
    field_vectors: np.ndarray | None

    @property
    def is_built(self) -> bool:
        return self.docs is not None and self.vectors is not None


@dataclass
class MergedSources:
    """Индексы всех источников + склеенные массивы для поиска."""

    specs: list[SourceSpec]
    bundles: list[SourceBundle]
    docs: list
    vectors: np.ndarray
    field_docs: list
    field_vectors: np.ndarray

    @property
    def source_labels(self) -> dict[str, str]:
        return {s.name: s.label for s in self.specs}


def resolve_local_index_dir(root: str) -> tuple[str, bool]:
    """Папка индекса локальной базы и флаг «старая раскладка».

    Новая раскладка: корень индексов содержит подпапки `local/` и `ckm/`.
    Если в `local/` нет индекса, но файлы индекса лежат прямо в корне (старая сборка) —
    читаем индекс оттуда (ничего не переносим и не удаляем).
    """
    root = root or "./data"
    local_dir = source_index_dir(root, LOCAL_SOURCE)
    if index_files_exist(local_dir):
        return local_dir, False
    if index_files_exist(root):
        logger.warning("Индекс локальной базы лежит прямо в корне %s (старая раскладка)", root)
        return root, True
    return local_dir, False


def discover_sources_from_values(
    adl_dir: str,
    index_dir: str,
    ckm_adl_dir: str = "",
) -> list[SourceSpec]:
    """Строит источники по значениям папок/индексов.

    index_dir — корень дерева индексов: индексы источников лежат в
    <index_dir>/local и <index_dir>/ckm.

    Папки .adl необязательны: источник добавляется, если есть сама папка ИЛИ
    в его папке индекса уже лежит готовый индекс (поиск по предсобранному индексу).
    """
    root = index_dir or "./data"
    local_idx, _legacy = resolve_local_index_dir(root)
    specs: list[SourceSpec] = [
        SourceSpec(LOCAL_SOURCE, "Локальная база", adl_dir or "", local_idx)
    ]
    ckm_dir = (ckm_adl_dir or "").strip()
    ckm_idx = source_index_dir(root, CKM_SOURCE)
    has_index = index_files_exist(ckm_idx)
    if (ckm_dir and os.path.isdir(ckm_dir)) or has_index:
        specs.append(SourceSpec(CKM_SOURCE, "CKM", ckm_dir, ckm_idx))
    return specs


def discover_sources(cfg: AppConfig) -> list[SourceSpec]:
    """Локальная база — всегда; CKM — только если есть каталог или готовый индекс."""
    return discover_sources_from_values(
        adl_dir=cfg.adl_dir or "",
        index_dir=cfg.index_dir or "./data",
        ckm_adl_dir=cfg.ckm_adl_dir or "",
    )


def load_all(cfg: AppConfig, sources: list[SourceSpec] | None = None) -> MergedSources:
    """Загружает индексы всех источников и склеивает их в единые массивы."""
    specs = sources if sources is not None else discover_sources(cfg)
    bundles: list[SourceBundle] = []

    for spec in specs:
        docs, vectors = load_archetype_index_safe(spec.index_dir)
        field_docs, field_vectors = load_field_index_safe(spec.index_dir)
        bundles.append(
            SourceBundle(
                spec=spec,
                docs=docs,
                vectors=vectors,
                field_docs=field_docs,
                field_vectors=field_vectors,
            )
        )

    docs_all: list[dict] = []
    field_docs_all: list[dict] = []
    vec_parts: list[np.ndarray] = []
    field_vec_parts: list[np.ndarray] = []

    for b in bundles:
        if not b.is_built or b.vectors is None:
            continue
        for d in b.docs or []:
            # Метку ставим принудительно: старые индексы могли быть собраны
            # с дефолтным source="local" (баг), иначе CKM-документы не отличить.
            d["source"] = b.spec.name
            docs_all.append(d)
        vec_parts.append(b.vectors)
        for d in b.field_docs or []:
            d["source"] = b.spec.name
            field_docs_all.append(d)
        if b.field_vectors is not None and b.field_vectors.size:
            field_vec_parts.append(b.field_vectors)

    vectors = (
        np.concatenate(vec_parts, axis=0).astype(np.float32)
        if vec_parts
        else np.array([], dtype=np.float32)
    )
    field_vectors = (
        np.concatenate(field_vec_parts, axis=0).astype(np.float32)
        if field_vec_parts
        else np.array([], dtype=np.float32)
    )

    return MergedSources(
        specs=specs,
        bundles=bundles,
        docs=docs_all,
        vectors=vectors,
        field_docs=field_docs_all,
        field_vectors=field_vectors,
    )