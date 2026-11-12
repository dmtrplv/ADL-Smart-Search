import hashlib
import json
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .adl_parser import (
    compact_adl,
    create_search_anchor,
    detect_content_lang,
    extract_details_metadata,
    extract_field_chunks_from_compact,
    has_language,
)
from .embedder import _normalize_st_model_name
from .index_files import CKM_SOURCE, index_filenames

INDEX_FORMAT_VERSION = 1
# Версия разбора .adl. Увеличивайте, когда меняется compact_adl/anchor/field-разбор:
# старые документы тогда переразбираются, а не переиспользуются инкрементально.
PARSER_VERSION = 2

# Опциональные ключи меты в docs_json (v1+). Их отсутствие = legacy индекс без проверки модели.
META_EMBED_MODEL = "embed_model"
META_EMBED_DIM = "embed_dim"


def _embedder_meta(embedder) -> Tuple[str, int]:
    """Имя модели и размерность векторов (0, если не определяется)."""
    name = str(getattr(embedder, "model_name", "") or "")
    dim = 0
    dim_fn = getattr(embedder, "embedding_dim", None)
    if callable(dim_fn):
        try:
            dim = int(dim_fn())
        except Exception:
            dim = 0
    return name, dim


def _read_index_meta_raw(docs_json: str) -> dict:
    """Читает top-level мету из docs_json. Возвращает {} для legacy/list формата."""
    try:
        with open(docs_json, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def index_meta(index_dir: str, kind: str = "archetype") -> dict:
    """Метаданные индекса: {'embed_model': str, 'embed_dim': int}. Пусто при legacy/отсутствии файла."""
    fn, _ = index_filenames(kind)
    raw = _read_index_meta_raw(os.path.join(index_dir, fn))
    try:
        dim = int(raw.get(META_EMBED_DIM) or 0)
    except (TypeError, ValueError):
        dim = 0
    return {
        META_EMBED_MODEL: str(raw.get(META_EMBED_MODEL) or ""),
        META_EMBED_DIM: dim,
    }


def ensure_embedding_dim_compatible(index_dir: str, embed_dim: int, kind: str = "archetype") -> None:
    """
    Проверяет, что индекс (если несёт embed_dim) совместим с текущим эмбеддером.
    Возбуждает ValueError с понятным сообщением при несовпадении. Legacy (без меты) — пропускает.
    """
    if not embed_dim:
        return
    meta = index_meta(index_dir, kind)
    stored_dim = meta.get(META_EMBED_DIM) or 0
    if stored_dim and int(stored_dim) != int(embed_dim):
        raise ValueError(
            f"Индекс в '{index_dir}' построен другой моделью эмбеддингов: "
            f"embed_dim={stored_dim}"
            + (f" (модель: {meta.get(META_EMBED_MODEL)})" if meta.get(META_EMBED_MODEL) else "")
            + f", а текущая модель даёт {embed_dim}. Пересоберите индекс той же моделью."
        )


def index_model_matches(index_dir: str, model_name: str, kind: str = "archetype") -> bool | None:
    """
    Быстрая проверка совместимости индекса с моделью эмбеддингов по мете — БЕЗ загрузки модели.

    Возвращает:
    - True  — индекс несёт embed_model и оно совпадает с текущей моделью;
    - False — embed_model в мете есть, но отличается (нужна ленивая проверка dim);
    - None  — меты нет (legacy) или файл отсутствует: проверить по имени нельзя.
    """
    if not model_name:
        return None
    meta = index_meta(index_dir, kind)
    stored_model = str(meta.get(META_EMBED_MODEL) or "").strip()
    if not stored_model:
        return None
    return stored_model == _normalize_st_model_name(model_name)


@dataclass(frozen=True)
class IndexPaths:
    dir_path: str

    @property
    def docs_json(self) -> str:
        fn, _ = index_filenames("archetype")
        return os.path.join(self.dir_path, fn)

    @property
    def vectors_npy(self) -> str:
        _, fn = index_filenames("archetype")
        return os.path.join(self.dir_path, fn)

    @property
    def field_docs_json(self) -> str:
        fn, _ = index_filenames("field")
        return os.path.join(self.dir_path, fn)

    @property
    def field_vectors_npy(self) -> str:
        _, fn = index_filenames("field")
        return os.path.join(self.dir_path, fn)


def _sha1_bytes(data: bytes) -> str:
    h = hashlib.sha1()
    h.update(data)
    return h.hexdigest()


def _norm_rel_path(rel_path: str) -> str:
    # В индексе храним относительный путь с "/" для стабильности между ОС.
    return rel_path.replace("\\", "/").lstrip("./")


def _iter_adl_files_recursive(adl_dir: str) -> List[str]:
    """
    Возвращает список относительных путей .adl файлов внутри adl_dir (рекурсивно).
    Важно: относительные пути нормализованы и стабильны для инкрементального индекса.
    """
    out: List[str] = []
    for root, _, files in os.walk(adl_dir):
        for fn in files:
            if not fn.lower().endswith(".adl"):
                continue
            full_path = os.path.join(root, fn)
            rel = os.path.relpath(full_path, adl_dir)
            out.append(_norm_rel_path(rel))
    return sorted(out)


def _build_old_doc_lookup(old_docs: List[dict]) -> Tuple[Dict[str, dict], Dict[str, List[dict]]]:
    """
    Поддерживает обратную совместимость:
    - новый формат: d["file"] содержит относительный путь (возможно с "/")
    - старый формат: d["file"] содержит только basename
    Возвращает:
    - map по точному ключу file
    - map по basename (для fallback, когда basename уникален)
    """
    by_file: Dict[str, dict] = {}
    by_base: Dict[str, List[dict]] = {}
    for d in old_docs:
        f = d.get("file")
        if not f:
            continue
        f2 = _norm_rel_path(str(f))
        by_file[f2] = d
        base = os.path.basename(f2)
        by_base.setdefault(base, []).append(d)
    return by_file, by_base


def _resolve_old_doc(rel_path: str, old_by_file: Dict[str, dict], old_by_base: Dict[str, List[dict]]) -> Optional[dict]:
    rel = _norm_rel_path(rel_path)
    direct = old_by_file.get(rel)
    if direct:
        return direct
    # Fallback: старый индекс мог хранить только basename; используем только если basename уникален.
    base = os.path.basename(rel)
    cands = old_by_base.get(base) or []
    return cands[0] if len(cands) == 1 else None


def _load_existing(paths: IndexPaths) -> Tuple[List[dict], np.ndarray]:
    if not os.path.exists(paths.docs_json) or not os.path.exists(paths.vectors_npy):
        return [], np.array([], dtype=np.float32)
    with open(paths.docs_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    docs: List[dict]
    if isinstance(raw, list):
        # v0: plain list of docs
        docs = raw
    elif isinstance(raw, dict):
        # v1+: {"index_version": 1, "docs": [...]}
        ver = raw.get("index_version", INDEX_FORMAT_VERSION)
        if isinstance(ver, int) and ver > INDEX_FORMAT_VERSION:
            raise ValueError(f"Неподдерживаемая версия индекса архетипов: {ver}")
        docs_raw = raw.get("docs")
        docs = docs_raw if isinstance(docs_raw, list) else []
    else:
        docs = []
    vectors = np.load(paths.vectors_npy)
    # Нормализуем на всякий случай (cosine via dot product).
    if vectors.size:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = (vectors / norms).astype(np.float32)
    return docs, vectors


def _load_existing_fields(paths: IndexPaths) -> Tuple[List[dict], np.ndarray]:
    if not os.path.exists(paths.field_docs_json) or not os.path.exists(paths.field_vectors_npy):
        return [], np.array([], dtype=np.float32)
    with open(paths.field_docs_json, "r", encoding="utf-8") as f:
        raw = json.load(f)
    docs: List[dict]
    if isinstance(raw, list):
        docs = raw
    elif isinstance(raw, dict):
        ver = raw.get("index_version", INDEX_FORMAT_VERSION)
        if isinstance(ver, int) and ver > INDEX_FORMAT_VERSION:
            raise ValueError(f"Неподдерживаемая версия индекса полей: {ver}")
        docs_raw = raw.get("docs")
        docs = docs_raw if isinstance(docs_raw, list) else []
    else:
        docs = []
    vectors = np.load(paths.field_vectors_npy)
    if vectors.size:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors = (vectors / norms).astype(np.float32)
    return docs, vectors


def load_archetype_index(index_dir: str = ".") -> Tuple[List[dict], np.ndarray]:
    """Загружает индекс архетипов с диска (без построения)."""
    paths = IndexPaths(dir_path=index_dir)
    docs, vectors = _load_existing(paths)
    if len(docs) == 0 or vectors.size == 0:
        raise FileNotFoundError("Индекс архетипов не найден (archetype_data.json/archetype_vectors.npy)")
    return docs, vectors


def load_field_index(index_dir: str = ".") -> Tuple[List[dict], np.ndarray]:
    """Загружает индекс полей с диска (без построения)."""
    paths = IndexPaths(dir_path=index_dir)
    docs, vectors = _load_existing_fields(paths)
    if len(docs) == 0 or vectors.size == 0:
        raise FileNotFoundError("Индекс полей не найден (field_data.json/field_vectors.npy)")
    return docs, vectors


def _effective_lang(source: str, preferred_lang: str) -> str:
    """CKM индексируем только по английской версии; local — по preferred_lang."""
    return "en" if source == CKM_SOURCE else preferred_lang


def build_or_load_index_incremental(
    adl_dir: str,
    embedder,
    preferred_lang: str = "ru",
    index_dir: str = ".",
    source: str = "local",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    embed_batch_size: int = 256,
) -> Tuple[List[dict], np.ndarray]:
    """
    Строит индекс по .adl файлам. Если индекс уже есть — переиспользует записи/вектора
    для неизменившихся файлов (по mtime/size/sha1).
    """
    lang = _effective_lang(source, preferred_lang)
    is_ckm = source == CKM_SOURCE
    paths = IndexPaths(dir_path=index_dir)
    old_docs, old_vectors = _load_existing(paths)

    old_by_file, old_by_base = _build_old_doc_lookup(old_docs)
    old_vec_by_file: Dict[str, np.ndarray] = {}
    if len(old_docs) == len(old_vectors) and len(old_docs) > 0:
        for d, v in zip(old_docs, old_vectors):
            fn = d.get("file")
            if fn:
                old_vec_by_file[_norm_rel_path(str(fn))] = v

    files = _iter_adl_files_recursive(adl_dir)
    new_docs: List[dict] = []
    new_vectors: List[np.ndarray] = []

    changed_anchors: List[str] = []
    changed_indices: List[int] = []

    for n_file, rel_path in enumerate(files, start=1):
        full_path = os.path.join(adl_dir, rel_path)
        st = os.stat(full_path)
        if progress_cb and (n_file % 10 == 0 or n_file == len(files)):
            progress_cb(n_file, len(files), "Сканирование архетипов")

        # Чтобы sha1 не считать лишний раз, сначала проверяем быстрые признаки.
        old = _resolve_old_doc(rel_path, old_by_file, old_by_base)
        fast_unchanged = False
        if old:
            if old.get("mtime") == st.st_mtime and old.get("size") == st.st_size and old.get("sha1"):
                fast_unchanged = True
            if old.get("parser_version") != PARSER_VERSION:
                fast_unchanged = False
            if is_ckm and old.get("lang") != "en":
                fast_unchanged = False

        norm_rel = _norm_rel_path(rel_path)
        if fast_unchanged and norm_rel in old_vec_by_file:
            old = dict(old)
            old["source"] = source
            new_docs.append(old)
            new_vectors.append(old_vec_by_file[norm_rel])
            continue

        data = open(full_path, "rb").read()
        sha1 = _sha1_bytes(data)
        if (
            old
            and old.get("sha1") == sha1
            and norm_rel in old_vec_by_file
            and old.get("parser_version") == PARSER_VERSION
            and (not is_ckm or old.get("lang") == "en")
        ):
            # Файл мог иметь другой mtime/size (копирование), но контент тот же.
            reused = dict(old)
            reused["mtime"] = st.st_mtime
            reused["size"] = st.st_size
            reused["source"] = source
            new_docs.append(reused)
            new_vectors.append(old_vec_by_file[norm_rel])
            continue

        content = data.decode("utf-8-sig", errors="replace")
        if is_ckm and not has_language(content, "en"):
            continue
        full_ctx = compact_adl(content, preferred_lang=lang)
        anchor = create_search_anchor(content, preferred_lang=lang)
        meta = extract_details_metadata(content, preferred_lang=lang)
        actual_lang = detect_content_lang(content, preferred_lang=lang)
        # Усиливаем якорь метаданными для поиска (purpose/use/keywords).
        # misuse хранится отдельно и не входит в anchor — только для LLM-контекста.
        meta_bits = " ".join(
            [meta.get("purpose", ""), meta.get("use", ""), meta.get("keywords", "")]
        ).strip()
        if meta_bits:
            anchor = f"{anchor} | {meta_bits}"

        doc = {
            "file": norm_rel,
            "full_context": full_ctx,
            "anchor": (anchor or "").lower(),
            "misuse": meta.get("misuse", "") or "",
            "mtime": st.st_mtime,
            "size": st.st_size,
            "sha1": sha1,
            "lang": actual_lang,
            "source": source,
            "parser_version": PARSER_VERSION,
        }
        new_docs.append(doc)
        new_vectors.append(None)  # заполним после пакетной векторизации
        changed_indices.append(len(new_docs) - 1)
        changed_anchors.append(anchor)

    if changed_anchors:
        # После изменения состава anchor (например, исключение misuse) нужна переиндексация архетипов.
        # Векторизуем пакетами, чтобы можно было показывать прогресс.
        total = len(changed_anchors)
        out_vecs = []
        for start in range(0, total, max(1, int(embed_batch_size))):
            end = min(total, start + max(1, int(embed_batch_size)))
            batch = changed_anchors[start:end]
            vecs = embedder.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            out_vecs.extend(list(vecs))
            if progress_cb:
                progress_cb(end, total, "Векторизация архетипов")
        for idx, v in zip(changed_indices, out_vecs):
            new_vectors[idx] = v

    # Гарантируем, что векторный массив плотный
    vectors_arr = np.vstack([v for v in new_vectors]).astype(np.float32)
    norms = np.linalg.norm(vectors_arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors_arr = (vectors_arr / norms).astype(np.float32)

    os.makedirs(paths.dir_path, exist_ok=True)
    embed_name, embed_dim = _embedder_meta(embedder)
    with open(paths.docs_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "index_version": INDEX_FORMAT_VERSION,
                META_EMBED_MODEL: embed_name,
                META_EMBED_DIM: embed_dim,
                "docs": new_docs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    np.save(paths.vectors_npy, vectors_arr)
    return new_docs, vectors_arr


def build_or_load_field_index_incremental(
    adl_dir: str,
    embedder,
    preferred_lang: str = "ru",
    index_dir: str = ".",
    source: str = "local",
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    embed_batch_size: int = 256,
) -> Tuple[List[dict], np.ndarray]:
    """
    Строит индекс по полям (field-level). Каждое поле архетипа — отдельный "документ".
    Инкрементально: если файл .adl не менялся (по sha1) — переиспользуем все его поля и их вектора.
    """
    lang = _effective_lang(source, preferred_lang)
    is_ckm = source == CKM_SOURCE
    paths = IndexPaths(dir_path=index_dir)
    old_docs, old_vectors = _load_existing_fields(paths)

    # группируем старые поля по файлу (и проверяем sha1)
    old_fields_by_file: Dict[str, List[Tuple[dict, np.ndarray]]] = {}
    if len(old_docs) == len(old_vectors) and len(old_docs) > 0:
        for d, v in zip(old_docs, old_vectors):
            fn = d.get("file")
            if not fn:
                continue
            old_fields_by_file.setdefault(_norm_rel_path(str(fn)), []).append((d, v))

    files = _iter_adl_files_recursive(adl_dir)
    new_docs: List[dict] = []
    new_vectors: List[np.ndarray] = []

    # пакет векторизации для изменившихся файлов
    to_embed_texts: List[str] = []
    to_embed_indices: List[int] = []

    # Для fallback старого индекса (basename) — используем только если уникально.
    old_fields_by_base: Dict[str, List[Tuple[dict, np.ndarray]]] = {}
    for k, items in old_fields_by_file.items():
        base = os.path.basename(k)
        old_fields_by_base.setdefault(base, []).extend(items)

    for n_file, rel_path in enumerate(files, start=1):
        full_path = os.path.join(adl_dir, rel_path)
        st = os.stat(full_path)
        data = open(full_path, "rb").read()
        sha1 = _sha1_bytes(data)
        if progress_cb and (n_file % 10 == 0 or n_file == len(files)):
            progress_cb(n_file, len(files), "Сканирование полей")

        norm_rel = _norm_rel_path(rel_path)
        reused = False
        old_items = old_fields_by_file.get(norm_rel)
        if not old_items:
            base = os.path.basename(norm_rel)
            cands = old_fields_by_base.get(base) or []
            # Не называем это `files`: внешний `files` нужен для прогресса.
            cand_files = {_norm_rel_path(str(d.get("file") or "")) for d, _ in cands}
            old_items = cands if len(cand_files) == 1 and cands else None

        if old_items and (old_items[0][0] or {}).get("parser_version") != PARSER_VERSION:
            # Разбор .adl изменился — старые поля пересобираем.
            old_items = None
        if old_items and is_ckm and (old_items[0][0] or {}).get("lang") != "en":
            # Не переиспользуем старые русские/иные переводы CKM.
            old_items = None

        if old_items:
            # если sha1 совпадает — переиспользуем все поля
            any_old = old_items[0][0]
            if any_old and any_old.get("sha1") == sha1:
                for d, v in old_items:
                    # освежим mtime/size (чтобы было честно)
                    d2 = dict(d)
                    d2["file"] = norm_rel
                    d2["mtime"] = st.st_mtime
                    d2["size"] = st.st_size
                    d2["source"] = source
                    new_docs.append(d2)
                    new_vectors.append(v)
                reused = True

        if reused:
            continue

        content = data.decode("utf-8-sig", errors="replace")
        if is_ckm and not has_language(content, "en"):
            continue
        full_ctx = compact_adl(content, preferred_lang=lang)
        meta = extract_details_metadata(content, preferred_lang=lang)
        actual_lang = detect_content_lang(content, preferred_lang=lang)
        chunks = extract_field_chunks_from_compact(full_ctx, file_name=norm_rel, max_fields=1200)

        for ch in chunks:
            # якорь для эмбеддинга (чуть усиливаем имя поля)
            text = ch.get("text") or ""
            field = ch.get("field") or ""
            at_code = ch.get("at_code") or ""
            rm_type = ch.get("rm_type") or ""
            node_kind = ch.get("node_kind") or ""
            value_types = ch.get("value_types") or []
            occurrences = ch.get("occurrences") or ""
            path = ch.get("path") or ""
            depth = ch.get("depth") or 0
            parent_field = ch.get("parent_field") or ""
            parent_at_code = ch.get("parent_at_code") or ""
            parent_rm_type = ch.get("parent_rm_type") or ""
            parent_desc = ch.get("parent_desc") or ""
            parent_path = ch.get("parent_path") or ""
            has_desc = bool(ch.get("has_desc"))
            anchor = f"{field} {field} {at_code} | {text}".lower()

            d = {
                "file": norm_rel,
                "at_code": at_code,
                "field": field,
                "rm_type": rm_type,
                "node_kind": node_kind,
                "value_types": value_types,
                "occurrences": occurrences,
                "path": path,
                "depth": depth,
                "parent_field": parent_field,
                "parent_at_code": parent_at_code,
                "parent_rm_type": parent_rm_type,
                "parent_desc": parent_desc,
                "parent_path": parent_path,
                "has_desc": has_desc,
                "text": text,
                "anchor": anchor,
                "purpose": meta.get("purpose", "") or "",
                "use": meta.get("use", "") or "",
                "misuse": meta.get("misuse", "") or "",
                "mtime": st.st_mtime,
                "size": st.st_size,
                "sha1": sha1,
                "lang": actual_lang,
                "source": source,
                "parser_version": PARSER_VERSION,
            }
            new_docs.append(d)
            new_vectors.append(None)
            to_embed_indices.append(len(new_docs) - 1)
            to_embed_texts.append(anchor)

    if to_embed_texts:
        total = len(to_embed_texts)
        out_vecs = []
        for start in range(0, total, max(1, int(embed_batch_size))):
            end = min(total, start + max(1, int(embed_batch_size)))
            batch = to_embed_texts[start:end]
            vecs = embedder.encode(batch, show_progress_bar=False, normalize_embeddings=True)
            out_vecs.extend(list(vecs))
            if progress_cb:
                progress_cb(end, total, "Векторизация полей")
        for idx, v in zip(to_embed_indices, out_vecs):
            new_vectors[idx] = v

    if new_vectors:
        vectors_arr = np.vstack([v for v in new_vectors]).astype(np.float32)
    else:
        vectors_arr = np.array([], dtype=np.float32)
    if vectors_arr.size:
        norms = np.linalg.norm(vectors_arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vectors_arr = (vectors_arr / norms).astype(np.float32)

    os.makedirs(paths.dir_path, exist_ok=True)
    embed_name, embed_dim = _embedder_meta(embedder)
    with open(paths.field_docs_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "index_version": INDEX_FORMAT_VERSION,
                META_EMBED_MODEL: embed_name,
                META_EMBED_DIM: embed_dim,
                "docs": new_docs,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    np.save(paths.field_vectors_npy, vectors_arr)
    return new_docs, vectors_arr

