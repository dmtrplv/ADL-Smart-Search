"""Единый источник истины для имён файлов индексов."""

import os

INDEX_FILE_NAMES = {
    "archetype": ("archetype_data.json", "archetype_vectors.npy"),
    "field": ("field_data.json", "field_vectors.npy"),
}

ALL_INDEX_FILE_NAMES = tuple(fn for names in INDEX_FILE_NAMES.values() for fn in names)

# Подпапки источников внутри корня индексов (root/local, root/ckm).
LOCAL_SOURCE = "local"
CKM_SOURCE = "ckm"

SOURCE_SUBDIRS = {
    LOCAL_SOURCE: "local",
    CKM_SOURCE: "ckm",
}


def source_index_dir(root: str, source: str) -> str:
    """Папка индекса конкретного источника внутри общего корня индексов."""
    sub = SOURCE_SUBDIRS.get(source, source)
    return os.path.join(root or ".", sub)


def index_filenames(kind: str) -> tuple[str, str]:
    """(docs_json, vectors_npy) для kind: 'archetype' | 'field'."""
    return INDEX_FILE_NAMES.get(kind, INDEX_FILE_NAMES["archetype"])


def index_files_exist(index_dir: str) -> bool:
    """Есть ли хоть один файл индекса в папке (готовый пресобранный индекс)."""
    try:
        return any(os.path.exists(os.path.join(index_dir, fn)) for fn in ALL_INDEX_FILE_NAMES)
    except OSError:
        return False


def index_kind_files_exist(index_dir: str, kind: str) -> bool:
    """Построен ли конкретный вид индекса (архетипы или поля): оба файла на месте."""
    try:
        return all(os.path.exists(os.path.join(index_dir, fn)) for fn in index_filenames(kind))
    except OSError:
        return False