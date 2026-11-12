"""Сборка индексов без Streamlit: CLI и интерактивное меню.

Используется как разработчиком (перед сборкой дистрибутива), так и отдельным
консольным exe для индексации базы без запуска интерфейса.

Примеры:

    python -m adl_smart_search.rebuild_index
    python -m adl_smart_search.rebuild_index --list
    python -m adl_smart_search.rebuild_index --adl-dir C:\\Archetypes
    python -m adl_smart_search.rebuild_index --source local --kind archetype
    python -m adl_smart_search.rebuild_index --full --source all --kind all
    python -m adl_smart_search.rebuild_index --verify .\\data

`--index-dir` — корень дерева индексов: локальная база пишется в `<root>/local`,
CKM — в `<root>/ckm` (внутри остаются `settings.json` и подпапки источников).

Билдеры инкрементальные: пересчитываются только изменившиеся .adl-файлы.
"""

from __future__ import annotations

import argparse
import os
import sys

from adl_smart_search.config import from_env
from adl_smart_search.embedder import get_default_embedder
from adl_smart_search.index_build_info import save_kind_build_info
from adl_smart_search.index_files import (
    CKM_SOURCE,
    LOCAL_SOURCE,
    index_filenames,
    index_files_exist,
    index_kind_files_exist,
    source_index_dir,
)
from adl_smart_search.ui.settings import (
    index_last_built_label,
    load_settings,
    save_settings,
)

SOURCES = (LOCAL_SOURCE, CKM_SOURCE)
KINDS = ("archetype", "field")


def _embedder_info(embedder) -> tuple[str, int]:
    name = str(getattr(embedder, "model_name", "") or "")
    dim = 0
    dim_fn = getattr(embedder, "embedding_dim", None)
    if callable(dim_fn):
        try:
            dim = int(dim_fn())
        except Exception:
            dim = 0
    return name, dim


def _record_build_info(index_dir: str, source: str, kind: str, embedder) -> None:
    from adl_smart_search.index_store import INDEX_FORMAT_VERSION, PARSER_VERSION

    model_name, embed_dim = _embedder_info(embedder)
    save_kind_build_info(
        os.path.dirname(index_dir),
        source,
        kind,
        parser_version=PARSER_VERSION,
        index_format_version=INDEX_FORMAT_VERSION,
        embed_model=model_name,
        embed_dim=embed_dim,
    )


def _progress_cb(done: int, total: int, label: str) -> None:
    print(f"    {label}: {done}/{total}", flush=True)


def _source_title(source: str) -> str:
    return "Локальная база" if source == LOCAL_SOURCE else "CKM"


def _title(source: str, kind: str) -> str:
    kind_title = "архетипы" if kind == "archetype" else "поля"
    return f"{_source_title(source)} — {kind_title}"


def _delete_kind_files(index_dir: str, kind: str) -> None:
    for fn in index_filenames(kind):
        path = os.path.join(index_dir, fn)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            print(f"    [WARN] Не удалось удалить {path}: {e}")


def build_source(adl_dir: str, index_dir: str, embedder, preferred_lang: str, source: str = LOCAL_SOURCE):
    """Строит (инкрементально) индексы архетипов и полей для одного источника."""
    from adl_smart_search.index_store import (
        build_or_load_field_index_incremental,
        build_or_load_index_incremental,
    )

    print(f"[1/2] Архетипы: {adl_dir} -> {index_dir}")
    docs, vectors = build_or_load_index_incremental(
        adl_dir=adl_dir,
        embedder=embedder,
        preferred_lang=preferred_lang,
        index_dir=index_dir,
        source=source,
        progress_cb=_progress_cb,
    )
    print(f"[2/2] Поля: {adl_dir} -> {index_dir}")
    field_docs, field_vectors = build_or_load_field_index_incremental(
        adl_dir=adl_dir,
        embedder=embedder,
        preferred_lang=preferred_lang,
        index_dir=index_dir,
        source=source,
        progress_cb=_progress_cb,
    )
    _record_build_info(index_dir, source, "archetype", embedder)
    _record_build_info(index_dir, source, "field", embedder)
    return docs, vectors, field_docs, field_vectors


def build_kind(
    adl_dir: str,
    index_dir: str,
    embedder,
    preferred_lang: str,
    *,
    source: str,
    kind: str,
    full: bool = False,
):
    """Строит один вид индекса (archetype/field) для одного источника.

    ``full=True`` сначала удаляет файлы этого вида индекса.
    Для полей нужен уже построенный индекс архетипов — он служит основой.
    """
    from adl_smart_search.index_store import (
        build_or_load_field_index_incremental,
        build_or_load_index_incremental,
    )

    if full:
        _delete_kind_files(index_dir, kind)
    if kind == "field" and not index_kind_files_exist(index_dir, "archetype"):
        print(
            "    [ERROR] Сначала постройте индекс архетипов: "
            "он нужен как основа для индекса полей."
        )
        return None

    if kind == "archetype":
        docs, _ = build_or_load_index_incremental(
            adl_dir=adl_dir,
            embedder=embedder,
            preferred_lang=preferred_lang,
            index_dir=index_dir,
            source=source,
            progress_cb=_progress_cb,
        )
    else:
        docs, _ = build_or_load_field_index_incremental(
            adl_dir=adl_dir,
            embedder=embedder,
            preferred_lang=preferred_lang,
            index_dir=index_dir,
            source=source,
            progress_cb=_progress_cb,
        )
    _record_build_info(index_dir, source, kind, embedder)
    return docs


def summarize_build(model_name: str, embed_dim: int, docs, field_docs) -> None:
    print("")
    print("=== Итог сборки ===")
    print(f"  Модель эмбеддингов: {model_name} (dim={embed_dim})")
    print(f"  Архетипов: {len(docs)}, полей: {len(field_docs)}")


def verify_index(index_dir: str) -> int:
    """Проверяет артефакт: меты (модель/размерность) и число документов. Exit 0 = ок."""
    from adl_smart_search.index_store import index_meta, load_archetype_index, load_field_index

    arch_meta = index_meta(index_dir, "archetype")
    field_meta = index_meta(index_dir, "field")
    arch_dim = arch_meta.get("embed_dim") or 0
    field_dim = field_meta.get("embed_dim") or 0

    try:
        docs, _ = load_archetype_index(index_dir=index_dir)
    except FileNotFoundError:
        docs = []
    try:
        field_docs, _ = load_field_index(index_dir=index_dir)
    except FileNotFoundError:
        field_docs = []

    print(f"=== Индекс: {index_dir} ===")
    print(f"  Архетипы: {len(docs)} документов,")
    print(f"    embed_model={arch_meta.get('embed_model') or '-'}, dim={arch_dim}")
    print(f"  Поля:     {len(field_docs)} документов,")
    print(f"    embed_model={field_meta.get('embed_model') or '-'}, dim={field_dim}")

    if not docs and not field_docs:
        print("[FAIL] Индекс пуст (нет ни архетипов, ни полей).")
        return 1
    if not arch_dim:
        print("[FAIL] У индекса архетипов нет embed_dim (мета отсутствует).")
        return 1
    if arch_dim != field_dim:
        print(f"[FAIL] Размерности архетипов ({arch_dim}) и полей ({field_dim}) не совпадают.")
        return 1
    print("[OK]")
    return 0


def verify_root(root: str) -> int:
    """Проверяет корень дерева индексов: подпапки local/ и ckm/ (если есть).

    Для старой плоской раскладки (файлы индекса прямо в корне) проверяется корень.
    """
    root = root or "./data"
    local_dir = source_index_dir(root, LOCAL_SOURCE)
    ckm_dir = source_index_dir(root, CKM_SOURCE)
    present = [d for d in (local_dir, ckm_dir) if index_files_exist(d)]
    if not present:
        if index_files_exist(root):
            return verify_index(root)
        print(f"[FAIL] Индексы не найдены в корне {root} (ожидались {local_dir} и/или {ckm_dir}).")
        return 1

    rc = 0
    for d in present:
        rc = max(rc, verify_index(d))
    if not index_files_exist(local_dir):
        print(f"[FAIL] Нет индекса локальной базы ({local_dir}).")
        rc = 1
    return rc


def _resolve_paths(args, cfg):
    """Пути из аргументов, data/settings.json или .env — как в dev-панели."""
    settings_root = (args.index_dir or cfg.index_dir or "./data").strip() or "./data"
    settings = load_settings(settings_root)
    index_root = (
        args.index_dir or settings.get("index_dir") or cfg.index_dir or "./data"
    ).strip() or "./data"
    adl_dir = (args.adl_dir or settings.get("adl_dir") or cfg.adl_dir or "").strip()
    ckm_dir = (args.ckm_adl_dir or settings.get("ckm_adl_dir") or cfg.ckm_adl_dir or "").strip()
    return adl_dir, ckm_dir, index_root, settings, settings_root


def _save_paths(settings_root: str, settings: dict, *, adl_dir: str, ckm_dir: str, index_root: str) -> None:
    """Сохраняет пути в data/settings.json, чтобы их видела и dev-панель."""
    data = dict(settings or {})
    if adl_dir:
        data["adl_dir"] = adl_dir
    if ckm_dir:
        data["ckm_adl_dir"] = ckm_dir
    if index_root:
        data["index_dir"] = index_root
    try:
        save_settings(settings_root, data)
    except Exception as e:
        print(f"  [WARN] Не удалось сохранить настройки: {e}")


def _print_status(adl_dir: str, ckm_dir: str, index_root: str) -> None:
    print(f"Корень индексов: {index_root}")
    print(f"Локальная база .adl: {adl_dir or '—'}")
    print(f"CKM .adl: {ckm_dir or '—'}")
    for source in SOURCES:
        idx_dir = source_index_dir(index_root, source)
        for kind in KINDS:
            print(f"  {_title(source, kind)}: {index_last_built_label(idx_dir, kind)}")


def _run_build(
    *,
    adl_dir: str,
    ckm_dir: str,
    index_root: str,
    embedder,
    preferred_lang: str,
    sources: tuple[str, ...],
    kinds: tuple[str, ...],
    full: bool,
) -> int:
    rc = 0
    for source in sources:
        src_adl = adl_dir if source == LOCAL_SOURCE else ckm_dir
        if not src_adl:
            print(f"  [SKIP] {_source_title(source)}: папка .adl не указана.")
            continue
        if not os.path.isdir(src_adl):
            print(f"  [ERROR] {_source_title(source)}: папка не существует: {src_adl}")
            rc = 1
            continue
        idx_dir = source_index_dir(index_root, source)
        os.makedirs(idx_dir, exist_ok=True)
        for kind in kinds:
            title = _title(source, kind)
            suffix = " (полная пересборка)" if full else ""
            print(f"  {title}: {src_adl} -> {idx_dir}{suffix}")
            docs = build_kind(
                src_adl,
                idx_dir,
                embedder,
                preferred_lang,
                source=source,
                kind=kind,
                full=full,
            )
            if docs is None:
                rc = 1
                continue
            print(f"  Готово: {title} = {len(docs)}")
        if full and kinds == ("archetype",):
            print("  [WARN] Индекс полей не пересобирался; при необходимости запустите его отдельно.")
    return rc


def _load_embedder(model_name: str):
    try:
        return get_default_embedder(model_name)
    except Exception as e:
        print(f"[ERROR] Не удалось загрузить модель эмбеддингов '{model_name}': {e}")
        return None


def _prompt_path(prompt: str) -> str:
    try:
        return input(prompt).strip().strip('"').strip("'")
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def interactive_menu(
    *,
    adl_dir: str,
    ckm_dir: str,
    index_root: str,
    settings: dict,
    settings_root: str,
    embed_model: str,
    preferred_lang: str,
) -> int:
    """Меню, повторяющее варианты кнопок dev-панели."""
    embedder = None

    def _ensure_embedder():
        nonlocal embedder
        if embedder is None:
            print(f"Модель эмбеддингов: {embed_model}")
            embedder = _load_embedder(embed_model)
        return embedder

    if not adl_dir:
        value = _prompt_path("Папка с .adl (локальная база), Enter — пропустить: ")
        if value:
            if not os.path.isdir(value):
                print(f"  [WARN] Папка не найдена: {value}")
            adl_dir = value
            _save_paths(settings_root, settings, adl_dir=adl_dir, ckm_dir=ckm_dir, index_root=index_root)
    if not ckm_dir:
        value = _prompt_path("Папка CKM .adl (опционально), Enter — пропустить: ")
        if value:
            if not os.path.isdir(value):
                print(f"  [WARN] Папка не найдена: {value}")
            ckm_dir = value
            _save_paths(settings_root, settings, adl_dir=adl_dir, ckm_dir=ckm_dir, index_root=index_root)

    while True:
        print()
        print("=" * 60)
        print(" ADL Smart Search — индексация без интерфейса")
        print("=" * 60)
        _print_status(adl_dir, ckm_dir, index_root)
        print()
        print(" 1) Локальная база — архетипы")
        print(" 2) Локальная база — поля")
        print(" 3) CKM — архетипы")
        print(" 4) CKM — поля")
        print(" 5) Индексировать всё (инкрементально)")
        print(" 6) Полная пересборка (удалить и собрать заново)")
        print(" 7) Проверить индексы")
        print(" 8) Показать настройки и статус")
        print(" 0) Выход")

        choice = _prompt_path("Выберите пункт: ")
        if not choice:
            return 0
        if choice == "0":
            return 0

        if choice in ("1", "2", "3", "4"):
            source = LOCAL_SOURCE if choice in ("1", "2") else CKM_SOURCE
            kind = "archetype" if choice in ("1", "3") else "field"
            emb = _ensure_embedder()
            if emb is None:
                continue
            rc = _run_build(
                adl_dir=adl_dir,
                ckm_dir=ckm_dir,
                index_root=index_root,
                embedder=emb,
                preferred_lang=preferred_lang,
                sources=(source,),
                kinds=(kind,),
                full=False,
            )
        elif choice == "5":
            emb = _ensure_embedder()
            if emb is None:
                continue
            rc = _run_build(
                adl_dir=adl_dir,
                ckm_dir=ckm_dir,
                index_root=index_root,
                embedder=emb,
                preferred_lang=preferred_lang,
                sources=SOURCES,
                kinds=KINDS,
                full=False,
            )
        elif choice == "6":
            answer = _prompt_path(
                "Будут удалены выбранные файлы индексов. Введите «да» для подтверждения: "
            ).lower()
            if answer not in ("да", "yes", "y"):
                print("Отменено.")
                continue
            emb = _ensure_embedder()
            if emb is None:
                continue
            rc = _run_build(
                adl_dir=adl_dir,
                ckm_dir=ckm_dir,
                index_root=index_root,
                embedder=emb,
                preferred_lang=preferred_lang,
                sources=SOURCES,
                kinds=KINDS,
                full=True,
            )
        elif choice == "7":
            rc = verify_root(index_root)
        elif choice == "8":
            _print_status(adl_dir, ckm_dir, index_root)
            rc = 0
        else:
            print("Неизвестный пункт, попробуйте ещё раз.")
            continue
        print(f"Код завершения: {rc}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m adl_smart_search.rebuild_index",
        description="Сборка/проверка индексов без Streamlit.",
    )
    parser.add_argument("--adl-dir", default="", help="Папка с .adl (локальная база).")
    parser.add_argument(
        "--index-dir",
        default="",
        help="Корень дерева индексов (default: ./data); локальная база -> <root>/local, CKM -> <root>/ckm.",
    )
    parser.add_argument("--ckm-adl-dir", default="", help="Папка CKM .adl (опционально).")
    parser.add_argument("--embed-model", default="", help="Модель эмбеддингов (default: ADL_EMBED_MODEL/встроенная).")
    parser.add_argument("--lang", default="", help="Предпочитаемый язык (default: ru).")
    parser.add_argument(
        "--source",
        choices=["all", LOCAL_SOURCE, CKM_SOURCE],
        default="all",
        help="Источник для сборки (default: all — все доступные).",
    )
    parser.add_argument(
        "--kind",
        choices=["all", "archetype", "field"],
        default="all",
        help="Вид индекса (default: all — архетипы и поля).",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Полная пересборка: удалить файлы выбранных индексов перед сборкой.",
    )
    parser.add_argument(
        "--verify",
        nargs="?",
        const="",
        default=None,
        help="Не собирать, а проверить индексы (можно указать корень; по умолчанию текущий).",
    )
    parser.add_argument("--list", action="store_true", help="Показать пути и статус индексов.")
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Открыть интерактивное меню (как кнопки в dev-панели).",
    )
    args = parser.parse_args(argv)
    raw_args = list(sys.argv[1:] if argv is None else argv)

    cfg = from_env("")
    adl_dir, ckm_dir, index_root, settings, settings_root = _resolve_paths(args, cfg)
    preferred_lang = args.lang or cfg.preferred_lang or "ru"

    if args.verify is not None:
        return verify_root(args.verify or index_root)
    if args.list:
        _print_status(adl_dir, ckm_dir, index_root)
        return 0

    interactive = bool(args.interactive) or not raw_args

    embed_model = args.embed_model or cfg.embed_model_name

    if interactive:
        return interactive_menu(
            adl_dir=adl_dir,
            ckm_dir=ckm_dir,
            index_root=index_root,
            settings=settings,
            settings_root=settings_root,
            embed_model=embed_model,
            preferred_lang=preferred_lang,
        )

    if args.source == LOCAL_SOURCE and not adl_dir:
        print("[ERROR] Не указана папка с .adl для локальной базы: --adl-dir (или ADL_DIR).")
        return 1
    if args.source == CKM_SOURCE and not ckm_dir:
        print("[ERROR] Не указана папка CKM .adl: --ckm-adl-dir (или ADL_CKM_DIR).")
        return 1
    if args.source == "all" and not adl_dir and not ckm_dir:
        print("[ERROR] Не указана ни одна папка с .adl: --adl-dir / --ckm-adl-dir.")
        return 1

    embedder = _load_embedder(embed_model)
    if embedder is None:
        return 1

    sources = SOURCES if args.source == "all" else (args.source,)
    kinds = KINDS if args.kind == "all" else (args.kind,)

    print(f"Модель эмбеддингов: {getattr(embedder, 'model_name', embed_model)}")
    rc = _run_build(
        adl_dir=adl_dir,
        ckm_dir=ckm_dir,
        index_root=index_root,
        embedder=embedder,
        preferred_lang=preferred_lang,
        sources=sources,
        kinds=kinds,
        full=args.full,
    )
    print("")
    print("Готово. Файлы индекса лежат в", index_root)
    return rc


if __name__ == "__main__":
    sys.exit(main())
