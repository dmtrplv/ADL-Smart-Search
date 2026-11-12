import logging
import os
import sys
import threading

import streamlit as st

from adl_smart_search import __version__
from adl_smart_search.app_paths import (
    CONFIG_DIR_ENV,
    DATA_ROOT_ENV,
    DEFAULT_EMIASTEMPLATES_DIR,
    INDEX_DIR_ENV,
    LOCAL_DIR_ENV,
    REPO_ROOT_ENV,
    USER_CONFIG_DIR_ENV,
    apply_paths_env,
    data_layout,
    data_root_from_repo,
    repo_root_from_data_root,
    resolve_app_paths,
    write_bootstrap,
)
from adl_smart_search.config import from_env
from adl_smart_search.embedder import warmup_embedder
from adl_smart_search.env_file import load_app_envs
from adl_smart_search.index_files import index_files_exist, source_index_dir
from adl_smart_search.index_store import index_model_matches
from adl_smart_search.logging_setup import setup_logging
from adl_smart_search.prompts import PromptConfigError, load_prompts
from adl_smart_search.rag import build_token_vocab
from adl_smart_search.sources import CKM_SOURCE, LOCAL_SOURCE, discover_sources, load_all
from adl_smart_search.ui.context import TabContext, init_session_defaults
from adl_smart_search.ui.indexing import (
    maybe_move_legacy_indexes_to_data,
    norm_rows,
)
from adl_smart_search.ui.llm_provider import (
    get_aux_llm_provider,
    get_embedder,
    get_llm_provider,
    init_llm_session_defaults,
    missing_llm_fields_message,
    selected_chat_provider,
)
from adl_smart_search.ui.settings import (
    apply_llm_settings,
    apply_search_settings,
    dev_mode_available,
    docs_signature,
    is_dev_mode,
    load_effective_llm_settings,
    load_effective_settings,
    remove_user_llm_keys,
)
from adl_smart_search.ui.sidebar import browse_dir, render_dev_settings
from adl_smart_search.ui.tab_archetypes import render_archetype_tab
from adl_smart_search.ui.tab_fields import render_fields_tab
from adl_smart_search.update_client import (
    check_for_update,
    launch_updater,
    restart_target,
    update_check_enabled,
    write_update_job,
)

logger = logging.getLogger(__name__)


def _app_root() -> str:
    """Папка, рядом с которой ищем `.env`: папка exe (frozen) или корень проекта."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _finish_shutdown() -> None:
    """Убирает runtime-файл и завершает процесс после отрисовки страницы."""
    runtime_file = (os.environ.get("ADL_RUNTIME_FILE") or "").strip()
    if runtime_file:
        try:
            os.remove(runtime_file)
        except OSError:
            pass
    try:
        logging.shutdown()
    except Exception:
        pass
    os._exit(0)


def _shutdown_app() -> None:
    """Мягкое завершение: сначала показываем страницу, потом гасим процесс."""
    st.session_state["_shutting_down"] = True
    st.rerun()


def _render_shutdown_page() -> None:
    st.title("ADL Smart Search")
    st.success("Приложение остановлено.")
    st.caption(
        "Сервер остановлен. Вкладку браузера можно закрыть. "
        "Если приложение запущено из терминала, его окно также можно закрыть."
    )
    # Даём браузеру получить и отрисовать этот ответ, затем выходим.
    timer = threading.Timer(1.0, _finish_shutdown)
    timer.daemon = False
    timer.start()


def _env_candidates(paths, install_root: str) -> list[str]:
    """Локальный `.env` важнее общего, общий — важнее старого рядом с exe."""
    return [
        os.path.join(paths.user_config_dir, ".env"),
        os.path.join(paths.config_dir, ".env"),
        os.path.join(paths.install_dir or install_root, ".env"),
    ]


def _needs_data_setup(paths) -> bool:
    if paths.is_dev:
        # Из проекта всегда работаем с проектом; отсутствие data/config — ошибка
        # разработчика, а не экран подключения пользовательского репозитория.
        return False
    return not paths.has_data


def _connect_data_repo(path: str, paths) -> bool:
    selected = (path or "").strip().strip('"').strip("'")
    if not selected:
        st.error("Укажите папку репозитория emiastemplates.")
        return False
    data_root = data_root_from_repo(selected)
    if data_root is None:
        st.error(
            r"Не найдена папка Soft\ADL Smart Search с config/ и data/. "
            "Выберите корень репозитория emiastemplates."
        )
        return False
    if os.path.normcase(os.path.abspath(data_root)) == os.path.normcase(os.path.abspath(selected)):
        repo_root = repo_root_from_data_root(data_root)
    else:
        repo_root = os.path.abspath(selected)
    layout = data_layout(data_root)
    if layout is None:
        return False
    config_dir, data_dir = layout
    write_bootstrap(repo_root, data_root, paths.user_config_dir)
    # Явно публикуем путь: на следующем rerun resolve_app_paths не должен
    # подхватить старый ADL_INDEX_DIR из предыдущего запуска.
    os.environ[REPO_ROOT_ENV] = os.path.abspath(repo_root)
    os.environ[DATA_ROOT_ENV] = data_root
    os.environ[INDEX_DIR_ENV] = data_dir
    os.environ[CONFIG_DIR_ENV] = config_dir
    os.environ[USER_CONFIG_DIR_ENV] = paths.user_config_dir
    os.environ[LOCAL_DIR_ENV] = paths.local_dir
    os.environ.setdefault("ADL_LOG_DIR", paths.logs_dir)
    st.success("Папка с данными подключена.")
    st.rerun()
    return True


def _render_data_setup(paths) -> None:
    title_col, exit_col = st.columns([5, 1], vertical_alignment="center")
    with title_col:
        st.title("ADL Smart Search")
    with exit_col:
        if st.button(
            "Завершить работу",
            use_container_width=True,
            key="setup_exit_btn",
            help="Остановить локальный сервер. Вкладку браузера нужно закрыть вручную.",
        ):
            _shutdown_app()
    st.error("Не найдена папка с данными (индексами и базовыми конфигами).")
    st.caption(
        "По умолчанию приложение использует "
        rf"`{DEFAULT_EMIASTEMPLATES_DIR}\Soft\ADL Smart Search`. "
        "Если папка репозитория emiastemplates на вашем ПК "
        "расположена в другом месте, укажите путь к ней. "
        "Путь сохранится в пользовательском профиле и не будет "
        "перезаписываться обновлениями."
    )

    st.session_state.setdefault("_data_setup_manual", "")
    col_path, col_browse = st.columns([5, 1], vertical_alignment="bottom")
    with col_path:
        manual = st.text_input(
            "Папка репозитория emiastemplates",
            key="_data_setup_manual",
            placeholder=DEFAULT_EMIASTEMPLATES_DIR,
        )
    with col_browse:
        st.button(
            "Обзор",
            use_container_width=True,
            key="_data_setup_browse",
            on_click=browse_dir,
            args=("_data_setup_manual", "Выберите папку emiastemplates"),
        )
    if st.button("Подключить", use_container_width=True):
        _connect_data_repo(manual, paths)

    st.info(
        "Если данных в репозитории ещё нет: "
        "склонируйте/подтяните репозиторий через Git, "
        "затем выберите папку emiastemplates."
    )


def _render_update_notice(paths) -> None:
    """Баннер обновления приложения (только в exe-режиме)."""
    if paths.is_dev or not update_check_enabled():
        return
    info = check_for_update(__version__)
    if not info:
        return
    manifest = info["manifest"]
    latest = str(manifest.get("version") or "").strip()
    if not latest or st.session_state.get("_update_dismissed") == latest:
        return

    notes = str(manifest.get("notes") or "").strip()
    st.info(f"Доступна новая версия приложения: **{latest}** (у вас {__version__}).")
    if notes:
        st.caption(notes)
    c_update, c_later = st.columns(2)
    with c_update:
        if st.button("Обновить сейчас", type="primary", use_container_width=True):
            try:
                job_path, _manifest_path = write_update_job(
                    manifest=manifest,
                    feed_url=info["feed_url"],
                    install_dir=paths.install_dir,
                    restart_exe=restart_target(paths.install_dir),
                    current_version=__version__,
                    channel=info["channel"],
                )
                launch_updater(job_path, paths.install_dir)
            except Exception as e:
                logger.exception("update: cannot start updater")
                st.error(
                    "Не удалось запустить обновление: "
                    f"{e}. Скачайте полный архив и распакуйте вручную."
                )
                return
            st.success("Обновление запущено. Приложение сейчас закроется, откроется updater.")
            _shutdown_app()
    with c_later:
        if st.button("Позже", use_container_width=True):
            st.session_state["_update_dismissed"] = latest
            st.rerun()


def main():
    st.set_page_config(page_title="ADL Smart Search", layout="wide")
    if st.session_state.get("_shutting_down"):
        _render_shutdown_page()
        return

    install_root = _app_root()
    paths = resolve_app_paths(install_root)
    needs_setup = _needs_data_setup(paths)
    if needs_setup:
        # Не фиксируем ADL_INDEX_DIR: иначе после подключения репозитория
        # resolve_app_paths на следующем rerun предпочтёт старый пустой путь.
        os.environ.pop(INDEX_DIR_ENV, None)
        os.environ[REPO_ROOT_ENV] = paths.repo_root
        os.environ[DATA_ROOT_ENV] = paths.data_root
        os.environ[CONFIG_DIR_ENV] = paths.config_dir
        os.environ[USER_CONFIG_DIR_ENV] = paths.user_config_dir
        os.environ[LOCAL_DIR_ENV] = paths.local_dir
        os.environ.setdefault("ADL_LOG_DIR", paths.logs_dir)
    else:
        apply_paths_env(paths)
    env_path = load_app_envs(_env_candidates(paths, install_root))
    setup_logging()
    logger = logging.getLogger("app.main")
    logger.info(
        "paths: repo=%s data_root=%s config=%s data=%s source=%s env=%s",
        paths.repo_root,
        paths.data_root,
        paths.config_dir,
        paths.data_dir,
        paths.source,
        env_path or "-",
    )

    if needs_setup:
        _render_data_setup(paths)
        return

    # Кнопка Deploy нужна только для Streamlit Community Cloud — в локальном приложении скрываем.
    st.markdown(
        '<style>[data-testid="stAppDeployButton"],.stAppDeployButton,'
        '[aria-label="Deploy"]{display:none !important}' ".app-version{"
        "position:fixed;right:0.9rem;bottom:0.45rem;font-size:0.72rem;line-height:1;"
        "opacity:0.35;pointer-events:none;user-select:none;z-index:1000}</style>"
        f'<div class="app-version">v{__version__}</div>',
        unsafe_allow_html=True,
    )

    can_use_dev_mode = dev_mode_available()
    if can_use_dev_mode:
        title_col, dev_col, exit_col = st.columns([5, 1.1, 0.8], vertical_alignment="center")
    else:
        title_col, exit_col = st.columns([6.1, 0.8], vertical_alignment="center")
    with title_col:
        st.title("ADL Smart Search")
    # По умолчанию — пользовательский режим; --dev / ADL_DEV_MODE=1 включают его сразу.
    # В стандартной exe-сборке переключателя нет: dev-режим отключён принудительно.
    if "dev_mode" not in st.session_state or not can_use_dev_mode:
        st.session_state["dev_mode"] = is_dev_mode()
    dev_mode = False
    if can_use_dev_mode:
        with dev_col:
            dev_mode = st.toggle(
                "dev-режим",
                key="dev_mode",
                help="Показать настройки индексации, поиска, LLM и промптов",
            )
    with exit_col:
        if st.button(
            "Завершить работу",
            use_container_width=True,
            key="app_exit_btn",
            help="Остановить локальный сервер. Вкладку браузера нужно закрыть вручную.",
        ):
            _shutdown_app()
    if dev_mode:
        st.session_state["dev_mode_used"] = True

    _render_update_notice(paths)

    logger.info("start: dev=%s", dev_mode)
    cfg = from_env(default_adl_dir="")
    settings = load_effective_settings(cfg.config_dir, cfg.user_config_dir)
    # Пользовательские overrides важнее базового config/ и .env: применяем их до инициализации сессии.
    apply_search_settings(cfg, settings)
    # Бюджеты контекста читаются вкладками из session_state (как dev-виджеты),
    # поэтому заполняем их и в пользовательском режиме.
    for _budget_key in ("context_tokens_archetypes", "context_tokens_fields"):
        st.session_state.setdefault(
            _budget_key, int(getattr(cfg, _budget_key, 10000))
        )
    # Выбор провайдера ИИ больше не сохраняется за пользователем: при запуске
    # всегда действует значение из config/llm.json (по умолчанию МЭШ.ИИ.Хаб).
    try:
        remove_user_llm_keys(cfg.user_config_dir, {"api_kind"})
    except Exception:
        logger.exception("cannot clean legacy api_kind from user overrides")
    llm_persisted = load_effective_llm_settings(cfg.config_dir, cfg.user_config_dir)
    apply_llm_settings(cfg, llm_persisted)
    init_llm_session_defaults(cfg, llm_persisted)
    try:
        prompts = load_prompts(cfg.config_dir)
    except PromptConfigError as e:
        # Промпты нужны только для запросов к LLM. Поиск и индексация
        # продолжают работать; ошибка появится при обращении к ИИ.
        prompts = {}
        logger.warning("prompts unavailable: %s", e)
    init_session_defaults(prompts)

    state = render_dev_settings(cfg, settings, dev_mode, write_config=paths.is_dev)

    if not os.path.isdir(state.adl_dir):
        # Индексы лежат в <index_dir>/local и <index_dir>/ckm.
        local_idx_dir = source_index_dir(state.index_dir or "./data", LOCAL_SOURCE)
        ckm_idx_dir = source_index_dir(state.index_dir or "./data", CKM_SOURCE)
        if not (index_files_exist(local_idx_dir) or index_files_exist(ckm_idx_dir)):
            if paths.is_dev:
                st.error(
                    "В проекте нет индексов. Ожидались "
                    f"{local_idx_dir} или {ckm_idx_dir}. "
                    "Соберите индексы (dev-панель или scripts/build-index.ps1) "
                    "или задайте ADL_REPO_ROOT / ADL_DATA_ROOT / ADL_INDEX_DIR."
                )
                st.stop()
            if dev_mode:
                st.error("Папка с .adl не найдена.")
                st.stop()
            else:
                st.error("Индекс не найден. Включите dev-режим переключателем сверху и постройте индекс.")
                st.stop()

    state.apply_to_cfg(cfg)
    # Провайдер ИИ выбирается радиокнопкой в чате. Это должно произойти
    # до get_llm_provider(), иначе выбор в интерфейсе не влияет на запросы.
    cfg.api_kind = selected_chat_provider()

    # Миграция индексов в ./data при необходимости (только для старой раскладки).
    if not cfg.repo_root:
        maybe_move_legacy_indexes_to_data(state.index_dir)

    sources = discover_sources(cfg)
    merged = None

    # Загружаем индексы только если они уже построены (без автопостроения)
    docs = st.session_state.get("docs")
    vectors = st.session_state.get("vectors")
    field_docs = st.session_state.get("field_docs")
    field_vectors = st.session_state.get("field_vectors")

    if docs is None or vectors is None or field_docs is None or field_vectors is None:
        merged = load_all(cfg, sources)
        docs = merged.docs
        vectors = merged.vectors
        field_docs = merged.field_docs
        field_vectors = merged.field_vectors
        st.session_state["docs"] = docs
        st.session_state["vectors"] = vectors
        st.session_state["field_docs"] = field_docs
        st.session_state["field_vectors"] = field_vectors

    # Нормализация векторов (cosine via dot product). Нужна для стабильного масштаба similarity,
    # иначе keyword_bonus может "тонуть" на ненормализованных dot-product значениях.
    vectors = norm_rows(vectors)
    st.session_state["vectors"] = vectors
    field_vectors = norm_rows(field_vectors)
    st.session_state["field_vectors"] = field_vectors

    # Словари токенов (для поддержки коротких запросов без хардкода)
    if "arch_token_vocab" not in st.session_state:
        st.session_state["arch_token_vocab"] = set()
    if "field_token_vocab" not in st.session_state:
        st.session_state["field_token_vocab"] = set()

    arch_sig = docs_signature(docs)
    field_sig = docs_signature(field_docs)
    if st.session_state.get("arch_token_vocab_sig") != arch_sig:
        st.session_state["arch_token_vocab"] = build_token_vocab(docs or [], field="anchor")
        st.session_state["arch_token_vocab_sig"] = arch_sig
    if st.session_state.get("field_token_vocab_sig") != field_sig:
        st.session_state["field_token_vocab"] = build_token_vocab(field_docs or [], field="anchor")
        st.session_state["field_token_vocab_sig"] = field_sig

    embedder = get_embedder(cfg.embed_model_name)

    # Предсобранный индекс мог быть построен другой моделью эмбеддингов.
    # Сверяем по embed_model из меты БЕЗ загрузки модели (загрузка SentenceTransformer ~10с).
    # Если имя совпадает — индекс совместим, пропускаем. При legacy/отличии — помечаем
    # index_dir'ы для ленивой проверки dim при первом поиске (там модель и так нужна на encode).
    pending_dim_check: list[tuple[str, str]] = []
    if merged is not None:
        for bundle in merged.bundles:
            if not bundle.is_built:
                continue
            for kind in ("archetype", "field"):
                match = index_model_matches(bundle.spec.index_dir, cfg.embed_model_name, kind)
                if match is False or match is None:
                    pending_dim_check.append((bundle.spec.index_dir, kind))
    st.session_state["embed_dim_check_pending"] = pending_dim_check

    llm_provider = None
    llm_aux_provider = None
    llm_setup_msg = missing_llm_fields_message(cfg)
    if llm_setup_msg:
        if dev_mode:
            llm_setup_msg += " Блок «Настройки (dev)» → «LLM»."
        else:
            llm_setup_msg += (
                " Включите dev-режим переключателем сверху и заполните «LLM»"
                " (настройки сохранятся) или задайте ключи в config/.env репозитория с данными."
            )
    if not llm_setup_msg:
        try:
            llm_provider = get_llm_provider(cfg)
        except Exception as e:
            llm_provider = None
            # Не падаем при рендеринге: покажем, когда пользователь нажмёт "Спросить у ИИ"
            st.session_state["last_error"] = f"Ошибка настройки LLM: {e}"
            logger.warning("LLM setup failed: %s", e)
        if llm_provider is not None:
            try:
                llm_aux_provider = get_aux_llm_provider(cfg)
            except Exception as e:
                # Вспомогательная модель не должна ломать ответы: откатываемся
                # на основную модель, перевод/расширение продолжат работать.
                llm_aux_provider = llm_provider
                logger.warning("Aux LLM setup failed, using main provider: %s", e)

    ctx = TabContext(
        dev_mode=dev_mode,
        cfg=cfg,
        adl_dir=state.adl_dir,
        index_dir=state.index_dir,
        preferred_lang=cfg.preferred_lang,
        sources=sources,
        source_labels={s.name: s.label for s in sources},
        docs=docs,
        vectors=vectors,
        field_docs=field_docs,
        field_vectors=field_vectors,
        embedder=embedder,
        llm_provider=llm_provider,
        llm_aux_provider=llm_aux_provider,
        llm_setup_msg=llm_setup_msg,
        auto_strictness=float(state.auto_strictness),
        keyword_bonus=float(state.keyword_bonus),
        misuse_penalty=float(state.misuse_penalty),
        weak_match_penalty=float(state.weak_match_penalty),
        enable_misuse_penalty=bool(state.enable_misuse_penalty),
        enable_weak_match_penalty=bool(state.enable_weak_match_penalty),
        max_results=int(state.max_results),
        sim_floor=float(state.sim_floor),
        sim_floor_pure=float(state.sim_floor_pure),
        sim_floor_pure_cross=float(cfg.sim_floor_pure_cross),
        sim_low_confidence=float(cfg.sim_low_confidence),
    )

    view = st.segmented_control(
        "Раздел",
        options=["Архетипы", "Поля"],
        default="Архетипы",
        key="active_view",
        label_visibility="collapsed",
    )
    if view == "Поля":
        render_fields_tab(ctx)
    else:
        render_archetype_tab(ctx)

    # Старый "одноэкранный" режим поиска/LLM удалён — используйте разделы "Архетипы" и "Поля".

    # Модель эмбеддингов не грузим на старте (проверка совместимости — по имени, см. выше).
    # Прогреваем её в фоне: пока пользователь читает раздел/печатает запрос, torch+веса
    # догружаются в RAM, первый поиск не «зависает». Если модель не успеет — поиск покажет спиннер.
    warmup_embedder(cfg.embed_model_name)


if __name__ == "__main__":
    main()
