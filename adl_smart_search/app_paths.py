"""Определение путей к внешнему Git-репозиторию, данным и конфигам.

Термины:

* ``repo_root`` — корень Git-репозитория (например, ``C:\\emiastemplates``);
* ``data_root`` — папка с ``config/`` и ``data/``
  (например, ``C:\\emiastemplates\\Soft\\ADL Smart Search``).

Есть два режима:

* запуск из проекта (``dev``) — данные и базовые конфиги берутся из
  ``<project>/data`` и ``<project>/config``, локальные overrides/логи — из
  ``<project>/.adl-local``;
* собранный exe (``exe``) — данные и конфиги берутся из штатной папки
  ``C:\\emiastemplates\\Soft\\ADL Smart Search``. Если её нет, пользователь
  выбирает папку репозитория ``emiastemplates``; путь запоминается в
  ``bootstrap.json``.

Пользовательские настройки и путь к репозиторию никогда не попадают в сам
репозиторий, поэтому обновление приложения и ``git pull`` их не затирают.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass

from adl_smart_search.index_files import index_files_exist

logger = logging.getLogger(__name__)

APP_DIR_NAME = "ADL-Smart-Search"
DEV_LOCAL_DIR_NAME = ".adl-local"
BOOTSTRAP_FILENAME = "bootstrap.json"
APP_MODE_ENV = "ADL_APP_MODE"

REPO_ROOT_ENV = "ADL_REPO_ROOT"
DATA_ROOT_ENV = "ADL_DATA_ROOT"
INDEX_DIR_ENV = "ADL_INDEX_DIR"
CONFIG_DIR_ENV = "ADL_CONFIG_DIR"
USER_CONFIG_DIR_ENV = "ADL_USER_CONFIG_DIR"
LOCAL_DIR_ENV = "ADL_LOCAL_DIR"

# Штатное место репозитория emiastemplates и папки с config/ и data/.
DEFAULT_EMIASTEMPLATES_DIR = r"C:\emiastemplates"
EMIASTEMPLATES_RELATIVE_PARTS = ("Soft", "ADL Smart Search")


@dataclass(frozen=True)
class AppPaths:
    """Разрешённые пути приложения."""

    install_dir: str
    repo_root: str
    data_root: str
    config_dir: str
    data_dir: str
    user_config_dir: str
    logs_dir: str
    local_dir: str = ""
    mode: str = "exe"
    source: str = "default"

    @property
    def has_data(self) -> bool:
        """Найдена папка с ``config/`` и ``data/``."""
        return bool(self.data_root)

    @property
    def is_dev(self) -> bool:
        return self.mode == "dev"


def app_mode() -> str:
    """Режим путей: ``dev`` (запуск из проекта) или ``exe`` (собранное приложение)."""
    explicit = (os.environ.get(APP_MODE_ENV) or "").strip().lower()
    if explicit in ("dev", "exe"):
        return explicit
    return "exe" if getattr(sys, "frozen", False) else "dev"


def local_app_dir(default_dir: str | None = None) -> str:
    """Каталог локальных настроек/логов.

    Явный ``ADL_LOCAL_DIR`` всегда важнее. Если он не задан, используется
    ``default_dir`` (например, ``<project>/.adl-local``) или ``%LOCALAPPDATA%``.
    """
    explicit = (os.environ.get(LOCAL_DIR_ENV) or "").strip()
    if explicit:
        return os.path.abspath(os.path.expandvars(os.path.expanduser(explicit)))
    if default_dir:
        return os.path.abspath(os.path.expandvars(os.path.expanduser(default_dir)))
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        return os.path.join(base, APP_DIR_NAME)
    return os.path.join(os.path.expanduser("~"), f".{APP_DIR_NAME.lower()}")


def dev_local_dir(install_dir: str) -> str:
    return os.path.join(install_dir or ".", DEV_LOCAL_DIR_NAME)


def local_config_dir(default_dir: str | None = None) -> str:
    return os.path.join(local_app_dir(default_dir), "config")


def local_logs_dir(default_dir: str | None = None) -> str:
    return os.path.join(local_app_dir(default_dir), "logs")


def bootstrap_path(local_config: str | None = None) -> str:
    return os.path.join(local_config or local_config_dir(), BOOTSTRAP_FILENAME)


def _expand_path(value: str) -> str:
    return os.path.abspath(os.path.expandvars(os.path.expanduser(str(value or "").strip())))


def _norm_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        logger.exception("app_paths: cannot read %s", path)
        return {}


def _write_json_atomic(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def data_layout(data_root: str) -> tuple[str, str] | None:
    """Проверяет папку данных ``<data_root>/config`` + ``<data_root>/data``.

    ``data_root`` — это, например, ``C:\\emiastemplates\\Soft\\ADL Smart Search``,
    а не корень Git-репозитория.
    """
    root = _expand_path(data_root) if data_root else ""
    if not root or not os.path.isdir(root):
        return None
    config_dir = os.path.join(root, "config")
    data_dir = os.path.join(root, "data")
    if not os.path.isdir(config_dir) or not os.path.isdir(data_dir):
        return None
    if not os.path.isfile(os.path.join(config_dir, "settings.json")):
        return None
    if not (
        index_files_exist(os.path.join(data_dir, "local"))
        or index_files_exist(os.path.join(data_dir, "ckm"))
    ):
        return None
    return config_dir, data_dir


def data_root_from_repo(repo_root: str) -> str | None:
    """Папка данных внутри Git-репозитория.

    Сначала проверяется штатная ``<repo_root>\\Soft\\ADL Smart Search``,
    затем сам ``repo_root`` — на случай, если config/data лежат прямо в нём.
    """
    root = _expand_path(repo_root) if repo_root else ""
    if not root or not os.path.isdir(root):
        return None
    candidates = [root]
    nested = os.path.join(root, *EMIASTEMPLATES_RELATIVE_PARTS)
    if _norm_key(nested) != _norm_key(root):
        candidates.append(nested)
    for candidate in candidates:
        if data_layout(candidate) is not None:
            return os.path.abspath(candidate)
    return None


def repo_root_from_data_root(data_root: str) -> str:
    """Обратное преобразование: ``...\\Soft\\ADL Smart Search`` -> корень репозитория."""
    root = _expand_path(data_root) if data_root else ""
    if not root:
        return ""
    nested = os.path.join(*EMIASTEMPLATES_RELATIVE_PARTS)
    suffix = os.path.join(os.sep, nested)
    if _norm_key(root).endswith(os.path.normcase(suffix)):
        return os.path.dirname(os.path.dirname(root))
    return root


def read_bootstrap(local_config: str | None = None) -> dict:
    return _read_json(bootstrap_path(local_config))


def bootstrap_repo_root(local_config: str | None = None) -> str:
    """Сохранённый корень Git-репозитория (например, ``C:\\emiastemplates``)."""
    data = read_bootstrap(local_config)
    root = str(data.get("repo_root") or "").strip()
    return _expand_path(root) if root else ""


def bootstrap_data_root(local_config: str | None = None) -> str:
    """Сохранённая папка с ``config/`` и ``data/``."""
    data = read_bootstrap(local_config)
    root = str(data.get("data_root") or "").strip()
    return _expand_path(root) if root else ""


def write_bootstrap(repo_root: str, data_root: str, local_config: str | None = None) -> None:
    """Запоминает корень репозитория и папку данных (вне самого репозитория)."""
    _write_json_atomic(
        bootstrap_path(local_config),
        {
            "schema": 1,
            "repo_root": _expand_path(repo_root) if repo_root else "",
            "data_root": _expand_path(data_root) if data_root else "",
            "configured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
    )


def _paths_from_data_root(
    data_root: str,
    repo_root: str,
    source: str,
    *,
    install_dir: str,
    mode: str,
    local_dir: str,
    user_config_dir: str,
    logs_dir: str,
) -> AppPaths | None:
    layout = data_layout(data_root)
    if layout is None:
        return None
    config_dir, data_dir = layout
    resolved_data = _expand_path(data_root)
    resolved_repo = _expand_path(repo_root) if repo_root else repo_root_from_data_root(resolved_data)
    return AppPaths(
        install_dir=install_dir,
        repo_root=resolved_repo,
        data_root=resolved_data,
        config_dir=config_dir,
        data_dir=data_dir,
        user_config_dir=user_config_dir,
        logs_dir=logs_dir,
        local_dir=local_dir,
        mode=mode,
        source=source,
    )


def resolve_app_paths(
    install_dir: str | None = None,
    *,
    mode: str | None = None,
) -> AppPaths:
    """Определяет пути к репозиторию, данным и конфигам.

    * ``repo_root`` — корень Git-репозитория (например, ``C:\\emiastemplates``).
    * ``data_root`` — папка с ``config/`` и ``data/``
      (например, ``C:\\emiastemplates\\Soft\\ADL Smart Search``).

    В режиме ``dev`` данные берутся из проекта. В ``exe`` — из сохранённого
    ``bootstrap.json`` или из штатной папки ``C:\\emiastemplates\\Soft\\ADL Smart Search``.
    Если ничего не найдено, возвращаются пустые пути — приложение покажет
    экран выбора папки.
    """
    install = _expand_path(install_dir or os.getcwd())
    resolved_mode = (mode or app_mode()).strip().lower()
    if resolved_mode not in ("dev", "exe"):
        resolved_mode = "exe" if getattr(sys, "frozen", False) else "dev"

    default_local = dev_local_dir(install) if resolved_mode == "dev" else None
    local_dir = local_app_dir(default_local)
    user_config = os.path.join(local_dir, "config")
    logs = os.path.join(local_dir, "logs")
    common = {
        "install_dir": install,
        "mode": resolved_mode,
        "local_dir": local_dir,
        "user_config_dir": user_config,
        "logs_dir": logs,
    }

    explicit_repo = (os.environ.get(REPO_ROOT_ENV) or "").strip()
    if explicit_repo:
        data_root = data_root_from_repo(explicit_repo)
        if data_root:
            paths = _paths_from_data_root(data_root, explicit_repo, "env_repo", **common)
            if paths is not None:
                return paths

    explicit_data = (os.environ.get(DATA_ROOT_ENV) or "").strip()
    if explicit_data:
        data_root = _expand_path(explicit_data)
        paths = _paths_from_data_root(data_root, "", "env_data_root", **common)
        if paths is not None:
            return paths

    explicit_index = (os.environ.get(INDEX_DIR_ENV) or "").strip()
    if explicit_index:
        data_dir = _expand_path(explicit_index)
        explicit_config = (os.environ.get(CONFIG_DIR_ENV) or "").strip()
        if explicit_config:
            config_dir = _expand_path(explicit_config)
        else:
            parent = os.path.dirname(data_dir)
            candidate = os.path.join(parent, "config")
            config_dir = candidate if os.path.isdir(candidate) else data_dir
        if _norm_key(os.path.dirname(data_dir)) == _norm_key(os.path.dirname(config_dir)):
            data_root = os.path.dirname(data_dir)
        else:
            data_root = data_dir
        repo_root = repo_root_from_data_root(data_root) if resolved_mode == "exe" else ""
        return AppPaths(
            repo_root=repo_root,
            data_root=data_root,
            config_dir=config_dir,
            data_dir=data_dir,
            source="env_index",
            **common,
        )

    if resolved_mode == "dev":
        config_dir = os.path.join(install, "config")
        data_dir = os.path.join(install, "data")
        project_ok = os.path.isdir(config_dir) and os.path.isdir(data_dir)
        return AppPaths(
            repo_root="",
            data_root=install,
            config_dir=config_dir,
            data_dir=data_dir,
            source="dev_project" if project_ok else "dev_default",
            **common,
        )

    saved_repo = bootstrap_repo_root(user_config)
    saved_data = bootstrap_data_root(user_config)
    if saved_data:
        paths = _paths_from_data_root(saved_data, saved_repo, "bootstrap", **common)
        if paths is not None:
            return paths
        logger.warning("app_paths: bootstrap data root is no longer valid: %s", saved_data)
    if saved_repo:
        resolved_data = data_root_from_repo(saved_repo)
        if resolved_data:
            paths = _paths_from_data_root(resolved_data, saved_repo, "bootstrap_repo", **common)
            if paths is not None:
                try:
                    write_bootstrap(saved_repo, resolved_data, user_config)
                except Exception:
                    logger.exception("app_paths: cannot refresh bootstrap path")
                return paths
        logger.warning("app_paths: bootstrap repo is no longer valid: %s", saved_repo)

    # Штатное место: C:\emiastemplates\Soft\ADL Smart Search.
    fixed_repo = _expand_path(DEFAULT_EMIASTEMPLATES_DIR)
    fixed_data = data_root_from_repo(fixed_repo)
    if fixed_data:
        paths = _paths_from_data_root(fixed_data, fixed_repo, "default_emiastemplates", **common)
        if paths is not None:
            return paths

    return AppPaths(
        repo_root="",
        data_root="",
        config_dir=os.path.join(install, "config"),
        data_dir=os.path.join(install, "data"),
        source="default",
        **common,
    )


def apply_paths_env(paths: AppPaths) -> None:
    """Публикует разрешённые пути в окружении для остального приложения."""
    os.environ[INDEX_DIR_ENV] = paths.data_dir
    os.environ[CONFIG_DIR_ENV] = paths.config_dir
    os.environ[USER_CONFIG_DIR_ENV] = paths.user_config_dir
    os.environ[REPO_ROOT_ENV] = paths.repo_root
    os.environ[DATA_ROOT_ENV] = paths.data_root
    os.environ[APP_MODE_ENV] = paths.mode
    if paths.local_dir:
        os.environ[LOCAL_DIR_ENV] = paths.local_dir
    os.environ.setdefault("ADL_LOG_DIR", paths.logs_dir)
