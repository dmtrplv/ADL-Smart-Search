"""Проверка обновлений приложения и запуск updater-процесса."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time

import requests

from adl_smart_search import __version__
from adl_smart_search.app_paths import local_app_dir, local_config_dir
from adl_smart_search.update_manifest import (
    is_newer,
    save_manifest,
    validate_manifest,
)

logger = logging.getLogger(__name__)

DEFAULT_UPDATE_BASE = "https://raw.githubusercontent.com/dmtrplv/ADL-Smart-Search/main/update"
UPDATE_CACHE_FILENAME = "update_check.json"
UPDATE_CHECK_SECONDS = 24 * 60 * 60
UPDATER_EXE_NAME = "ADL-Smart-Search-Updater.exe"

_UPDATE_CHECK_ENV = "ADL_UPDATE_CHECK"
_UPDATE_FEED_ENV = "ADL_UPDATE_FEED"
_UPDATE_CHANNEL_ENV = "ADL_UPDATE_CHANNEL"


def update_check_enabled() -> bool:
    value = (os.environ.get(_UPDATE_CHECK_ENV) or "1").strip().lower()
    return value not in ("0", "false", "no", "off")


def update_channel() -> str:
    value = (os.environ.get(_UPDATE_CHANNEL_ENV) or "stable").strip().lower()
    return value if value in ("stable", "beta") else "stable"


def update_feed_url(channel: str | None = None) -> str:
    explicit = (os.environ.get(_UPDATE_FEED_ENV) or "").strip()
    if explicit:
        return explicit
    return f"{DEFAULT_UPDATE_BASE}/{channel or update_channel()}.json"


def update_cache_path() -> str:
    return os.path.join(local_config_dir(), UPDATE_CACHE_FILENAME)


def _read_cache() -> dict:
    try:
        with open(update_cache_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        logger.exception("update: cannot read cache")
        return {}


def _write_cache(data: dict) -> None:
    try:
        path = update_cache_path()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except Exception:
        logger.exception("update: cannot write cache")


def fetch_manifest(url: str, *, timeout: float = 6.0) -> dict:
    """Скачивает и валидирует манифест. При ошибке поднимает исключение."""
    response = requests.get(url, timeout=timeout, headers={"User-Agent": f"ADL-Smart-Search/{__version__}"})
    response.raise_for_status()
    data = response.json()
    validate_manifest(data)
    return data


def check_for_update(current_version: str | None = None, *, force: bool = False) -> dict | None:
    """Возвращает словарь с манифестом, если доступна более новая версия."""
    if not update_check_enabled():
        return None
    current = str(current_version or __version__)
    channel = update_channel()
    url = update_feed_url(channel)
    now = time.time()
    cache = _read_cache()

    cached_manifest = cache.get("manifest") if cache.get("feed_url") == url else None
    checked_at = float(cache.get("checked_at") or 0)
    recent = (now - checked_at) < UPDATE_CHECK_SECONDS

    if not force and recent:
        if not cached_manifest:
            # Недавняя неудачная проверка: не дёргаем сеть на каждом rerun.
            return None
        manifest = cached_manifest
    else:
        try:
            manifest = fetch_manifest(url, timeout=4.0)
            _write_cache(
                {
                    "checked_at": now,
                    "feed_url": url,
                    "version": manifest.get("version"),
                    "manifest": manifest,
                }
            )
        except Exception as e:
            logger.info("update: check failed: %s", e)
            if cached_manifest:
                # Не теряем ранее полученный манифест из-за временной сети.
                manifest = cached_manifest
                _write_cache(
                    {
                        "checked_at": now,
                        "feed_url": url,
                        "version": manifest.get("version"),
                        "manifest": manifest,
                    }
                )
            else:
                # Запоминаем неудачную проверку, чтобы не тормозить каждый rerun.
                _write_cache(
                    {
                        "checked_at": now,
                        "feed_url": url,
                        "error": str(e)[:500],
                    }
                )
                return None

    try:
        validate_manifest(manifest)
    except Exception:
        return None
    if not is_newer(str(manifest.get("version") or ""), current):
        return None
    return {"manifest": manifest, "feed_url": url, "current_version": current, "channel": channel}


def _updates_dir() -> str:
    return os.path.join(local_app_dir(), "updates")


def write_update_job(
    *,
    manifest: dict,
    feed_url: str,
    install_dir: str,
    restart_exe: str,
    current_version: str,
    channel: str,
) -> tuple[str, str]:
    """Сохраняет манифест и job-файл; возвращает (job_path, manifest_path)."""
    updates = _updates_dir()
    os.makedirs(updates, exist_ok=True)
    version = str(manifest.get("version") or "unknown")
    manifest_path = os.path.join(updates, f"manifest-{version}.json")
    save_manifest(manifest_path, manifest)
    job_path = os.path.join(updates, "job.json")
    job = {
        "schema": 1,
        "manifest_path": manifest_path,
        "feed_url": feed_url,
        "install_dir": os.path.abspath(install_dir),
        "restart_exe": os.path.abspath(restart_exe),
        "wait_pid": os.getpid(),
        "current_version": str(current_version),
        "channel": str(channel),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    tmp = f"{job_path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(job, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, job_path)
    return job_path, manifest_path


def launch_updater(job_path: str, install_dir: str) -> subprocess.Popen:
    """Копирует updater во временную папку и запускает его.

    Временная копия нужна, чтобы updater мог заменить собственный exe в
    установленной папке.
    """
    source = os.path.join(install_dir, UPDATER_EXE_NAME)
    if not os.path.isfile(source):
        raise FileNotFoundError(f"updater not found: {source}")
    temp_dir = tempfile.gettempdir()
    target = os.path.join(temp_dir, f"ADL-Smart-Search-Updater-{os.getpid()}.exe")
    shutil.copy2(source, target)
    creationflags = 0
    if os.name == "nt":
        creationflags = 0x00000010  # CREATE_NEW_CONSOLE
    return subprocess.Popen(
        [target, "--job", job_path],
        cwd=install_dir,
        creationflags=creationflags,
    )


def restart_target(install_dir: str) -> str:
    candidate = os.path.join(install_dir, "ADL-Smart-Search.exe")
    if os.path.isfile(candidate):
        return candidate
    return sys.executable
