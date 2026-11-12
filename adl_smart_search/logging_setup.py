"""Настройка структурированного логирования приложения.

Один файл-лог в каталоге из ADL_LOG_DIR (по умолчанию ./logs): app.log.
Формат: время | уровень | модуль | сообщение. Консольный вывод — только в dev-режиме
(в exe с --noconsole вывода в консоль нет, остаётся только файл). Файл ротируется,
чтобы лог не рос бесконечно (в отличие от старого streamlit.log).
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_LOG_DIR = "logs"
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 3

# Шумные сторонние логгеры — не засоряем app.log их INFO-выводом.
_QUIET_LOGGERS = (
    "streamlit",
    "urllib3",
    "httpx",
    "httpcore",
    "transformers",
    "sentence_transformers",
    "huggingface_hub",
    "PIL",
    "torch",
)

_configured = False


def _log_dir() -> Path:
    env = (os.getenv("ADL_LOG_DIR") or "").strip()
    return Path(env) if env else Path(_DEFAULT_LOG_DIR)


def _is_dev() -> bool:
    from adl_smart_search.ui.settings import is_dev_mode

    return is_dev_mode()


def setup_logging() -> None:
    """Настраивает корневой логгер один раз (idempotent).

    Безопасен при двойном вызове (app.py + launcher.py в frozen-режиме).
    """
    global _configured
    if _configured:
        return
    _configured = True

    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(logging.INFO)

    for name in _QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    try:
        d = _log_dir()
        d.mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(
            str(d / "app.log"),
            maxBytes=_MAX_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter(_LOG_FORMAT))
        root.addHandler(fh)
    except Exception:
        # Если рядом с exe нельзя писать файл — продолжаем без файлового лога.
        pass

    if _is_dev():
        try:
            ch = logging.StreamHandler()
            ch.setFormatter(logging.Formatter(_LOG_FORMAT))
            root.addHandler(ch)
        except Exception:
            pass
