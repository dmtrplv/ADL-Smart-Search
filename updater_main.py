"""Entry point для ADL-Smart-Search-Updater.exe."""

import logging
import os
import sys

from adl_smart_search.app_paths import local_logs_dir
from adl_smart_search.update_apply import main as apply_main


def _setup_logging() -> None:
    log_dir = local_logs_dir()
    try:
        os.makedirs(log_dir, exist_ok=True)
        handler = logging.FileHandler(os.path.join(log_dir, "updater.log"), encoding="utf-8")
    except Exception:
        handler = logging.StreamHandler()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[handler],
    )


def main() -> int:
    _setup_logging()
    rc = apply_main(sys.argv[1:])
    if rc != 0:
        try:
            input("\nНажмите Enter, чтобы закрыть окно...")
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
