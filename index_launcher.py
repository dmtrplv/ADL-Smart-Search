"""Тонкий лаунчер индексатора.

Собирается в ADL-Smart-Search-Index.exe. Сам не содержит тяжёлых библиотек:
находит основной ADL-Smart-Search-*.exe рядом с собой и запускает его с --index.
Основной exe уже работает в onedir-режиме, поэтому использует свои
torch/sentence-transformers из _internal и не распаковывает их заново.
"""

from __future__ import annotations

import glob
import os
import subprocess
import sys


def _resource_root() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _find_main_exe(root: str) -> str | None:
    fixed = os.path.join(root, "ADL-Smart-Search.exe")
    if os.path.exists(fixed):
        return fixed
    candidates = [
        p
        for p in glob.glob(os.path.join(root, "ADL-Smart-Search-*.exe"))
        if not os.path.basename(p).lower().startswith(
            ("adl-smart-search-index", "adl-smart-search-updater")
        )
    ]
    if not candidates:
        return None
    # В релизной папке основной exe один; если рядом оказалось несколько —
    # берём самый свежий.
    return max(candidates, key=lambda p: os.path.getmtime(p))


def main() -> int:
    root = _resource_root()
    main_exe = _find_main_exe(root)
    if not main_exe:
        print("Не найден основной ADL-Smart-Search-*.exe рядом с индексатором.")
        return 1
    args = [main_exe, "--index", *sys.argv[1:]]
    return subprocess.run(args, cwd=root).returncode


if __name__ == "__main__":
    raise SystemExit(main())
