"""PyInstaller runtime hook: make bundled NumPy/SciPy DLLs discoverable.

In frozen builds Windows does not always inherit the package-local DLL search
paths (``numpy.libs``, ``scipy.libs``, ``sklearn/.libs``). NumPy normally adds
``numpy.libs`` itself, but that happens after the extension loader has already
started resolving dependencies. Registering the directories here runs before
any application import and fixes ``numpy._core.multiarray failed to import``.
"""

import os
import sys


def _add_dll_dir(path: str) -> None:
    if not path or not os.path.isdir(path):
        return
    try:
        os.add_dll_directory(path)  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass


if getattr(sys, "frozen", False):
    base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(sys.executable)))
    for relative in (
        "",
        "numpy.libs",
        "scipy.libs",
        "sklearn/.libs",
        "torch/lib",
    ):
        _add_dll_dir(os.path.join(base, relative))
