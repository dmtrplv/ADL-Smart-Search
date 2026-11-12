from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    root_dir = Path(__file__).resolve().parents[1]
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    import launcher

    root = Path(launcher._resource_root())  # noqa: SLF001
    assert root.exists(), "resource root does not exist"

    port = launcher._pick_free_port()  # noqa: SLF001
    assert isinstance(port, int) and port > 0

    app_py = root / "app.py"
    assert app_py.exists(), f"app.py not found at {app_py}"

    from adl_smart_search import index_store

    assert getattr(index_store, "INDEX_FORMAT_VERSION", None) is not None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

