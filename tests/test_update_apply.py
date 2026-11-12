import zipfile

import pytest

from adl_smart_search.update_apply import (
    apply_archive,
    choose_asset,
    extract_archive,
    verify_files,
)
from adl_smart_search.update_manifest import build_manifest, create_delta_zip


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _manifest(root, version, managed=("app.exe", "_internal")):
    return build_manifest(str(root), version=version, managed_roots=managed)


def test_apply_delta_updates_changed_and_removes_stale(tmp_path):
    install = tmp_path / "install"
    _write(install, "app.exe", "v1")
    _write(install, "_internal/a.dll", "same")
    _write(install, "_internal/stale.dll", "old")
    old = _manifest(install, "1.0.0")

    new_root = tmp_path / "new"
    _write(new_root, "app.exe", "v2")
    _write(new_root, "_internal/a.dll", "same")
    new = _manifest(new_root, "1.1.0")
    delta = tmp_path / "delta.zip"
    create_delta_zip(str(new_root), old, new, str(delta))

    apply_archive(
        install_dir=str(install),
        archive=str(delta),
        manifest=new,
        asset_kind="delta",
        work_dir=str(tmp_path / "work"),
    )

    assert (install / "app.exe").read_text(encoding="utf-8") == "v2"
    assert (install / "_internal/a.dll").read_text(encoding="utf-8") == "same"
    assert not (install / "_internal/stale.dll").exists()
    assert verify_files(str(install), new) == []


def test_apply_full_replaces_everything(tmp_path):
    install = tmp_path / "install"
    _write(install, "app.exe", "old")
    _write(install, "_internal/old.dll", "old")

    new_root = tmp_path / "new"
    _write(new_root, "app.exe", "new")
    _write(new_root, "_internal/new.dll", "new")
    new = _manifest(new_root, "2.0.0")
    full = tmp_path / "full.zip"
    with zipfile.ZipFile(full, "w") as zf:
        for rel in ("app.exe", "_internal/new.dll"):
            zf.write(new_root / rel, arcname=rel)

    apply_archive(
        install_dir=str(install),
        archive=str(full),
        manifest=new,
        asset_kind="full",
        work_dir=str(tmp_path / "work"),
    )

    assert (install / "app.exe").read_text(encoding="utf-8") == "new"
    assert (install / "_internal/new.dll").exists()
    assert not (install / "_internal/old.dll").exists()
    assert verify_files(str(install), new) == []


def test_extract_rejects_path_traversal(tmp_path):
    archive = tmp_path / "evil.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../evil.txt", "bad")

    with pytest.raises(ValueError):
        extract_archive(str(archive), str(tmp_path / "out"))


def test_choose_asset_prefers_matching_delta():
    manifest = {
        "assets": {
            "full": {"url": "full.zip"},
            "deltas": {"1.0.0": {"url": "delta.zip"}},
        }
    }

    assert choose_asset(manifest, "1.0.0") == ("delta", {"url": "delta.zip"})
    assert choose_asset(manifest, "0.9.0") == ("full", {"url": "full.zip"})
