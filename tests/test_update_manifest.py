import json
import zipfile

import pytest

from adl_smart_search.update_manifest import (
    build_manifest,
    compare_manifests,
    create_delta_zip,
    is_newer,
    load_manifest,
    save_manifest,
    validate_manifest,
)


def _write(root, rel, text):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_build_manifest_hashes_program_files_only(tmp_path):
    _write(tmp_path, "app.exe", "v1")
    _write(tmp_path, "_internal/a.dll", "dll")
    _write(tmp_path, "data/local/index.json", "user-data")
    _write(tmp_path, "config/settings.json", "config")

    manifest = build_manifest(
        str(tmp_path),
        version="1.0.0",
        managed_roots=("app.exe", "_internal"),
    )

    paths = [item["path"] for item in manifest["files"]]
    assert paths == ["_internal/a.dll", "app.exe"]
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])
    assert manifest["version"] == "1.0.0"


def test_manifest_roundtrip_and_compare(tmp_path):
    _write(tmp_path, "app.exe", "v1")
    _write(tmp_path, "_internal/a.dll", "one")
    first = build_manifest(str(tmp_path), version="1.0.0", managed_roots=("app.exe", "_internal"))

    _write(tmp_path, "app.exe", "v2")
    (tmp_path / "_internal" / "a.dll").unlink()
    _write(tmp_path, "_internal/b.dll", "two")
    second = build_manifest(str(tmp_path), version="1.1.0", managed_roots=("app.exe", "_internal"))

    changed, deleted = compare_manifests(first, second)

    assert changed == {"app.exe", "_internal/b.dll"}
    assert deleted == {"_internal/a.dll"}

    path = tmp_path / "manifest.json"
    save_manifest(str(path), second)
    assert load_manifest(str(path))["version"] == "1.1.0"


def test_create_delta_zip_contains_only_changed_files(tmp_path):
    _write(tmp_path, "app.exe", "v1")
    _write(tmp_path, "_internal/a.dll", "one")
    old = build_manifest(str(tmp_path), version="1.0.0", managed_roots=("app.exe", "_internal"))

    _write(tmp_path, "app.exe", "v2")
    _write(tmp_path, "_internal/b.dll", "two")
    new = build_manifest(str(tmp_path), version="1.1.0", managed_roots=("app.exe", "_internal"))

    delta = tmp_path / "delta.zip"
    count, size = create_delta_zip(str(tmp_path), old, new, str(delta))

    assert count == 2
    assert size > 0
    with zipfile.ZipFile(delta) as zf:
        assert sorted(zf.namelist()) == ["_internal/b.dll", "app.exe"]


def test_validate_rejects_unsafe_paths():
    bad = {
        "schema": 1,
        "version": "1.0.0",
        "managed_roots": ["app.exe"],
        "files": [{"path": "../evil.exe", "size": 1, "sha256": "a" * 64}],
    }
    with pytest.raises(ValueError):
        validate_manifest(bad)


def test_is_newer_semver():
    assert is_newer("0.9.1", "0.9.0")
    assert is_newer("1.0", "0.99.9")
    assert not is_newer("0.9.0", "0.9.0")
    assert not is_newer("0.9.0", "0.10.0")


def test_manifest_json_is_utf8(tmp_path):
    _write(tmp_path, "app.exe", "v1")
    manifest = build_manifest(
        str(tmp_path),
        version="1.0.0",
        notes="Обновление интерфейса",
        managed_roots=("app.exe",),
    )
    path = tmp_path / "m.json"
    save_manifest(str(path), manifest)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["notes"] == "Обновление интерфейса"
