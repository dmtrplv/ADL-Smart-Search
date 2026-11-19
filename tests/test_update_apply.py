import json
import zipfile

import pytest

from adl_smart_search import update_apply
from adl_smart_search.update_apply import (
    apply_archive,
    choose_asset,
    download_file,
    extract_archive,
    run_job,
    verify_files,
)
from adl_smart_search.update_manifest import build_manifest, create_delta_zip, file_sha256


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

class _FakeResponse:
    status_code = 200

    def __init__(self, payload: bytes):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        return None

    def iter_content(self, size):
        yield self.payload


def _job(install, manifest_path, *, current="1.0.0", restart=None):
    return {
        "install_dir": str(install),
        "manifest_path": str(manifest_path),
        "current_version": current,
        "wait_pid": 0,
        "restart_exe": str(restart or install / "ADL-Smart-Search.exe"),
    }


def test_download_skips_matching_file(tmp_path, monkeypatch):
    dest = tmp_path / "download.zip"
    dest.write_bytes(b"same")
    expected = file_sha256(str(dest))
    monkeypatch.setattr(
        update_apply.requests,
        "get",
        lambda *a, **k: pytest.fail("download must not start"),
    )

    download_file("https://example.test/a.zip", str(dest), expected_sha256=expected)

    assert dest.read_bytes() == b"same"


def test_download_rejects_checksum_mismatch(tmp_path, monkeypatch):
    dest = tmp_path / "download.zip"
    monkeypatch.setattr(
        update_apply.requests, "get", lambda *a, **k: _FakeResponse(b"payload")
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        download_file("https://example.test/a.zip", str(dest), expected_sha256="0" * 64)

    assert not dest.exists()
    assert not (tmp_path / "download.zip.part").exists()


def test_run_job_downloads_delta_applies_and_restarts(tmp_path, monkeypatch):
    install = tmp_path / "install"
    _write(install, "app.exe", "old")
    new_root = tmp_path / "new"
    _write(new_root, "app.exe", "new")
    manifest = _manifest(new_root, "2.0.0")
    manifest["assets"] = {
        "deltas": {"1.0.0": {"url": "delta.zip", "sha256": "a" * 64}},
        "full": {"url": "full.zip", "sha256": "b" * 64},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(_job(install, manifest_path)), encoding="utf-8")

    downloaded, applied, launched = [], [], []
    monkeypatch.setattr(
        update_apply, "download_file", lambda url, dest, **kw: downloaded.append(url)
    )
    monkeypatch.setattr(
        update_apply, "apply_archive", lambda **kw: applied.append(kw["asset_kind"])
    )
    monkeypatch.setattr(
        update_apply.subprocess,
        "Popen",
        lambda args, cwd=None: launched.append((args, cwd)),
    )

    assert run_job(str(job_path)) == 0

    assert downloaded == ["delta.zip"]
    assert applied == ["delta"]
    assert not job_path.exists()
    assert launched == [([str(install / "ADL-Smart-Search.exe")], str(install))]


def test_run_job_falls_back_to_full_when_delta_fails(tmp_path, monkeypatch):
    install = tmp_path / "install"
    new_root = tmp_path / "new"
    _write(new_root, "app.exe", "new")
    manifest = _manifest(new_root, "2.0.0")
    manifest["assets"] = {
        "deltas": {"1.0.0": {"url": "delta.zip", "sha256": "a" * 64}},
        "full": {"url": "full.zip", "sha256": "b" * 64},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(_job(install, manifest_path)), encoding="utf-8")

    downloaded, applied = [], []

    def _apply(**kw):
        applied.append(kw["asset_kind"])
        if kw["asset_kind"] == "delta":
            raise ValueError("delta is broken")

    monkeypatch.setattr(
        update_apply, "download_file", lambda url, dest, **kw: downloaded.append(url)
    )
    monkeypatch.setattr(update_apply, "apply_archive", _apply)
    monkeypatch.setattr(update_apply.subprocess, "Popen", lambda *a, **k: None)

    assert run_job(str(job_path)) == 0

    assert downloaded == ["delta.zip", "full.zip"]
    assert applied == ["delta", "full"]


def test_run_job_reraises_delta_failure_without_full(tmp_path, monkeypatch):
    install = tmp_path / "install"
    new_root = tmp_path / "new"
    _write(new_root, "app.exe", "new")
    manifest = _manifest(new_root, "2.0.0")
    manifest["assets"] = {
        "deltas": {"1.0.0": {"url": "delta.zip", "sha256": "a" * 64}},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    job_path = tmp_path / "job.json"
    job_path.write_text(json.dumps(_job(install, manifest_path)), encoding="utf-8")

    monkeypatch.setattr(update_apply, "download_file", lambda *a, **k: None)

    def _apply(**kw):
        raise ValueError("delta is broken")

    monkeypatch.setattr(update_apply, "apply_archive", _apply)

    with pytest.raises(ValueError, match="delta is broken"):
        run_job(str(job_path))


def test_apply_archive_rejects_unknown_file(tmp_path):
    install = tmp_path / "install"
    new_root = tmp_path / "new"
    _write(new_root, "app.exe", "new")
    manifest = _manifest(new_root, "2.0.0")
    archive = tmp_path / "update.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("app.exe", "new")
        zf.writestr("extra.txt", "unexpected")

    with pytest.raises(ValueError, match="not in manifest"):
        apply_archive(
            install_dir=str(install),
            archive=str(archive),
            manifest=manifest,
            asset_kind="full",
            work_dir=str(tmp_path / "work"),
        )


def test_apply_archive_rejects_incomplete_full(tmp_path):
    install = tmp_path / "install"
    new_root = tmp_path / "new"
    _write(new_root, "app.exe", "new")
    _write(new_root, "_internal/a.dll", "dll")
    manifest = _manifest(new_root, "2.0.0")
    archive = tmp_path / "update.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("app.exe", "new")

    with pytest.raises(ValueError, match="incomplete"):
        apply_archive(
            install_dir=str(install),
            archive=str(archive),
            manifest=manifest,
            asset_kind="full",
            work_dir=str(tmp_path / "work"),
        )


def test_apply_archive_rejects_staging_checksum_mismatch(tmp_path):
    install = tmp_path / "install"
    new_root = tmp_path / "new"
    _write(new_root, "app.exe", "new")
    manifest = _manifest(new_root, "2.0.0")
    archive = tmp_path / "update.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("app.exe", "tampered")

    with pytest.raises(ValueError, match="checksum mismatch"):
        apply_archive(
            install_dir=str(install),
            archive=str(archive),
            manifest=manifest,
            asset_kind="full",
            work_dir=str(tmp_path / "work"),
        )


def test_apply_archive_restores_backup_when_verification_fails(tmp_path, monkeypatch):
    install = tmp_path / "install"
    _write(install, "app.exe", "old")
    new_root = tmp_path / "new"
    _write(new_root, "app.exe", "new")
    old = _manifest(install, "1.0.0")
    new = _manifest(new_root, "2.0.0")
    archive = tmp_path / "delta.zip"
    create_delta_zip(str(new_root), old, new, str(archive))
    monkeypatch.setattr(update_apply, "verify_files", lambda root, manifest: ["app.exe"])

    with pytest.raises(ValueError, match="failed verification"):
        apply_archive(
            install_dir=str(install),
            archive=str(archive),
            manifest=new,
            asset_kind="delta",
            work_dir=str(tmp_path / "work"),
        )

    assert (install / "app.exe").read_text(encoding="utf-8") == "old"
