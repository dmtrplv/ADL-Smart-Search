import json

import adl_smart_search.update_client as client


def _manifest(version="1.0.1", notes="test"):
    return {
        "schema": 1,
        "version": version,
        "channel": "stable",
        "managed_roots": ["app.exe"],
        "files": [{"path": "app.exe", "size": 1, "sha256": "a" * 64}],
        "assets": {"full": {"url": "full.zip", "sha256": "b" * 64, "size": 1}},
        "notes": notes,
    }


def test_update_feed_url_env(monkeypatch):
    monkeypatch.setenv("ADL_UPDATE_FEED", "https://example.test/feed.json")
    assert client.update_feed_url() == "https://example.test/feed.json"


def test_check_for_update_returns_newer(monkeypatch, tmp_path):
    cache = tmp_path / "cache.json"
    monkeypatch.setattr(client, "update_cache_path", lambda: str(cache))
    monkeypatch.setattr(client, "fetch_manifest", lambda url, timeout=6.0: _manifest())
    monkeypatch.setenv("ADL_UPDATE_CHANNEL", "stable")

    info = client.check_for_update("1.0.0", force=True)

    assert info is not None
    assert info["manifest"]["version"] == "1.0.1"
    assert cache.exists()


def test_check_for_update_ignores_same_version(monkeypatch, tmp_path):
    cache = tmp_path / "cache.json"
    monkeypatch.setattr(client, "update_cache_path", lambda: str(cache))
    monkeypatch.setattr(client, "fetch_manifest", lambda url, timeout=6.0: _manifest("1.0.0"))

    assert client.check_for_update("1.0.0", force=True) is None


def test_check_for_update_uses_cache_on_failure(monkeypatch, tmp_path):
    cache = tmp_path / "cache.json"
    monkeypatch.setattr(client, "update_cache_path", lambda: str(cache))
    monkeypatch.setattr(client, "fetch_manifest", lambda url, timeout=6.0: _manifest())
    assert client.check_for_update("1.0.0", force=True) is not None

    def _fail(url, timeout=6.0):
        raise RuntimeError("offline")

    monkeypatch.setattr(client, "fetch_manifest", _fail)
    info = client.check_for_update("1.0.0", force=True)

    assert info is not None
    assert info["manifest"]["version"] == "1.0.1"


def test_failed_check_is_cached(tmp_path, monkeypatch):
    cache = tmp_path / "cache.json"
    monkeypatch.setattr(client, "update_cache_path", lambda: str(cache))
    calls = {"n": 0}

    def _fail(url, timeout=6.0):
        calls["n"] += 1
        raise RuntimeError("offline")

    monkeypatch.setattr(client, "fetch_manifest", _fail)

    assert client.check_for_update("1.0.0") is None
    assert client.check_for_update("1.0.0") is None
    assert calls["n"] == 1


def test_write_update_job(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "_updates_dir", lambda: str(tmp_path / "updates"))
    manifest = _manifest()

    job_path, manifest_path = client.write_update_job(
        manifest=manifest,
        feed_url="https://example.test/feed.json",
        install_dir=str(tmp_path / "install"),
        restart_exe=str(tmp_path / "install" / "ADL-Smart-Search.exe"),
        current_version="1.0.0",
        channel="stable",
    )

    assert manifest_path.endswith("manifest-1.0.1.json")
    job = json.loads(open(job_path, encoding="utf-8").read())
    assert job["current_version"] == "1.0.0"
    assert job["manifest_path"] == manifest_path
    assert json.loads(open(manifest_path, encoding="utf-8").read())["version"] == "1.0.1"


def test_launch_updater_requires_exe(tmp_path):
    try:
        client.launch_updater(str(tmp_path / "job.json"), str(tmp_path))
    except FileNotFoundError as e:
        assert "updater not found" in str(e)
    else:
        raise AssertionError("expected FileNotFoundError")
