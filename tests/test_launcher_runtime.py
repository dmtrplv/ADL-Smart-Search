import json
import logging
import os
import threading
import time

import launcher


def test_idle_exit_minutes_default(monkeypatch):
    monkeypatch.delenv(launcher.IDLE_EXIT_ENV, raising=False)
    assert launcher._idle_exit_minutes() == launcher.DEFAULT_IDLE_EXIT_MINUTES


def test_idle_exit_minutes_env(monkeypatch):
    monkeypatch.setenv(launcher.IDLE_EXIT_ENV, "7.5")
    assert launcher._idle_exit_minutes() == 7.5
    monkeypatch.setenv(launcher.IDLE_EXIT_ENV, "0")
    assert launcher._idle_exit_minutes() == 0
    monkeypatch.setenv(launcher.IDLE_EXIT_ENV, "bad")
    assert launcher._idle_exit_minutes() == launcher.DEFAULT_IDLE_EXIT_MINUTES


def test_runtime_state_roundtrip(tmp_path):
    path = tmp_path / "runtime.json"
    launcher._write_runtime_state(str(path), 8532)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["pid"] == os.getpid()
    assert data["url"] == "http://127.0.0.1:8532"


def test_existing_instance_url_requires_alive_pid_and_health(tmp_path, monkeypatch):
    path = tmp_path / "runtime.json"
    path.write_text(
        json.dumps({"pid": 123, "url": "http://127.0.0.1:8532"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(launcher, "_pid_alive", lambda pid: False)
    assert launcher._existing_instance_url(str(path), retries=1, delay=0) is None

    monkeypatch.setattr(launcher, "_pid_alive", lambda pid: True)
    monkeypatch.setattr(launcher, "_health_ok", lambda url: False)
    assert launcher._existing_instance_url(str(path), retries=1, delay=0) is None

    monkeypatch.setattr(launcher, "_health_ok", lambda url: True)
    assert launcher._existing_instance_url(str(path), retries=1, delay=0) == "http://127.0.0.1:8532"


def test_existing_instance_url_ignores_missing_state(tmp_path):
    assert launcher._existing_instance_url(str(tmp_path / "missing.json"), retries=1, delay=0) is None


def test_idle_monitor_exits_when_no_browser_connects(tmp_path, monkeypatch):
    import streamlit.runtime.runtime as st_runtime

    class _Sessions:
        def num_active_sessions(self):
            return 0

    class _Runtime:
        _session_mgr = _Sessions()

    monkeypatch.setattr(st_runtime.Runtime, "instance", classmethod(lambda cls: _Runtime()))
    monkeypatch.setattr(launcher.logging, "shutdown", lambda: None)
    monkeypatch.setattr(launcher, "_remove_runtime_state", lambda path: None)

    fired = threading.Event()

    def _fake_exit(code):
        fired.set()
        # Не даём потоку продолжить цикл и не роняем pytest: daemon-поток
        # останется до конца процесса.
        while True:
            time.sleep(1)

    monkeypatch.setattr(launcher.os, "_exit", _fake_exit)

    launcher._start_idle_monitor(
        0.0005,
        str(tmp_path / "runtime.json"),
        logging.getLogger("test"),
        poll_s=0.005,
    )

    assert fired.wait(timeout=2.0)

