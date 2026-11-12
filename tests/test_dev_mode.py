import sys

from adl_smart_search.ui.settings import (
    DEV_MODE_MARKER,
    dev_mode_available,
    dev_mode_marker_path,
    is_dev_mode,
)


def test_dev_mode_available_when_running_from_python(monkeypatch):
    monkeypatch.delattr(sys, "frozen", raising=False)

    assert dev_mode_available() is True


def test_dev_mode_disabled_in_standard_exe(monkeypatch, tmp_path):
    exe = tmp_path / "ADL-Smart-Search.exe"
    exe.write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))
    monkeypatch.setenv("ADL_DEV_MODE", "1")

    assert dev_mode_marker_path() == str(tmp_path / DEV_MODE_MARKER)
    assert dev_mode_available() is False
    assert is_dev_mode() is False


def test_dev_mode_enabled_by_marker_in_admin_build(monkeypatch, tmp_path):
    exe = tmp_path / "ADL-Smart-Search.exe"
    exe.write_text("", encoding="utf-8")
    (tmp_path / DEV_MODE_MARKER).write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(exe))

    assert dev_mode_available() is True
