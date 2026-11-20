from types import SimpleNamespace

from adl_smart_search.ui import sidebar


def test_folder_dialog_reads_cyrillic_path_as_utf8(monkeypatch):
    captured = {}

    def _run(args, **kwargs):
        captured["args"] = args
        captured.update(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout="C:\\Users\\urche\\OneDrive\\Рабочий стол\\Emias Templates\r\n",
        )

    monkeypatch.setattr(sidebar.subprocess, "run", _run)

    path = sidebar._run_folder_dialog("Выберите папку emiastemplates")

    assert path == r"C:\Users\urche\OneDrive\Рабочий стол\Emias Templates"
    assert captured["encoding"] == "utf-8"
    assert captured["errors"] == "replace"
    assert captured["text"] is True
    script = captured["args"][-1]
    assert "[Console]::OutputEncoding" in script


def test_folder_dialog_returns_empty_on_error(monkeypatch):
    monkeypatch.setattr(
        sidebar.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stdout=""),
    )

    assert sidebar._run_folder_dialog("Выберите папку") == ""
