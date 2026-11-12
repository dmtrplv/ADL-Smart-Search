import json
from types import SimpleNamespace

from adl_smart_search import rebuild_index
from adl_smart_search.config import AppConfig
from adl_smart_search.index_files import CKM_SOURCE, LOCAL_SOURCE


def _cfg(index_dir=""):
    cfg = AppConfig(adl_dir="")
    cfg.index_dir = index_dir
    return cfg


def test_resolve_paths_uses_settings(tmp_path):
    settings = {
        "adl_dir": str(tmp_path / "adl"),
        "ckm_adl_dir": str(tmp_path / "ckm"),
        "index_dir": str(tmp_path / "data"),
    }
    (tmp_path / "settings.json").write_text(json.dumps(settings), encoding="utf-8")

    args = SimpleNamespace(index_dir="", adl_dir="", ckm_adl_dir="")
    adl, ckm, root, loaded, settings_root = rebuild_index._resolve_paths(
        args, _cfg(index_dir=str(tmp_path))
    )

    assert adl == settings["adl_dir"]
    assert ckm == settings["ckm_adl_dir"]
    assert root == settings["index_dir"]
    assert loaded["adl_dir"] == settings["adl_dir"]
    assert settings_root == str(tmp_path)


def test_resolve_paths_prefers_args(tmp_path):
    args = SimpleNamespace(index_dir=str(tmp_path), adl_dir="X", ckm_adl_dir="Y")

    adl, ckm, root, _, _ = rebuild_index._resolve_paths(args, _cfg())

    assert (adl, ckm, root) == ("X", "Y", str(tmp_path))


def test_run_build_skips_missing_sources(capsys):
    rc = rebuild_index._run_build(
        adl_dir="",
        ckm_dir="",
        index_root="unused",
        embedder=None,
        preferred_lang="ru",
        sources=(LOCAL_SOURCE, CKM_SOURCE),
        kinds=("archetype",),
        full=False,
    )

    assert rc == 0
    assert "SKIP" in capsys.readouterr().out


def test_main_list(tmp_path, capsys):
    rc = rebuild_index.main(["--index-dir", str(tmp_path), "--list"])

    assert rc == 0
    assert "Корень индексов" in capsys.readouterr().out


def test_main_interactive_exit(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    rc = rebuild_index.main(["--interactive", "--index-dir", str(tmp_path)])

    assert rc == 0
