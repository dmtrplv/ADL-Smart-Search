import os

import pytest

from adl_smart_search.config import from_env
from adl_smart_search.env_file import (
    ENV_FILE_OVERRIDE,
    load_app_env,
    load_env_file,
    parse_env_text,
)


@pytest.fixture(autouse=True)
def _clean_env_override(monkeypatch):
    monkeypatch.delenv(ENV_FILE_OVERRIDE, raising=False)


def test_parse_handles_comments_quotes_and_empty_values():
    text = (
        "# комментарий\n"
        "\n"
        "LLM_MODE=api\n"
        'export LLM_API_KEY="key with spaces # and hash"\n'
        "LLM_CORP_PROVIDER='provider #1'\n"
        "EMPTY=\n"
        "SPACED = value with spaces # tail comment\n"
        "HASH=abc#def\n"
    )
    assert parse_env_text(text) == {
        "LLM_MODE": "api",
        "LLM_API_KEY": "key with spaces # and hash",
        "LLM_CORP_PROVIDER": "provider #1",
        "EMPTY": "",
        "SPACED": "value with spaces",
        "HASH": "abc#def",
    }


def test_parse_double_quote_escapes():
    text = 'V="C:\\temp\\new \\"q\\" end"\n'
    assert parse_env_text(text)["V"] == 'C:\\temp\\new "q" end'


def test_parse_skips_malformed_lines():
    assert parse_env_text("NOT A PAIR\nGOOD=1\n") == {"GOOD": "1"}


def test_parse_last_duplicate_wins():
    assert parse_env_text("A=1\nA=2\n") == {"A": "2"}


def test_load_env_file_keeps_existing_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("KEEP", "system")
    monkeypatch.delenv("FROM_FILE", raising=False)
    path = tmp_path / ".env"
    path.write_text("KEEP=file\nFROM_FILE=file\n", encoding="utf-8")

    applied = load_env_file(path)

    assert os.environ["KEEP"] == "system"
    assert os.environ["FROM_FILE"] == "file"
    assert applied == ["FROM_FILE"]


def test_load_env_file_override(tmp_path, monkeypatch):
    monkeypatch.setenv("KEEP", "system")
    path = tmp_path / ".env"
    path.write_text("KEEP=file\n", encoding="utf-8")

    load_env_file(path, override=True)

    assert os.environ["KEEP"] == "file"


def test_load_env_file_accepts_bom_and_cp1251(tmp_path, monkeypatch):
    monkeypatch.delenv("BOM_KEY", raising=False)
    monkeypatch.delenv("CP_KEY", raising=False)
    path = tmp_path / ".env"

    path.write_text("BOM_KEY=1\n", encoding="utf-8-sig")
    load_env_file(path)
    assert os.environ["BOM_KEY"] == "1"

    path.write_bytes("CP_KEY=2\n# комментарий\n".encode("cp1251"))
    load_env_file(path)
    assert os.environ["CP_KEY"] == "2"


def test_load_env_file_missing_is_noop(tmp_path):
    assert load_env_file(tmp_path / "nope.env") == []


def test_load_app_env_prefers_app_root(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("FROM_ROOT=1\n", encoding="utf-8")
    monkeypatch.delenv("FROM_ROOT", raising=False)

    loaded = load_app_env(project)

    assert loaded == (project / ".env").resolve()
    assert os.environ["FROM_ROOT"] == "1"
    assert os.environ[ENV_FILE_OVERRIDE] == str((project / ".env").resolve())


def test_load_app_env_uses_override_path(tmp_path, monkeypatch):
    custom = tmp_path / "custom.env"
    custom.write_text("FROM_CUSTOM=1\n", encoding="utf-8")
    monkeypatch.setenv(ENV_FILE_OVERRIDE, str(custom))
    monkeypatch.delenv("FROM_CUSTOM", raising=False)

    assert load_app_env(tmp_path) == custom.resolve()
    assert os.environ["FROM_CUSTOM"] == "1"


def test_load_app_env_falls_back_to_cwd(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / ".env").write_text("FROM_CWD=1\n", encoding="utf-8")
    monkeypatch.chdir(cwd)
    monkeypatch.delenv("FROM_CWD", raising=False)

    assert load_app_env(tmp_path / "missing") == (cwd / ".env").resolve()
    assert os.environ["FROM_CWD"] == "1"


def test_load_app_env_falls_back_when_override_missing(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV_FILE_OVERRIDE, str(tmp_path / "missing.env"))
    project = tmp_path / "project"
    project.mkdir()
    (project / ".env").write_text("FROM_FALLBACK=1\n", encoding="utf-8")
    monkeypatch.delenv("FROM_FALLBACK", raising=False)

    assert load_app_env(project) == (project / ".env").resolve()
    assert os.environ["FROM_FALLBACK"] == "1"


def test_load_app_env_missing_returns_none(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)

    assert load_app_env(tmp_path / "project") is None


def test_env_file_feeds_app_config(tmp_path, monkeypatch):
    for key in ("LLM_API_KEY", "GIGACHAT_CREDENTIALS"):
        monkeypatch.setenv(key, "system")
    path = tmp_path / ".env"
    path.write_text(
        "LLM_API_KEY=file-key\nGIGACHAT_CREDENTIALS=file-creds\nLLM_API_MODEL=ignored-model\n",
        encoding="utf-8",
    )

    load_env_file(path, override=True)
    cfg = from_env(default_adl_dir="")

    # Из .env берутся только секреты LLM.
    assert cfg.api_key == "file-key"
    assert cfg.gigachat_credentials == "file-creds"
    assert cfg.api_model == ""
