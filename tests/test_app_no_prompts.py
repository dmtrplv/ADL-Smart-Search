from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _prepare(tmp_path, monkeypatch):
    (tmp_path / "adl").mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setenv("ADL_DIR", str(tmp_path / "adl"))
    monkeypatch.setenv("ADL_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("ADL_INDEX_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ADL_DEV_MODE", "0")
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    monkeypatch.setenv("ADL_ENV_FILE", str(empty_env))
    # Нет prompts.json: приложение должно работать, ошибка — только при вызове LLM.
    monkeypatch.delenv("ADL_PROMPTS_FILE", raising=False)


def test_app_starts_without_prompts_file(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)

    at = AppTest.from_file(APP, default_timeout=120).run()

    assert not at.exception
    assert at.title[0].value == "ADL Smart Search"
    assert not (tmp_path / "config" / "prompts.json").exists()
