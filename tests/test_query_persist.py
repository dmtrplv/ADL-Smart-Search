from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _prepare(tmp_path, monkeypatch):
    (tmp_path / "adl").mkdir()
    monkeypatch.setenv("ADL_DIR", str(tmp_path / "adl"))
    monkeypatch.setenv("ADL_INDEX_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("ADL_DEV_MODE", "0")
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    monkeypatch.setenv("ADL_ENV_FILE", str(empty_env))


def test_query_text_survives_tab_switch(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    at = AppTest.from_file(APP, default_timeout=120).run()

    at.text_input(key="arch_query").set_value("одышка")
    next(b for b in at.button if b.label == "Искать").click().run()

    at.session_state["active_view"] = "Поля"
    at.run()
    at.session_state["active_view"] = "Архетипы"
    at.run()

    assert at.text_input(key="arch_query").value == "одышка"
