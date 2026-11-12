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


def _run_with_history(tmp_path, monkeypatch):
    _prepare(tmp_path, monkeypatch)
    at = AppTest.from_file(APP, default_timeout=120).run()

    # История уже есть, но ни один результат не выбран.
    at.session_state["arch_chat_messages"] = [
        {"role": "user", "content": "вопрос"},
        {"role": "assistant", "content": "ответ"},
    ]
    at.run()
    return at


def test_chat_input_is_disabled_without_selection(tmp_path, monkeypatch):
    at = _run_with_history(tmp_path, monkeypatch)

    assert len(at.chat_input) == 1
    assert at.chat_input[0].disabled is True


def test_selection_note_is_visible_before_sending(tmp_path, monkeypatch):
    at = _run_with_history(tmp_path, monkeypatch)

    assert any("Отметьте хотя бы один результат" in c.value for c in at.caption)
    # Примечание больше не появляется только как предупреждение после отправки.
    assert not any("Отметьте хотя бы один результат" in w.value for w in at.warning)


def test_reset_button_stays_enabled_without_selection(tmp_path, monkeypatch):
    at = _run_with_history(tmp_path, monkeypatch)

    reset = next(b for b in at.button if b.label == "Сбросить чат")
    assert reset.disabled is False
