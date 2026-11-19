import json
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def _make_repo(repo_root: Path) -> Path:
    data_root = repo_root / "Soft" / "ADL Smart Search"
    (data_root / "config").mkdir(parents=True)
    (data_root / "config" / "settings.json").write_text("{}", encoding="utf-8")
    local = data_root / "data" / "local"
    local.mkdir(parents=True)
    (local / "archetype_data.json").write_text("{}", encoding="utf-8")
    return data_root


def test_data_setup_connects_repo_and_writes_bootstrap(tmp_path, monkeypatch):
    repo_root = tmp_path / "emiastemplates"
    _make_repo(repo_root)
    local = tmp_path / "local"
    empty_env = tmp_path / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    monkeypatch.setenv("ADL_APP_MODE", "exe")
    monkeypatch.setenv("ADL_LOCAL_DIR", str(local))
    monkeypatch.setenv("ADL_ENV_FILE", str(empty_env))
    monkeypatch.setattr(
        "adl_smart_search.app_paths.DEFAULT_EMIASTEMPLATES_DIR",
        str(tmp_path / "no-default"),
    )
    for key in ("ADL_REPO_ROOT", "ADL_DATA_ROOT", "ADL_INDEX_DIR", "ADL_CONFIG_DIR"):
        monkeypatch.delenv(key, raising=False)

    at = AppTest.from_file(APP, default_timeout=120).run()
    assert "Не найдена папка с данными" in " ".join(e.value for e in at.error)

    at.text_input(key="_data_setup_manual").set_value(str(repo_root)).run()
    next(b for b in at.button if b.label == "Подключить").click().run()

    bootstrap = json.loads(
        (local / "config" / "bootstrap.json").read_text(encoding="utf-8")
    )
    assert bootstrap["repo_root"] == str(repo_root)
    assert "data_root" not in bootstrap
    assert at.title[0].value == "ADL Smart Search"
