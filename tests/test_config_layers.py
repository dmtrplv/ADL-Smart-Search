import json
from pathlib import Path

from adl_smart_search.ui.settings import (
    LLM_SETTINGS_TYPES,
    load_effective_llm_settings,
    load_effective_settings,
    load_user_llm_settings,
    load_user_settings,
    remove_user_llm_keys,
    reset_user_overrides,
    save_llm_settings,
    save_settings,
    save_user_llm_settings,
    save_user_settings,
    user_llm_settings_path,
    user_settings_path,
)


def _base_settings(config_dir):
    save_settings(
        str(config_dir),
        {"keyword_bonus": 0.5, "max_results": 10, "adl_dir": "", "ckm_adl_dir": ""},
    )


def test_effective_settings_merge_base_and_user(tmp_path):
    config_dir = tmp_path / "config"
    user_dir = tmp_path / "local" / "config"
    _base_settings(config_dir)
    Path(user_settings_path(str(user_dir))).parent.mkdir(parents=True)
    Path(user_settings_path(str(user_dir))).write_text(
        json.dumps({"keyword_bonus": 0.8}), encoding="utf-8"
    )

    merged = load_effective_settings(str(config_dir), str(user_dir))

    assert merged["keyword_bonus"] == 0.8
    assert merged["max_results"] == 10


def test_save_user_settings_stores_only_diff_and_removes_defaults(tmp_path):
    config_dir = tmp_path / "config"
    user_dir = tmp_path / "local" / "config"
    _base_settings(config_dir)

    save_user_settings(
        str(config_dir),
        str(user_dir),
        {"keyword_bonus": 0.7, "max_results": 10},
    )
    assert load_user_settings(str(user_dir)) == {"keyword_bonus": 0.7}

    save_user_settings(
        str(config_dir),
        str(user_dir),
        {"keyword_bonus": 0.5, "max_results": 10},
    )
    assert load_user_settings(str(user_dir)) == {}
    assert not Path(user_settings_path(str(user_dir))).exists()


def test_effective_llm_merge_and_secrets_not_persisted(tmp_path):
    config_dir = tmp_path / "config"
    user_dir = tmp_path / "local" / "config"
    save_llm_settings(str(config_dir), {"llm_mode": "api", "api_model": "base", "api_key": "secret"})
    save_user_llm_settings(
        str(config_dir),
        str(user_dir),
        {"api_model": "user", "api_key": "secret-2"},
    )

    merged = load_effective_llm_settings(str(config_dir), str(user_dir))

    assert merged["api_model"] == "user"
    assert "api_key" not in merged
    assert "api_key" not in load_user_llm_settings(str(user_dir))
    assert "api_key" not in LLM_SETTINGS_TYPES


def test_effective_llm_ignores_user_api_kind(tmp_path):
    config_dir = tmp_path / "config"
    user_dir = tmp_path / "local" / "config"
    save_llm_settings(str(config_dir), {"api_kind": "corp_async", "api_model": "base"})
    # Старый пользовательский выбор провайдера больше не действует.
    Path(user_llm_settings_path(str(user_dir))).parent.mkdir(parents=True, exist_ok=True)
    Path(user_llm_settings_path(str(user_dir))).write_text(
        '{"api_kind": "gigachat", "api_model": "user"}', encoding="utf-8"
    )

    merged = load_effective_llm_settings(str(config_dir), str(user_dir))

    assert merged["api_kind"] == "corp_async"
    assert merged["api_model"] == "user"


def test_remove_user_llm_keys_deletes_legacy_api_kind(tmp_path):
    user_dir = tmp_path / "local" / "config"
    Path(user_llm_settings_path(str(user_dir))).parent.mkdir(parents=True, exist_ok=True)
    Path(user_llm_settings_path(str(user_dir))).write_text(
        '{"api_kind": "gigachat", "api_model": "user"}', encoding="utf-8"
    )

    assert remove_user_llm_keys(str(user_dir), {"api_kind"}) is True
    assert load_user_llm_settings(str(user_dir)) == {"api_model": "user"}


def test_remove_user_llm_keys_removes_empty_file(tmp_path):
    user_dir = tmp_path / "local" / "config"
    Path(user_llm_settings_path(str(user_dir))).parent.mkdir(parents=True, exist_ok=True)
    Path(user_llm_settings_path(str(user_dir))).write_text(
        '{"api_kind": "gigachat"}', encoding="utf-8"
    )

    assert remove_user_llm_keys(str(user_dir), {"api_kind"}) is True
    assert not Path(user_llm_settings_path(str(user_dir))).exists()


def test_reset_user_overrides_makes_backup(tmp_path):
    user_dir = tmp_path / "local" / "config"
    Path(user_settings_path(str(user_dir))).parent.mkdir(parents=True)
    Path(user_settings_path(str(user_dir))).write_text('{"keyword_bonus": 0.9}', encoding="utf-8")
    Path(user_llm_settings_path(str(user_dir))).write_text('{"api_model": "m"}', encoding="utf-8")

    touched = reset_user_overrides(str(user_dir))

    assert len(touched) == 2
    assert not Path(user_settings_path(str(user_dir))).exists()
    assert not Path(user_llm_settings_path(str(user_dir))).exists()
    backups = list(user_dir.parent.glob("config/*.bak-*"))
    assert len(backups) == 2


def test_reset_only_search_keys_keeps_llm(tmp_path):
    user_dir = tmp_path / "local" / "config"
    Path(user_settings_path(str(user_dir))).parent.mkdir(parents=True)
    Path(user_settings_path(str(user_dir))).write_text(
        '{"keyword_bonus": 0.9, "adl_dir": "C:/adl"}', encoding="utf-8"
    )
    Path(user_llm_settings_path(str(user_dir))).write_text('{"api_model": "m"}', encoding="utf-8")

    reset_user_overrides(str(user_dir), settings_keys={"keyword_bonus"})

    assert load_user_settings(str(user_dir)) == {"adl_dir": "C:/adl"}
    assert Path(user_llm_settings_path(str(user_dir))).exists()
