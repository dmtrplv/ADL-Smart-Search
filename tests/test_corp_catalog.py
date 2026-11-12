import json

from adl_smart_search.corp_catalog import (
    catalog_providers,
    corp_catalog_by_name,
    load_catalog_json,
    resolve_corp_catalog,
    resolve_corp_catalog_from_paths,
)


def test_resolve_returns_empty_when_no_path():
    assert resolve_corp_catalog("") == []
    assert resolve_corp_catalog("no/such/file.json") == []


def test_load_json_ignores_invalid_entries(tmp_path):
    p = tmp_path / "catalog.json"
    p.write_text(
        json.dumps(
            [
                {"model_name": "OK", "provider": "p1", "model": "/m", "model_kwargs": {}},
                {"provider": "no-name"},
                "junk",
                {"model_name": "", "provider": "empty-name"},
            ]
        ),
        encoding="utf-8",
    )
    loaded = load_catalog_json(str(p))
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0]["model_name"] == "OK"


def test_load_json_bad_file_returns_none():
    assert load_catalog_json("") is None
    assert load_catalog_json("missing.json") is None


def test_resolve_uses_external_file(tmp_path):
    p = tmp_path / "custom.json"
    p.write_text(
        json.dumps([{"model_name": "Custom", "provider": "p-x", "model": "m", "model_kwargs": {}}]),
        encoding="utf-8",
    )
    catalog = resolve_corp_catalog(str(p))
    assert [c["model_name"] for c in catalog] == ["Custom"]
    assert corp_catalog_by_name(catalog)["Custom"]["provider"] == "p-x"
    assert catalog_providers(catalog) == ["p-x"]


def test_resolve_from_paths_prefers_first_existing(tmp_path):
    legacy = tmp_path / "corp_catalog.json"
    legacy.write_text(
        json.dumps([{"model_name": "Legacy", "provider": "old"}]), encoding="utf-8"
    )
    preferred = tmp_path / "data" / "corp_catalog.json"
    preferred.parent.mkdir()
    preferred.write_text(
        json.dumps([{"model_name": "New", "provider": "new"}]), encoding="utf-8"
    )

    catalog = resolve_corp_catalog_from_paths(
        str(tmp_path / "missing.json"), str(preferred), str(legacy)
    )

    assert [c["model_name"] for c in catalog] == ["New"]


def test_resolve_from_paths_falls_back_to_legacy(tmp_path):
    legacy = tmp_path / "corp_catalog.json"
    legacy.write_text(
        json.dumps([{"model_name": "Legacy", "provider": "old"}]), encoding="utf-8"
    )

    catalog = resolve_corp_catalog_from_paths(
        str(tmp_path / "missing.json"), str(legacy)
    )

    assert [c["model_name"] for c in catalog] == ["Legacy"]


def test_resolve_from_paths_returns_empty_when_nothing_found(tmp_path):
    assert resolve_corp_catalog_from_paths(str(tmp_path / "a.json"), "") == []


def test_catalog_providers_deduplicates_and_orders():
    catalog = [
        {"model_name": "A", "provider": "p2"},
        {"model_name": "B", "provider": "p1"},
        {"model_name": "C", "provider": "p2"},
        {"model_name": "D", "provider": ""},
    ]
    assert catalog_providers(catalog) == ["p2", "p1"]