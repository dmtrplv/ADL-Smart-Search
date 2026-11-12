import os
from datetime import datetime

from adl_smart_search.index_build_info import (
    index_built_at,
    index_built_label,
    load_build_info,
    save_kind_build_info,
)
from adl_smart_search.ui.search_tab import _index_date


def _touch_index(root, source, kind):
    name = "archetype_data.json" if kind == "archetype" else "field_data.json"
    path = root / source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


def _local_date(iso_value: str) -> str:
    return datetime.fromtimestamp(
        datetime.fromisoformat(iso_value).timestamp()
    ).strftime("%d.%m.%Y")


def _local_datetime(iso_value: str) -> str:
    return datetime.fromtimestamp(
        datetime.fromisoformat(iso_value).timestamp()
    ).strftime("%d.%m.%Y %H:%M")


def test_build_info_keeps_different_dates_per_source_and_kind(tmp_path):
    _touch_index(tmp_path, "local", "archetype")
    _touch_index(tmp_path, "local", "field")
    _touch_index(tmp_path, "ckm", "archetype")
    _touch_index(tmp_path, "ckm", "field")

    save_kind_build_info(
        str(tmp_path),
        "local",
        "archetype",
        parser_version=2,
        index_format_version=1,
        embed_model="m",
        embed_dim=384,
        built_at="2026-10-05T12:00:00+03:00",
    )
    save_kind_build_info(
        str(tmp_path),
        "local",
        "field",
        parser_version=2,
        index_format_version=1,
        embed_model="m",
        embed_dim=384,
        built_at="2026-10-06T12:00:00+03:00",
    )
    save_kind_build_info(
        str(tmp_path),
        "ckm",
        "archetype",
        parser_version=2,
        index_format_version=1,
        embed_model="m",
        embed_dim=384,
        built_at="2026-10-07T12:00:00+03:00",
    )

    local_date = _index_date(str(tmp_path / "local"), "archetype")
    field_date = _index_date(str(tmp_path / "local"), "field")
    ckm_date = _index_date(str(tmp_path / "ckm"), "archetype")

    assert local_date == _local_date("2026-10-05T12:00:00+03:00")
    assert field_date == _local_date("2026-10-06T12:00:00+03:00")
    assert ckm_date == _local_date("2026-10-07T12:00:00+03:00")
    assert load_build_info(str(tmp_path))["local"]["archetype"]["embed_dim"] == 384


def test_build_info_falls_back_to_mtime(tmp_path):
    ts = datetime(2026, 10, 5, 12, 0).timestamp()
    path = _touch_index(tmp_path, "local", "archetype")
    os.utime(path, (ts, ts))

    assert _index_date(str(tmp_path / "local"), "archetype") == "05.10.2026"
    assert index_built_at(str(tmp_path), "local", "archetype") == ts


def test_index_built_label_uses_build_info_time(tmp_path):
    _touch_index(tmp_path, "local", "archetype")
    built_at = "2026-10-05T12:34:00+03:00"
    save_kind_build_info(
        str(tmp_path),
        "local",
        "archetype",
        parser_version=2,
        index_format_version=1,
        embed_model="m",
        embed_dim=384,
        built_at=built_at,
    )

    # Дата показывается в локальной timezone машины, а не в Москве.
    assert index_built_label(str(tmp_path), "local", "archetype") == _local_datetime(built_at)
