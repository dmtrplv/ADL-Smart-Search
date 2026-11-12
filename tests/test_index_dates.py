import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from adl_smart_search.ui.search_tab import (
    _index_date,
    _index_date_for_source,
    _source_label_with_date,
)


@dataclass
class _Source:
    name: str
    index_dir: str


def _touch_index(index_dir: Path, kind: str, ts: float) -> Path:
    name = "archetype_data.json" if kind == "archetype" else "field_data.json"
    path = index_dir / name
    path.write_text("{}", encoding="utf-8")
    os.utime(path, (ts, ts))
    return path


def test_index_date_formats_only_the_date(tmp_path):
    ts = datetime(2026, 10, 5, 12, 0).timestamp()
    _touch_index(tmp_path, "archetype", ts)

    assert _index_date(str(tmp_path), "archetype") == "05.10.2026"


def test_index_date_is_empty_without_file(tmp_path):
    assert _index_date(str(tmp_path), "archetype") == ""
    assert _index_date(str(tmp_path), "field") == ""


def test_index_date_for_source_uses_its_own_index(tmp_path):
    local = tmp_path / "local"
    ckm = tmp_path / "ckm"
    local.mkdir()
    ckm.mkdir()
    _touch_index(local, "archetype", datetime(2026, 10, 5, 12, 0).timestamp())
    _touch_index(ckm, "archetype", datetime(2026, 10, 7, 12, 0).timestamp())
    sources = [_Source("local", str(local)), _Source("ckm", str(ckm))]

    assert _index_date_for_source(sources, "local", "archetype") == "05.10.2026"
    assert _index_date_for_source(sources, "ckm", "archetype") == "07.10.2026"
    # На вкладке «Поля» — дата файла полей, а не архетипов.
    assert _index_date_for_source(sources, "local", "field") == ""


def test_source_label_with_date_shows_date_under_label():
    assert _source_label_with_date("Локальная база", "05.10.2026") == (
        "Локальная база  \n05.10.2026"
    )


def test_source_label_without_date():
    assert _source_label_with_date("CKM", "") == "CKM"
