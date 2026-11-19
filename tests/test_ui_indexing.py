from adl_smart_search.ui.indexing import (
    delete_archetype_index_files,
    delete_field_index_files,
    maybe_move_legacy_indexes_to_data,
)


def test_legacy_indexes_move_into_local_source_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "archetype_data.json").write_text("{}", encoding="utf-8")
    (tmp_path / "field_vectors.npy").write_bytes(b"v")

    maybe_move_legacy_indexes_to_data(str(tmp_path / "data"))

    local = tmp_path / "data" / "local"
    assert (local / "archetype_data.json").read_text(encoding="utf-8") == "{}"
    assert (local / "field_vectors.npy").read_bytes() == b"v"
    assert not (tmp_path / "archetype_data.json").exists()


def test_legacy_migration_does_not_overwrite_existing_target(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "archetype_data.json").write_text("legacy", encoding="utf-8")
    local = tmp_path / "data" / "local"
    local.mkdir(parents=True)
    (local / "archetype_data.json").write_text("current", encoding="utf-8")

    maybe_move_legacy_indexes_to_data(str(tmp_path / "data"))

    assert (local / "archetype_data.json").read_text(encoding="utf-8") == "current"
    assert (tmp_path / "archetype_data.json").exists()


def test_delete_index_files_removes_only_requested_kind(tmp_path):
    (tmp_path / "archetype_data.json").write_text("{}", encoding="utf-8")
    (tmp_path / "archetype_vectors.npy").write_bytes(b"v")
    (tmp_path / "field_data.json").write_text("{}", encoding="utf-8")

    delete_archetype_index_files(str(tmp_path))

    assert not (tmp_path / "archetype_data.json").exists()
    assert not (tmp_path / "archetype_vectors.npy").exists()
    assert (tmp_path / "field_data.json").exists()


def test_delete_field_index_files_is_safe_when_missing(tmp_path):
    delete_field_index_files(str(tmp_path))
    (tmp_path / "field_data.json").write_text("{}", encoding="utf-8")

    delete_field_index_files(str(tmp_path))

    assert not (tmp_path / "field_data.json").exists()
