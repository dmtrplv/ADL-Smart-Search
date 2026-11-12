import index_launcher


def test_find_main_exe_prefers_fixed_name(tmp_path):
    main = tmp_path / "ADL-Smart-Search.exe"
    main.write_text("", encoding="utf-8")
    (tmp_path / "ADL-Smart-Search-Index.exe").write_text("", encoding="utf-8")

    assert index_launcher._find_main_exe(str(tmp_path)) == str(main)


def test_find_main_exe_falls_back_to_timestamped_name(tmp_path):
    main = tmp_path / "ADL-Smart-Search-20260101-120000.exe"
    main.write_text("", encoding="utf-8")
    (tmp_path / "ADL-Smart-Search-Index.exe").write_text("", encoding="utf-8")

    assert index_launcher._find_main_exe(str(tmp_path)) == str(main)


def test_find_main_exe_ignores_indexer(tmp_path):
    (tmp_path / "ADL-Smart-Search-Index.exe").write_text("", encoding="utf-8")

    assert index_launcher._find_main_exe(str(tmp_path)) is None


def test_find_main_exe_ignores_updater(tmp_path):
    (tmp_path / "ADL-Smart-Search-Updater.exe").write_text("", encoding="utf-8")

    assert index_launcher._find_main_exe(str(tmp_path)) is None
