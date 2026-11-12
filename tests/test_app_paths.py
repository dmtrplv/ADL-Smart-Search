from pathlib import Path

from adl_smart_search.app_paths import (
    AppPaths,
    bootstrap_data_root,
    bootstrap_repo_root,
    data_layout,
    data_root_from_repo,
    repo_root_from_data_root,
    resolve_app_paths,
    write_bootstrap,
)
from app import _needs_data_setup


def _make_data_root(root: Path, *, with_ckm: bool = False) -> Path:
    """Создаёт папку с config/ и data/."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "config").mkdir(exist_ok=True)
    (root / "config" / "settings.json").write_text("{}", encoding="utf-8")
    (root / "data" / "local").mkdir(parents=True, exist_ok=True)
    (root / "data" / "local" / "archetype_data.json").write_text("{}", encoding="utf-8")
    if with_ckm:
        (root / "data" / "ckm").mkdir(parents=True, exist_ok=True)
        (root / "data" / "ckm" / "archetype_data.json").write_text("{}", encoding="utf-8")
    return root


def _make_repo(repo_root: Path) -> Path:
    """Создаёт Git-репозиторий с штатной папкой Soft/ADL Smart Search."""
    return _make_data_root(repo_root / "Soft" / "ADL Smart Search")


def _clear_path_env(monkeypatch):
    for key in (
        "ADL_REPO_ROOT",
        "ADL_DATA_ROOT",
        "ADL_INDEX_DIR",
        "ADL_CONFIG_DIR",
        "ADL_USER_CONFIG_DIR",
        "ADL_LOCAL_DIR",
        "ADL_APP_MODE",
    ):
        monkeypatch.delenv(key, raising=False)


def test_data_layout_requires_config_and_indexes(tmp_path):
    data_root = _make_data_root(tmp_path / "data-root")

    assert data_layout(str(data_root)) == (str(data_root / "config"), str(data_root / "data"))
    (data_root / "data" / "local" / "archetype_data.json").unlink()
    assert data_layout(str(data_root)) is None


def test_data_root_from_repo_checks_emiastemplates_layout(tmp_path):
    parent = tmp_path / "emiastemplates"
    data_root = _make_repo(parent)

    assert data_root_from_repo(str(parent)) == str(data_root)
    assert data_root_from_repo(str(data_root)) == str(data_root)
    assert data_root_from_repo(str(tmp_path / "missing")) is None


def test_repo_root_from_data_root():
    root = Path(r"C:\emiastemplates")
    data_root = root / "Soft" / "ADL Smart Search"

    assert repo_root_from_data_root(str(data_root)) == str(root)


def test_bootstrap_roundtrip(tmp_path, monkeypatch):
    _clear_path_env(monkeypatch)
    monkeypatch.setenv("ADL_LOCAL_DIR", str(tmp_path / "local"))
    repo_root = tmp_path / "emiastemplates"
    data_root = _make_repo(repo_root)

    write_bootstrap(str(repo_root), str(data_root))

    assert bootstrap_repo_root() == str(repo_root)
    assert bootstrap_data_root() == str(data_root)
    assert (tmp_path / "local" / "config" / "bootstrap.json").is_file()


def test_bootstrap_reuses_repo_when_data_root_moved(tmp_path, monkeypatch):
    _clear_path_env(monkeypatch)
    local = tmp_path / "local"
    monkeypatch.setenv("ADL_LOCAL_DIR", str(local))
    repo_root = tmp_path / "emiastemplates"
    data_root = _make_repo(repo_root)
    write_bootstrap(str(repo_root), str(tmp_path / "missing-data-root"), str(local / "config"))

    paths = resolve_app_paths(str(tmp_path / "install"), mode="exe")

    assert paths.repo_root == str(repo_root)
    assert paths.data_root == str(data_root)
    assert paths.source == "bootstrap_repo"


def test_resolve_uses_explicit_repo_env(tmp_path, monkeypatch):
    _clear_path_env(monkeypatch)
    monkeypatch.setenv("ADL_LOCAL_DIR", str(tmp_path / "local"))
    repo_root = tmp_path / "emiastemplates"
    data_root = _make_repo(repo_root)
    monkeypatch.setenv("ADL_REPO_ROOT", str(repo_root))

    paths = resolve_app_paths(str(tmp_path / "install"))

    assert isinstance(paths, AppPaths)
    assert paths.repo_root == str(repo_root)
    assert paths.data_root == str(data_root)
    assert paths.config_dir == str(data_root / "config")
    assert paths.data_dir == str(data_root / "data")
    assert paths.source == "env_repo"


def test_resolve_uses_explicit_data_root_env(tmp_path, monkeypatch):
    _clear_path_env(monkeypatch)
    monkeypatch.setenv("ADL_LOCAL_DIR", str(tmp_path / "local"))
    repo_root = tmp_path / "emiastemplates"
    data_root = _make_repo(repo_root)
    monkeypatch.setenv("ADL_DATA_ROOT", str(data_root))

    paths = resolve_app_paths(str(tmp_path / "install"))

    assert paths.repo_root == str(repo_root)
    assert paths.data_root == str(data_root)
    assert paths.source == "env_data_root"


def test_resolve_uses_explicit_index_and_sibling_config(tmp_path, monkeypatch):
    _clear_path_env(monkeypatch)
    monkeypatch.setenv("ADL_LOCAL_DIR", str(tmp_path / "local"))
    data_root = _make_data_root(tmp_path / "data-root")
    monkeypatch.setenv("ADL_INDEX_DIR", str(data_root / "data"))
    monkeypatch.setenv("ADL_CONFIG_DIR", str(data_root / "config"))

    paths = resolve_app_paths(str(tmp_path / "install"), mode="exe")

    assert paths.data_root == str(data_root)
    assert paths.config_dir == str(data_root / "config")
    assert paths.data_dir == str(data_root / "data")
    assert paths.source == "env_index"


def test_resolve_uses_default_emiastemplates_path(tmp_path, monkeypatch):
    _clear_path_env(monkeypatch)
    monkeypatch.setenv("ADL_LOCAL_DIR", str(tmp_path / "local"))
    repo_root = tmp_path / "emiastemplates"
    data_root = _make_repo(repo_root)
    monkeypatch.setattr(
        "adl_smart_search.app_paths.DEFAULT_EMIASTEMPLATES_DIR",
        str(repo_root),
    )

    paths = resolve_app_paths(str(tmp_path / "install"), mode="exe")

    assert paths.repo_root == str(repo_root)
    assert paths.data_root == str(data_root)
    assert paths.source == "default_emiastemplates"


def test_resolve_does_not_search_other_folders(tmp_path, monkeypatch):
    _clear_path_env(monkeypatch)
    monkeypatch.setenv("ADL_LOCAL_DIR", str(tmp_path / "local"))
    monkeypatch.setattr(
        "adl_smart_search.app_paths.DEFAULT_EMIASTEMPLATES_DIR",
        str(tmp_path / "no-default"),
    )
    # Папка с данными существует, но приложение не должно её искать само.
    _make_data_root(tmp_path / "somewhere" / "adl-smart-search-data")

    paths = resolve_app_paths(str(tmp_path / "install"), mode="exe")

    assert paths.data_root == ""
    assert paths.source == "default"


def test_bootstrap_with_repo_only_derives_data_root(tmp_path, monkeypatch):
    _clear_path_env(monkeypatch)
    local = tmp_path / "local"
    monkeypatch.setenv("ADL_LOCAL_DIR", str(local))
    repo_root = tmp_path / "emiastemplates"
    data_root = _make_repo(repo_root)
    write_bootstrap(str(repo_root), "", str(local / "config"))

    paths = resolve_app_paths(str(tmp_path / "install"), mode="exe")

    assert paths.repo_root == str(repo_root)
    assert paths.data_root == str(data_root)
    assert paths.source == "bootstrap_repo"


def test_dev_mode_uses_project_paths(tmp_path, monkeypatch):
    _clear_path_env(monkeypatch)
    install = tmp_path / "project"
    data_root = _make_data_root(install)
    monkeypatch.setenv("ADL_LOCAL_DIR", str(install / ".test-local"))

    paths = resolve_app_paths(str(install), mode="dev")

    assert paths.mode == "dev"
    assert paths.repo_root == ""
    assert paths.data_root == str(install)
    assert paths.config_dir == str(data_root / "config")
    assert paths.data_dir == str(data_root / "data")
    assert paths.source == "dev_project"
    assert paths.local_dir == str(install / ".test-local")
    assert paths.user_config_dir == str(install / ".test-local" / "config")


def test_dev_mode_defaults_local_dir_to_project(tmp_path, monkeypatch):
    _clear_path_env(monkeypatch)
    install = tmp_path / "project"
    _make_data_root(install)

    paths = resolve_app_paths(str(install), mode="dev")

    assert paths.local_dir == str(install / ".adl-local")
    assert paths.user_config_dir == str(install / ".adl-local" / "config")
    assert paths.logs_dir == str(install / ".adl-local" / "logs")


def test_dev_mode_ignores_bootstrap_from_other_repo(tmp_path, monkeypatch):
    _clear_path_env(monkeypatch)
    install = tmp_path / "project"
    project_data_root = _make_data_root(install)
    other_repo = tmp_path / "other-repo"
    other_data_root = _make_repo(other_repo)
    local = tmp_path / "local"
    monkeypatch.setenv("ADL_LOCAL_DIR", str(local))
    write_bootstrap(str(other_repo), str(other_data_root), str(local / "config"))

    paths = resolve_app_paths(str(install), mode="dev")

    assert paths.data_root == str(install)
    assert paths.data_dir == str(project_data_root / "data")
    assert paths.config_dir == str(project_data_root / "config")
    assert paths.repo_root == ""


def test_exe_mode_uses_bootstrap_repo(tmp_path, monkeypatch):
    _clear_path_env(monkeypatch)
    repo_root = tmp_path / "emiastemplates"
    data_root = _make_repo(repo_root)
    local = tmp_path / "local"
    monkeypatch.setenv("ADL_LOCAL_DIR", str(local))
    write_bootstrap(str(repo_root), str(data_root), str(local / "config"))

    paths = resolve_app_paths(str(tmp_path / "install"), mode="exe")

    assert paths.mode == "exe"
    assert paths.repo_root == str(repo_root)
    assert paths.data_root == str(data_root)
    assert paths.config_dir == str(data_root / "config")
    assert paths.data_dir == str(data_root / "data")
    assert paths.source == "bootstrap"


def test_needs_data_setup_is_disabled_for_dev_mode(tmp_path):
    paths = AppPaths(
        install_dir=str(tmp_path),
        repo_root="",
        data_root=str(tmp_path),
        config_dir=str(tmp_path / "config"),
        data_dir=str(tmp_path / "data"),
        user_config_dir=str(tmp_path / ".adl-local" / "config"),
        logs_dir=str(tmp_path / ".adl-local" / "logs"),
        local_dir=str(tmp_path / ".adl-local"),
        mode="dev",
        source="dev_default",
    )

    assert _needs_data_setup(paths) is False


def test_needs_data_setup_for_empty_exe_mode(tmp_path):
    paths = AppPaths(
        install_dir=str(tmp_path),
        repo_root="",
        data_root="",
        config_dir=str(tmp_path / "config"),
        data_dir=str(tmp_path / "data"),
        user_config_dir=str(tmp_path / "local" / "config"),
        logs_dir=str(tmp_path / "local" / "logs"),
        local_dir=str(tmp_path / "local"),
        mode="exe",
        source="default",
    )

    assert _needs_data_setup(paths) is True
