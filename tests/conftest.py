import os
import shutil
import sys
from pathlib import Path

import pytest

# Allow importing `adl_smart_search` without installing the package.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


_APP_PATH_ENV = (
    "ADL_INDEX_DIR",
    "ADL_CONFIG_DIR",
    "ADL_USER_CONFIG_DIR",
    "ADL_REPO_ROOT",
    "ADL_DATA_ROOT",
    "ADL_LOCAL_DIR",
    "ADL_LOG_DIR",
    "ADL_ENV_FILE",
    "ADL_PROMPTS_FILE",
    "ADL_APP_MODE",
)


@pytest.fixture(autouse=True)
def _restore_app_path_env(tmp_path):
    """Не даём путям, выставленным AppTest/app_paths, протекать между тестами."""
    saved = {key: os.environ.get(key) for key in _APP_PATH_ENV}
    prompts_copy = tmp_path / "app_prompts" / "prompts.json"
    prompts_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ROOT / "tests" / "fixtures" / "prompts.json", prompts_copy)
    os.environ.setdefault("ADL_PROMPTS_FILE", str(prompts_copy))
    yield
    for key, value in saved.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(autouse=True)
def _disable_embedder_warmup():
    """Не поднимаем фоновую загрузку модели в тестах (иначе поток живёт после pytest)."""
    from adl_smart_search import embedder

    embedder._WARMUP_DONE = True
    yield
    # Оставляем флаг включённым: следующие AppTest-тесты не должны грузить torch.
    embedder._WARMUP_DONE = True

