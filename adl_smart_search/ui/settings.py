"""Чистые helpers настроек/индексов (без прямого рендеринга)."""

import argparse
import json
import logging
import os
import shutil
import sys
from datetime import datetime

import numpy as np

from adl_smart_search.config import DEFAULT_AUTO_STRICTNESS

logger = logging.getLogger(__name__)

LLM_SETTINGS_FILENAME = "llm.json"
# Секреты не сохраняем в llm.json: они живут в .env / окружении.
LLM_SECRET_KEYS = {"api_key", "openai_api_key", "gigachat_credentials"}


# Параметры поиска, которые сохраняются в settings.json из dev-панели.
SEARCH_SETTINGS_TYPES: dict[str, type] = {
    "auto_strictness": float,
    "keyword_bonus": float,
    "misuse_penalty": float,
    "weak_match_penalty": float,
    "enable_misuse_penalty": bool,
    "enable_weak_match_penalty": bool,
    "max_results": int,
    "context_tokens_archetypes": int,
    "context_tokens_fields": int,
    "sim_floor": float,
    "sim_floor_pure": float,
}

# Параметры LLM, которые сохраняются в llm.json из dev-панели.
LLM_SETTINGS_TYPES: dict[str, type] = {
    "llm_mode": str,
    "ollama_model": str,
    "ollama_base_url": str,
    "ollama_use_temperature": bool,
    "ollama_temperature": float,
    "ollama_use_top_p": bool,
    "ollama_top_p": float,
    "ollama_use_repeat_penalty": bool,
    "ollama_repeat_penalty": float,
    "ollama_use_num_predict": bool,
    "ollama_num_predict": int,
    "api_base_url": str,
    "api_model": str,
    "openai_base_url": str,
    "openai_model": str,
    "corp_provider": str,
    "corp_model_kwargs_json": str,
    "corp_poll_timeout_s": int,
    "corp_poll_interval_s": float,
    "corp_use_temperature": bool,
    "corp_temperature": float,
    "corp_use_max_tokens": bool,
    "corp_max_tokens": int,
    "corp_use_top_p": bool,
    "corp_top_p": float,
    "corp_use_top_k": bool,
    "corp_top_k": int,
    "corp_use_enable_thinking": bool,
    "corp_enable_thinking": bool,
    "corp_use_presence_penalty": bool,
    "corp_presence_penalty": float,
    "corp_use_repetition_penalty": bool,
    "corp_repetition_penalty": float,
    "corp_aux_use_temperature": bool,
    "corp_aux_temperature": float,
    "corp_aux_use_max_tokens": bool,
    "corp_aux_max_tokens": int,
    "corp_aux_use_top_p": bool,
    "corp_aux_top_p": float,
    "corp_aux_use_top_k": bool,
    "corp_aux_top_k": int,
    "corp_aux_use_enable_thinking": bool,
    "corp_aux_enable_thinking": bool,
    "corp_aux_use_presence_penalty": bool,
    "corp_aux_presence_penalty": float,
    "corp_aux_use_repetition_penalty": bool,
    "corp_aux_repetition_penalty": float,
    "gigachat_scope": str,
    "gigachat_model": str,
    "gigachat_base_url": str,
    "gigachat_ca_bundle_file": str,
    "gigachat_verify_ssl_certs": bool,
    "gigachat_timeout_s": float,
    "gigachat_temperature": float,
    "gigachat_max_tokens": int,
    "gigachat_context_window": int,
    # Отдельная модель для перевода и расширения запроса (dev).
    "llm_aux_model_enabled": bool,
    "ollama_aux_model": str,
    "api_aux_model": str,
    "openai_aux_model": str,
    "corp_aux_provider": str,
    "corp_aux_model_kwargs_json": str,
    "gigachat_aux_model": str,
}

# Только UI-состояние каталога моделей (в cfg таких полей нет).
LLM_UI_DEFAULTS = {
    "corp_model_name": "(вручную)",
    "corp_aux_model_name": "(вручную)",
    "corp_context_window": 8192,
}


def config_settings_path(config_dir: str) -> str:
    """Базовый settings.json из общего config/ (управляется администратором)."""
    return os.path.join(config_dir or ".", "settings.json")


def user_settings_path(user_config_dir: str) -> str:
    """Пользовательские переопределения поисковых настроек."""
    return os.path.join(user_config_dir or ".", "settings.overrides.json")


def settings_path(index_dir: str) -> str:
    """Совместимое имя для базового settings.json."""
    return config_settings_path(index_dir)


def _read_json_dict(path: str) -> dict:
    try:
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.exception("settings: cannot read %s", path)
        return {}


def _write_json_atomic(path: str, data: dict, *, private: bool = False) -> bool:
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        if private:
            try:
                os.chmod(tmp, 0o600)
            except OSError:
                pass
        os.replace(tmp, path)
        return True
    except Exception:
        logger.exception("settings: cannot write %s", path)
        return False


def load_settings(config_dir: str) -> dict:
    """Базовые настройки из ``config/settings.json``."""
    path = config_settings_path(config_dir)
    data = _read_json_dict(path)
    if not data:
        return {}
    raw_idx = str(data.get("index_dir") or "").strip()
    if raw_idx and not os.path.isabs(raw_idx):
        # Относительный путь из старых настроек: ориентируемся на текущий корень.
        data["index_dir"] = config_dir
    return data


def save_settings(config_dir: str, settings: dict) -> None:
    """Записывает базовый settings.json (используется админом/тестами)."""
    data = settings or {}
    try:
        if load_settings(config_dir) == data:
            return
        _write_json_atomic(config_settings_path(config_dir), data)
    except Exception:
        logger.exception("save_settings: cannot write settings")


def load_user_settings(user_config_dir: str) -> dict:
    """Только пользовательские переопределения (не базовый конфиг)."""
    if not user_config_dir:
        return {}
    return _read_json_dict(user_settings_path(user_config_dir))


def load_effective_settings(config_dir: str, user_config_dir: str = "") -> dict:
    """Базовые настройки + пользовательские overrides (overrides важнее)."""
    merged = load_settings(config_dir)
    merged.update(load_user_settings(user_config_dir))
    return merged


def save_user_settings(config_dir: str, user_config_dir: str, values: dict) -> bool:
    """Сохраняет только отличия от базового конфига.

    Если пользователь вернул базовое значение, соответствующий ключ удаляется
    из override-файла. Так будущие изменения базового config/ применяются ко
    всем, у кого нет явного переопределения.
    """
    if not user_config_dir:
        return False
    base = load_settings(config_dir)
    current = load_user_settings(user_config_dir)
    changed = dict(current)
    for key, value in (values or {}).items():
        if key in base and base.get(key) == value:
            changed.pop(key, None)
        else:
            changed[key] = value
    path = user_settings_path(user_config_dir)
    if not changed:
        try:
            if os.path.exists(path):
                os.remove(path)
            return True
        except OSError:
            logger.exception("save_user_settings: cannot remove %s", path)
            return False
    return _write_json_atomic(path, changed)


def config_llm_settings_path(config_dir: str) -> str:
    return os.path.join(config_dir or ".", LLM_SETTINGS_FILENAME)


def user_llm_settings_path(user_config_dir: str) -> str:
    return os.path.join(user_config_dir or ".", "llm.overrides.json")


def llm_settings_path(index_dir: str) -> str:
    """Совместимое имя для базового llm.json."""
    return config_llm_settings_path(index_dir)


def _without_secrets(values: dict) -> dict:
    return {k: v for k, v in (values or {}).items() if k not in LLM_SECRET_KEYS}


def load_llm_settings(config_dir: str) -> dict:
    """Базовые настройки LLM из ``config/llm.json``."""
    return _read_json_dict(config_llm_settings_path(config_dir))


def save_llm_settings(config_dir: str, values: dict) -> bool:
    """Записывает базовый llm.json (без секретов)."""
    data = _without_secrets(values)
    try:
        if load_llm_settings(config_dir) == data:
            return False
        return _write_json_atomic(config_llm_settings_path(config_dir), data, private=True)
    except Exception:
        logger.exception("save_llm_settings: cannot write llm settings")
        return False


def load_user_llm_settings(user_config_dir: str) -> dict:
    if not user_config_dir:
        return {}
    return _read_json_dict(user_llm_settings_path(user_config_dir))


def load_effective_llm_settings(config_dir: str, user_config_dir: str = "") -> dict:
    base = load_llm_settings(config_dir)
    user = load_user_llm_settings(user_config_dir)
    # Провайдер ИИ не сохраняется за пользователем: при запуске действует
    # значение из config/llm.json (по умолчанию МЭШ.ИИ.Хаб).
    user.pop("api_kind", None)
    merged = dict(base)
    merged.update(user)
    return merged


def remove_user_llm_keys(user_config_dir: str, keys: set[str]) -> bool:
    """Удаляет устаревшие ключи из пользовательского llm.overrides.json.

    Возвращает True, если файл был изменён. Если ключей не осталось,
    файл удаляется.
    """
    if not user_config_dir:
        return False
    current = load_user_llm_settings(user_config_dir)
    present = [key for key in keys if key in current]
    if not present:
        return False
    for key in present:
        current.pop(key, None)
    path = user_llm_settings_path(user_config_dir)
    if not current:
        try:
            if os.path.exists(path):
                os.remove(path)
            return True
        except OSError:
            logger.exception("remove_user_llm_keys: cannot remove %s", path)
            return False
    return _write_json_atomic(path, current, private=True)


def save_user_llm_settings(config_dir: str, user_config_dir: str, values: dict) -> bool:
    """Сохраняет только отличия LLM-настроек от базового llm.json."""
    if not user_config_dir:
        return False
    base = load_llm_settings(config_dir)
    current = load_user_llm_settings(user_config_dir)
    changed = dict(current)
    for key, value in _without_secrets(values).items():
        if key in base and base.get(key) == value:
            changed.pop(key, None)
        else:
            changed[key] = value
    path = user_llm_settings_path(user_config_dir)
    if not changed:
        try:
            if os.path.exists(path):
                os.remove(path)
            return True
        except OSError:
            logger.exception("save_user_llm_settings: cannot remove %s", path)
            return False
    return _write_json_atomic(path, changed, private=True)


def _backup_path(path: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{path}.bak-{stamp}"


def reset_user_overrides(
    user_config_dir: str,
    *,
    settings_keys: set[str] | None = None,
    llm_keys: set[str] | None = None,
    backup: bool = True,
) -> list[str]:
    """Удаляет пользовательские overrides (полностью или только указанные ключи).

    ``settings_keys=None`` и ``llm_keys=None`` — удалить оба файла целиком.
    Возвращает список затронутых файлов.
    """
    if not user_config_dir:
        return []
    touched: list[str] = []
    reset_all = settings_keys is None and llm_keys is None
    targets = (
        (user_settings_path(user_config_dir), settings_keys),
        (user_llm_settings_path(user_config_dir), llm_keys),
    )
    for path, keys in targets:
        if not reset_all and keys is None:
            continue
        if not os.path.exists(path):
            continue
        if keys is None:
            if backup:
                try:
                    os.replace(path, _backup_path(path))
                except OSError:
                    logger.exception("reset_user_overrides: cannot backup %s", path)
                    continue
            else:
                try:
                    os.remove(path)
                except OSError:
                    logger.exception("reset_user_overrides: cannot remove %s", path)
                    continue
            touched.append(path)
            continue

        if backup:
            try:
                shutil.copy2(path, _backup_path(path))
            except OSError:
                logger.exception("reset_user_overrides: cannot backup %s", path)
        data = _read_json_dict(path)
        for key in keys:
            data.pop(key, None)
        if data:
            _write_json_atomic(path, data)
        else:
            try:
                os.remove(path)
            except OSError:
                pass
        touched.append(path)
    return touched


def _coerce_value(kind: type, value):
    if kind is bool:
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)
    if kind is int:
        return int(value)
    if kind is float:
        return float(value)
    return str(value)


def apply_search_settings(cfg, settings: dict) -> None:
    """Переносит сохранённые параметры поиска в cfg (значения из файла важнее env)."""
    for key, kind in SEARCH_SETTINGS_TYPES.items():
        if key not in settings:
            continue
        try:
            setattr(cfg, key, _coerce_value(kind, settings[key]))
        except (TypeError, ValueError):
            logger.warning("apply_search_settings: bad value for %s: %r", key, settings[key])


def apply_llm_settings(cfg, values: dict) -> None:
    """Переносит сохранённые настройки LLM из data/llm.json в cfg.

    Секреты сюда не попадают: они читаются только из .env/окружения.
    """
    for key, kind in LLM_SETTINGS_TYPES.items():
        if key not in values:
            continue
        # Пустой путь/URL означает «автоопределение» (например, сертификат рядом с exe).
        if key in ("gigachat_ca_bundle_file", "gigachat_base_url") and not str(values[key]).strip():
            continue
        try:
            setattr(cfg, key, _coerce_value(kind, values[key]))
        except (TypeError, ValueError):
            logger.warning("apply_llm_settings: bad value for %s: %r", key, values[key])
    # Старый дефолт 800/4096 токенов слишком мал для длинных ответов. Пока лимит
    # выключен, значение ни на что не влияет; при включении лучше начинать с 8192.
    if not bool(getattr(cfg, "corp_use_max_tokens", False)):
        try:
            if int(getattr(cfg, "corp_max_tokens", 0) or 0) < 8192:
                cfg.corp_max_tokens = 8192
        except (TypeError, ValueError):
            pass



def index_last_built_label(index_dir: str, kind: str) -> str:
    """Дата/время сборки индекса: build_info.json, иначе mtime файла."""
    root = os.path.dirname(index_dir or ".")
    source = os.path.basename(index_dir or ".")
    from adl_smart_search.index_build_info import index_built_label

    label = index_built_label(root, source, kind)
    return label or "ещё не индексировалось"


DEV_MODE_MARKER = "dev_mode.enabled"


def dev_mode_marker_path() -> str | None:
    """Путь к маркеру dev-режима рядом с exe (для frozen-сборки)."""
    if getattr(sys, "frozen", False):
        return os.path.join(os.path.dirname(os.path.abspath(sys.executable)), DEV_MODE_MARKER)
    return None


def dev_mode_available() -> bool:
    """Dev-режим доступен при запуске из Python или в админской сборке с маркером."""
    if not getattr(sys, "frozen", False):
        return True
    marker = dev_mode_marker_path()
    return bool(marker and os.path.isfile(marker))


def is_dev_mode() -> bool:
    """
    Стартовое значение dev-режима: streamlit run app.py -- --dev
    или ADL_DEV_MODE=1/true/yes. Дальше режим переключается в интерфейсе.

    В стандартной exe-сборке dev-режим недоступен: нужен маркер
    ``dev_mode.enabled`` рядом с exe (админская сборка с ``-WithDevMode``).
    """
    if not dev_mode_available():
        return False
    env = (os.getenv("ADL_DEV_MODE") or "").strip().lower()
    if env in ("1", "true", "yes", "on"):
        return True
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dev", action="store_true")
    args, _ = parser.parse_known_args(sys.argv[1:])
    return bool(getattr(args, "dev", False))


def docs_signature(docs: list[dict] | None) -> tuple:
    """
    Небольшая сигнатура, чтобы понимать, что индекс изменился,
    и нужно пересобрать token_vocab (иначе короткие токены 2–3 могут не учитываться).
    """
    if not docs:
        return ()
    try:
        first = docs[0]
        last = docs[-1]
        return (
            len(docs),
            first.get("file"),
            first.get("sha1"),
            last.get("file"),
            last.get("sha1"),
        )
    except Exception:
        logger.exception("docs_signature: unexpected error")
        return (len(docs),)


def select_indices_by_relevance(
    scores_in,
    sims_in=None,
    lex_in=None,
    *,
    strictness: float = DEFAULT_AUTO_STRICTNESS,
    max_k: int = 10,
    min_k: int = 3,
    sources=None,
    sim_floor: float = 0.30,
    sim_floor_pure: float = 0.55,
    sim_floor_pure_by_idx=None,
    extend_k: int = 0,
    rank_within_group=None,
) -> tuple[list[int], dict]:
    """
    Отбор результатов по релевантности (per-source, затем слияние источников).

    Внутри одного источника:
      * документы сортируются по score по убыванию;
      * берём документы, пока разрыв с предыдущим не больше ``gap_max``
        и score не ниже ``top - band``;
      * документы со схожестью ниже порога отбрасываются. Порог разный для
        документов с буквальным совпадением (``sim_floor``) и для «чисто
        смысловых» (``sim_floor_pure``): буквальное совпадение — сильный
        признак, а чистая семантика требует более высокой схожести,
        иначе в выдачу попадает мусор;
      * ``sim_floor_pure_by_idx`` — необязательные персональные пороги для
        «чисто смысловых» документов (например, более низкий порог для
        кросс-языковых пар); по умолчанию используется ``sim_floor_pure``;
      * если набралось меньше ``min_k`` — добираем лучшими по score
        (тоже прошедшими порог).

    Между источниками: сравниваются абсолютные score, а z-score используется
    только как tie-breaker. Раньше порядок задавался только z, из-за чего
    документ с меньшим score мог обгонять более релевантный из другого источника.
    Итоговый список ограничен ``max_k``.

    ``extend_k`` — если > 0, в ``info["candidates"]`` дополнительно возвращается
    расширенный упорядоченный список (не больше extend_k): сначала выбранные
    (``picked``), затем следующие документы, прошедшие порог схожести, по score.
    Это «запас» для кнопки «Показать ещё»: он не влияет на основной отбор.

    ``rank_within_group(picked)`` — необязательный callback: возвращает
    ``(порядок, скорректированные score)`` для документов одного источника
    (например, предпочтение SIMI/версий). Влияет только на порядок внутри источника.

    strictness:
      0.0 — мягко (больше результатов)
      1.0 — строго (меньше результатов)
    """
    try:
        arr = np.asarray(scores_in, dtype=np.float32)
    except Exception:
        arr = np.array(scores_in, dtype=np.float32)
    if arr.size == 0:
        return [], {
            "groups": {},
            "gap_max": 0.0,
            "band": 0.0,
            "sim_floor": float(sim_floor),
            "sim_floor_pure": float(sim_floor_pure),
            "candidates": [],
        }

    sims = None
    if sims_in is not None:
        try:
            sims = np.asarray(sims_in, dtype=np.float32)
        except Exception:
            sims = None

    lex = None
    if lex_in is not None:
        try:
            lex = np.asarray(lex_in, dtype=np.float32)
        except Exception:
            lex = None

    pure_floors = None
    if sim_floor_pure_by_idx is not None:
        try:
            pure_floors = np.asarray(sim_floor_pure_by_idx, dtype=np.float32)
            if pure_floors.size != arr.size:
                pure_floors = None
        except Exception:
            pure_floors = None

    s = float(strictness)
    if not np.isfinite(s):
        s = DEFAULT_AUTO_STRICTNESS
    s = max(0.0, min(1.0, s))

    gap_max = 0.10 - 0.05 * s
    band = 0.15 - 0.07 * s
    floor = float(sim_floor)
    floor_pure = float(sim_floor_pure)

    def _sim_ok(i: int) -> bool:
        if sims is None:
            return True
        try:
            v = float(sims[i])
        except Exception:
            return True
        if not np.isfinite(v):
            return True
        has_lex = False
        if lex is not None:
            try:
                has_lex = float(lex[i]) > 0.0
            except Exception:
                has_lex = False
        if has_lex:
            return v >= floor
        pure = floor_pure
        if pure_floors is not None:
            try:
                pv = float(pure_floors[i])
                if np.isfinite(pv):
                    pure = pv
            except Exception:
                pass
        return v >= pure

    # Группировка по источнику.
    groups: dict[str, list[int]] = {}
    for i in range(arr.size):
        name = ""
        if sources is not None:
            try:
                name = str(sources[i] or "")
            except Exception:
                name = ""
        groups.setdefault(name, []).append(i)

    group_info: dict[str, dict] = {}
    group_stats: dict[str, tuple[float, float]] = {}
    zscores: dict[int, float] = {}
    rank_scores: dict[int, float] = {}

    for name, idxs in groups.items():
        order = sorted(idxs, key=lambda i: (-float(arr[i]), i))
        info = {
            "total": len(idxs),
            "top_score": float(arr[order[0]]) if order else 0.0,
            "top_sim": None,
            "picked": [],
            "reason": "",
        }

        chosen: list[int] = [order[0]] if order else []
        for prev, cur in zip(order, order[1:]):
            gap = float(arr[prev]) - float(arr[cur])
            if gap > gap_max:
                break
            if float(arr[cur]) < float(arr[order[0]]) - band:
                break
            chosen.append(cur)

        chosen = [i for i in chosen if _sim_ok(i)]
        if not chosen:
            info["reason"] = "below_floor"
            group_info[name] = info
            continue

        if len(chosen) < max(1, int(min_k)):
            for i in order:
                if i in chosen:
                    continue
                if not _sim_ok(i):
                    continue
                chosen.append(i)
                if len(chosen) >= max(1, int(min_k)):
                    break
            chosen = sorted(chosen, key=lambda i: (-float(arr[i]), i))

        group_scores = {i: float(arr[i]) for i in chosen}
        if rank_within_group is not None:
            try:
                res = rank_within_group(list(chosen))
                ordered = res[0] if isinstance(res, tuple) and len(res) == 2 else res
                if isinstance(res, tuple) and len(res) == 2 and res[1]:
                    for i in chosen:
                        if i in res[1]:
                            group_scores[i] = float(res[1][i])
                if ordered:
                    seen: set[int] = set()
                    picked_ordered = []
                    for i in ordered:
                        if i in chosen and i not in seen:
                            picked_ordered.append(i)
                            seen.add(i)
                    for i in chosen:
                        if i not in seen:
                            picked_ordered.append(i)
                    chosen = picked_ordered
            except Exception:
                logger.warning("select_indices_by_relevance: rank_within_group failed", exc_info=True)

        vals = np.asarray([group_scores[i] for i in chosen], dtype=np.float32)
        mean = float(vals.mean()) if vals.size else 0.0
        std = float(vals.std()) if vals.size else 0.0
        group_stats[name] = (mean, std)
        for i in chosen:
            rank_scores[i] = float(group_scores[i])
            zscores[i] = (group_scores[i] - mean) / std if std > 1e-9 else 0.0

        if extend_k > 0:
            # Запасные кандидаты для «Показать ещё»: всё, что проходит порог, по score.
            info["ranked"] = [i for i in order if _sim_ok(i)]

        top_sim = None
        if sims is not None:
            try:
                top_sim = float(max(float(sims[i]) for i in chosen))
            except Exception:
                top_sim = None
        info["top_sim"] = top_sim
        info["picked"] = chosen
        info["reason"] = "ok" if len(chosen) > 1 else "single"
        group_info[name] = info

    merged: list[tuple[float, float, int, int]] = []
    for _name, info in group_info.items():
        for pos, i in enumerate(info["picked"]):
            merged.append(
                (float(rank_scores.get(i, arr[i])), zscores.get(i, 0.0), pos, i)
            )
    merged.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))

    limit = max(1, int(max_k))
    picked = [i for _, _, _, i in merged][:limit]

    candidates = picked
    if extend_k > 0:
        cap = max(1, int(extend_k))
        chosen_set = set(picked)
        extra: list[tuple[float, float, int, int]] = []
        for name, info in group_info.items():
            mean, std = group_stats.get(name, (0.0, 0.0))
            for pos, i in enumerate(info.get("ranked") or []):
                if i in chosen_set:
                    continue
                z = (float(arr[i]) - mean) / std if std > 1e-9 else 0.0
                zscores[i] = z
                extra.append((float(arr[i]), z, pos, i))
        extra.sort(key=lambda t: (-t[0], -t[1], t[2], t[3]))
        candidates = (picked + [i for _, _, _, i in extra])[:cap]

    return picked, {
        "groups": group_info,
        "gap_max": gap_max,
        "band": band,
        "sim_floor": floor,
        "sim_floor_pure": floor_pure,
        "zscores": zscores,
        "candidates": candidates,
    }


