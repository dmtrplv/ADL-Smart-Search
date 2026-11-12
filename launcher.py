import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

from adl_smart_search.app_paths import (
    CONFIG_DIR_ENV,
    DATA_ROOT_ENV,
    LOCAL_DIR_ENV,
    REPO_ROOT_ENV,
    USER_CONFIG_DIR_ENV,
    apply_paths_env,
    resolve_app_paths,
)
from adl_smart_search.env_file import load_app_envs
from adl_smart_search.logging_setup import setup_logging


def _resource_root() -> str:
    """
    Возвращает папку, где лежат ресурсы рядом с exe (или рядом со скриптом в dev-режиме).
    PyInstaller (onedir) устанавливает sys._MEIPASS на временную папку с распакованными ресурсами,
    но рядом с exe обычно лежат user-данные (data/models). Мы ориентируемся на папку exe.
    """
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


class _Tee:
    """Пишет одновременно в консоль и в лог (для режима --index)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def _attach_console() -> None:
    """Windows: подключить консоль для режима --index у --noconsole exe."""
    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        ATTACH_PARENT_PROCESS = -1
        if not kernel32.AttachConsole(ATTACH_PARENT_PROCESS):
            kernel32.AllocConsole()
        # Консоль и лог — в UTF-8, чтобы русский текст не превращался в кракозябры.
        kernel32.SetConsoleOutputCP(65001)
        kernel32.SetConsoleCP(65001)
        sys.stdout = open("CONOUT$", "w", encoding="utf-8", buffering=1)
        sys.stderr = open("CONOUT$", "w", encoding="utf-8", buffering=1)
        sys.stdin = open("CONIN$", "r", encoding="utf-8")
        # Дублируем меню и прогресс в лог: удобно, если консоль не видна.
        log_dir = os.environ.get("ADL_LOG_DIR", "logs")
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_stream = open(
                os.path.join(log_dir, "indexer_console.log"),
                "a",
                encoding="utf-8",
                buffering=1,
            )
            sys.stdout = _Tee(sys.stdout, log_stream)
        except Exception:
            pass
    except Exception:
        # Если консоль подключить не удалось, индексатор всё равно запустится;
        # диагностика останется в logs/app.log.
        pass


def _pick_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _wait_for_port(host: str, port: int, timeout_s: float = 15.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            s.connect((host, port))
            s.close()
            return True
        except Exception:
            time.sleep(0.2)
    return False


RUNTIME_STATE_FILENAME = "runtime.json"
IDLE_EXIT_ENV = "ADL_IDLE_EXIT_MINUTES"
DEFAULT_IDLE_EXIT_MINUTES = 15.0


def _idle_exit_minutes() -> float:
    raw = (os.environ.get(IDLE_EXIT_ENV) or "").strip()
    if not raw:
        return DEFAULT_IDLE_EXIT_MINUTES
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_IDLE_EXIT_MINUTES
    return max(0.0, value)


def _runtime_state_path(local_dir: str) -> str:
    return os.path.join(local_dir, RUNTIME_STATE_FILENAME)


def _read_runtime_state(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_runtime_state(path: str, port: int) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "pid": os.getpid(),
        "port": int(port),
        "url": f"http://127.0.0.1:{int(port)}",
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def _remove_runtime_state(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            STILL_ACTIVE = 259
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
            if not handle:
                return False
            try:
                code = ctypes.c_ulong()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                    return False
                return code.value == STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _health_ok(url: str, timeout: float = 1.0) -> bool:
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f"{url.rstrip('/')}/_stcore/health", timeout=timeout) as response:
            return int(getattr(response, "status", 200)) == 200
    except Exception:
        return False


def _existing_instance_url(path: str, *, retries: int = 40, delay: float = 0.5) -> str | None:
    """URL уже запущенного экземпляра, если он жив и отвечает на health-check."""
    state = _read_runtime_state(path)
    try:
        pid = int(state.get("pid") or 0)
    except (TypeError, ValueError):
        return None
    url = str(state.get("url") or "").strip()
    if not pid or not url or not _pid_alive(pid):
        return None
    for _ in range(max(1, retries)):
        if _health_ok(url):
            return url
        time.sleep(delay)
    return None


def _try_open_existing_instance(path: str, logger: logging.Logger) -> bool:
    url = _existing_instance_url(path)
    if not url:
        return False
    logger.info("launcher: already running at %s, opening browser", url)
    try:
        webbrowser.open(url)
    except Exception:
        logger.exception("launcher: cannot open existing instance URL")
    return True


def _start_idle_monitor(
    idle_minutes: float,
    runtime_file: str,
    logger: logging.Logger,
    *,
    poll_s: float = 15.0,
) -> None:
    """Завершает процесс, если браузерных сессий нет дольше idle_minutes."""
    if idle_minutes <= 0:
        return

    def _shutdown(reason: str) -> None:
        logger.info("idle shutdown: %s", reason)
        _remove_runtime_state(runtime_file)
        logging.shutdown()
        os._exit(0)

    def run() -> None:
        from streamlit.runtime.runtime import Runtime

        timeout_s = idle_minutes * 60.0
        started = time.monotonic()
        seen_active = False
        idle_since: float | None = None
        while True:
            time.sleep(max(0.001, float(poll_s)))
            try:
                runtime = Runtime.instance()
            except RuntimeError:
                if not seen_active and time.monotonic() - started >= timeout_s:
                    _shutdown("браузер так и не подключился")
                continue
            try:
                active = int(runtime._session_mgr.num_active_sessions())  # type: ignore[attr-defined]
            except Exception:
                logger.warning("idle monitor: session manager unavailable; auto-exit disabled")
                return
            now = time.monotonic()
            if active > 0:
                seen_active = True
                idle_since = None
                continue
            if not seen_active:
                if now - started >= timeout_s:
                    _shutdown("браузер так и не подключился")
                continue
            if idle_since is None:
                idle_since = now
            elif now - idle_since >= timeout_s:
                _shutdown(f"нет активных сессий {idle_minutes:g} мин")

    threading.Thread(target=run, daemon=True, name="adl-idle-monitor").start()


def _run_streamlit_in_process(
    app_py: str,
    port: int,
    *,
    idle_minutes: float = 0.0,
    runtime_file: str = "",
) -> int:
    """
    В frozen-режиме sys.executable указывает на launcher.exe, поэтому нельзя делать
    subprocess: получится рекурсивный запуск самого себя. Запускаем Streamlit в-process.
    """
    import streamlit.web.cli as stcli  # импортируем лениво

    if runtime_file:
        _start_idle_monitor(idle_minutes, runtime_file, logging.getLogger("launcher"))

    # In no-console mode we still want a persistent log file.
    log_path = os.environ.get("ADL_STREAMLIT_LOG", "")
    if not log_path:
        log_dir = os.environ.get("ADL_LOG_DIR") or os.path.join(_resource_root(), "logs")
        log_path = os.path.join(log_dir, "streamlit.log")
        os.environ["ADL_STREAMLIT_LOG"] = log_path

    try:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log_f = open(log_path, "a", encoding="utf-8")
    except Exception:
        log_f = None

    argv = [
        "streamlit",
        "run",
        app_py,
        "--global.developmentMode",
        "false",
        # В exe скрываем Deploy, но оставляем меню Streamlit с выбором темы.
        "--client.toolbarMode",
        "viewer",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        str(port),
        "--server.headless",
        "true",  # браузер открываем сами, один раз
        "--browser.gatherUsageStats",
        "false",
    ]

    old_argv = sys.argv[:]
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.argv = argv
        if log_f is not None:
            sys.stdout = log_f  # type: ignore[assignment]
            sys.stderr = log_f  # type: ignore[assignment]
        stcli.main()
        return 0
    finally:
        sys.argv = old_argv
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        try:
            if log_f is not None:
                log_f.flush()
                log_f.close()
        except Exception:
            pass


def main() -> int:
    root = _resource_root()
    paths = resolve_app_paths(root)
    env_path = load_app_envs(
        [
            os.path.join(paths.user_config_dir, ".env"),
            os.path.join(paths.config_dir, ".env"),
            os.path.join(root, ".env"),
        ]
    )

    # Модели остаются частью установки; данные и конфиги — внешние.
    models_dir = os.path.join(root, "models")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(paths.logs_dir, exist_ok=True)
    os.environ.setdefault("ADL_LOG_DIR", paths.logs_dir)

    if paths.has_data:
        apply_paths_env(paths)
    else:
        # Нет подключённого репозитория и нет старой data/: не задаём ADL_INDEX_DIR,
        # чтобы приложение показало экран подключения данных.
        os.environ.pop("ADL_INDEX_DIR", None)
        os.environ[REPO_ROOT_ENV] = paths.repo_root
        os.environ[DATA_ROOT_ENV] = paths.data_root
        os.environ[CONFIG_DIR_ENV] = paths.config_dir
        os.environ[USER_CONFIG_DIR_ENV] = paths.user_config_dir
        os.environ[LOCAL_DIR_ENV] = paths.local_dir

    setup_logging()
    logger = logging.getLogger("launcher")
    logger.info("launcher start: root=%s frozen=%s", root, getattr(sys, "frozen", False))
    logger.info("launcher: env file=%s", env_path or "не найден")
    logger.info(
        "launcher: repo=%s data_root=%s data=%s config=%s",
        paths.repo_root,
        paths.data_root,
        paths.data_dir,
        paths.config_dir,
    )

    # В frozen-режиме работаем из папки рядом с exe: относительные пути (например, "./data"
    # в settings.json) должны резолвиться именно оттуда.
    if getattr(sys, "frozen", False):
        try:
            os.chdir(root)
        except Exception:
            logger.warning("launcher: cannot chdir to %s", root)

    # sentence-transformers / HF cache
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = models_dir
    os.environ["HF_HOME"] = models_dir
    # Project-style layout: cached model folders live at <models_dir>\models--org--repo\...
    os.environ["HF_HUB_CACHE"] = models_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(models_dir, "transformers")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    # Жёстко офлайн: если модели уже лежат в models_dir, Streamlit/индексация не будут ходить в сеть.
    # Если нужно "первый запуск скачает" — уберите эти переменные.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # У некоторых пользователей/сборочных окружений эта настройка может быть включена глобально,
    # и тогда Streamlit запрещает задавать server.port. Нас это ломает, поэтому принудительно выключаем.
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"

    # Write basic diagnostics for frozen/no-console runs.
    try:
        env_lines = [
            f"python={sys.version}",
            f"executable={sys.executable}",
            f"frozen={getattr(sys, 'frozen', False)}",
            f"root={root}",
            f"data_dir={paths.data_dir}",
            f"config_dir={paths.config_dir}",
            f"repo_root={paths.repo_root}",
            f"data_root={paths.data_root}",
            f"models_dir={models_dir}",
            f"ADL_INDEX_DIR={os.environ.get('ADL_INDEX_DIR','')}",
            f"SENTENCE_TRANSFORMERS_HOME={os.environ.get('SENTENCE_TRANSFORMERS_HOME','')}",
            f"HF_HOME={os.environ.get('HF_HOME','')}",
            f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE','')}",
            f"TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE','')}",
            f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE','')}",
            f"TRANSFORMERS_OFFLINE={os.environ.get('TRANSFORMERS_OFFLINE','')}",
        ]
        # Probe expected model file (best-effort)
        safes = list(Path(models_dir).glob("models--*/*/model.safetensors"))
        if safes:
            p = safes[0]
            env_lines.append(f"model_safetensors={p}")
            env_lines.append(f"model_safetensors_bytes={p.stat().st_size}")
        Path(paths.logs_dir, "runtime_env.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    except Exception:
        pass

    # Режим индексации: работаем в том же onedir-комплекте библиотек, что и
    # приложение, без второго torch/sentence-transformers и без распаковки.
    if "--index" in sys.argv[1:]:
        if getattr(sys, "frozen", False):
            _attach_console()
        from adl_smart_search.rebuild_index import main as rebuild_main

        index_args = [a for a in sys.argv[1:] if a != "--index"]
        return rebuild_main(index_args)

    is_frozen = getattr(sys, "frozen", False)
    runtime_file = _runtime_state_path(paths.local_dir)
    idle_minutes = _idle_exit_minutes()

    if is_frozen and _try_open_existing_instance(runtime_file, logger):
        return 0

    port = _pick_free_port()
    url = f"http://127.0.0.1:{port}"
    if is_frozen:
        try:
            _write_runtime_state(runtime_file, port)
            os.environ["ADL_RUNTIME_FILE"] = runtime_file
        except Exception:
            logger.exception("launcher: cannot write runtime state")

    # app.py:
    # - в dev-режиме лежит рядом с launcher.py
    # - в собранном onedir (PyInstaller) попадает во внутреннюю папку (_internal),
    #   доступную через sys._MEIPASS
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        app_py = os.path.join(sys._MEIPASS, "app.py")  # type: ignore[attr-defined]
    else:
        app_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    # На всякий случай — старое расположение рядом с exe (если кто-то захочет держать app.py снаружи)
    if not os.path.exists(app_py):
        app_py = os.path.join(root, "app.py")

    def _open_browser_once() -> None:
        if _wait_for_port("127.0.0.1", port, timeout_s=20.0):
            webbrowser.open(url)

    threading.Thread(target=_open_browser_once, daemon=True).start()

    try:
        if is_frozen:
            return _run_streamlit_in_process(
                app_py,
                port,
                idle_minutes=idle_minutes,
                runtime_file=runtime_file,
            )

        # dev-режим: запускаем Streamlit как модуль: python -m streamlit run app.py
        cmd = [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            app_py,
            "--global.developmentMode",
            "false",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ]

        p = subprocess.Popen(
            cmd,
            env=os.environ.copy(),
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            return int(p.wait())
        except KeyboardInterrupt:
            try:
                p.terminate()
            except Exception:
                pass
            return 0
    finally:
        if is_frozen:
            _remove_runtime_state(runtime_file)


if __name__ == "__main__":
    raise SystemExit(main())

