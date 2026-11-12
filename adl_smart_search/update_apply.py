"""Скачивание и применение обновления (используется updater-процессом)."""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
import zipfile
from datetime import datetime

import requests

from adl_smart_search.update_manifest import (
    file_sha256,
    manifest_files,
    validate_manifest,
)

logger = logging.getLogger(__name__)

BUFFER = 1024 * 1024


def _safe_member(name: str) -> str:
    rel = name.replace("\\", "/").strip()
    if not rel or rel.startswith("/") or (len(rel) > 1 and rel[1] == ":"):
        raise ValueError(f"unsafe archive path: {name!r}")
    parts = [p for p in rel.split("/") if p not in ("", ".")]
    if any(p == ".." for p in parts):
        raise ValueError(f"unsafe archive path: {name!r}")
    return "/".join(parts)


def download_file(url: str, dest: str, *, expected_sha256: str, expected_size: int | None = None) -> None:
    """Скачивает файл с поддержкой resume и проверяет SHA-256."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    part = f"{dest}.part"
    if os.path.exists(dest) and file_sha256(dest) == expected_sha256.lower():
        return
    if os.path.exists(dest):
        os.remove(dest)
    have = os.path.getsize(part) if os.path.exists(part) else 0
    headers = {}
    if have:
        headers["Range"] = f"bytes={have}-"
    with requests.get(url, stream=True, timeout=60, headers=headers) as response:
        if have and response.status_code == 206:
            mode = "ab"
        else:
            have = 0
            mode = "wb"
        response.raise_for_status()
        with open(part, mode) as f:
            done = have
            for chunk in response.iter_content(BUFFER):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if expected_size:
                    done_mb = done / (1024 * 1024)
                    total_mb = expected_size / (1024 * 1024)
                    print(f"\r  {done_mb:.1f} / {total_mb:.1f} МБ", end="", flush=True)
    print()
    actual = file_sha256(part)
    if actual != expected_sha256.lower():
        try:
            os.remove(part)
        except OSError:
            pass
        raise ValueError(f"download checksum mismatch: {actual} != {expected_sha256}")
    os.replace(part, dest)


def extract_archive(archive: str, dest: str) -> list[str]:
    """Безопасно распаковывает zip. Возвращает список относительных путей."""
    shutil.rmtree(dest, ignore_errors=True)
    os.makedirs(dest, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(archive) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = _safe_member(info.filename)
            target = os.path.join(dest, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(target) or dest, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out, length=BUFFER)
            extracted.append(rel)
    return extracted


def verify_files(root: str, manifest: dict) -> list[str]:
    """Возвращает список файлов, которых нет или у которых неверный SHA-256."""
    bad: list[str] = []
    for rel, item in manifest_files(manifest).items():
        full = os.path.join(root, rel.replace("/", os.sep))
        try:
            if not os.path.isfile(full) or file_sha256(full) != str(item.get("sha256") or "").lower():
                bad.append(rel)
        except OSError:
            bad.append(rel)
    return bad


def managed_files(root: str, manifest: dict) -> set[str]:
    """Все текущие файлы внутри managed_roots (для удаления устаревших)."""
    out: set[str] = set()
    for rel_root in manifest.get("managed_roots") or []:
        rel_root = str(rel_root).replace("\\", "/").strip("/")
        if not rel_root:
            continue
        full_root = os.path.join(root, rel_root.replace("/", os.sep))
        if os.path.isfile(full_root):
            out.add(rel_root)
            continue
        if not os.path.isdir(full_root):
            continue
        for dirpath, _dirnames, filenames in os.walk(full_root, followlinks=False):
            for name in filenames:
                full = os.path.join(dirpath, name)
                rel = os.path.relpath(full, root).replace("\\", "/")
                if not rel.startswith("update") and not rel.startswith("updates/"):
                    out.add(rel)
    return out


def _backup_file(root: str, backup_root: str, rel: str) -> None:
    src = os.path.join(root, rel.replace("/", os.sep))
    if not os.path.isfile(src):
        return
    dst = os.path.join(backup_root, rel.replace("/", os.sep))
    os.makedirs(os.path.dirname(dst) or backup_root, exist_ok=True)
    try:
        os.replace(src, dst)
    except OSError:
        shutil.copy2(src, dst)
        os.remove(src)


def _restore_backup(backup_root: str, root: str) -> None:
    if not os.path.isdir(backup_root):
        return
    for dirpath, _dirnames, filenames in os.walk(backup_root):
        for name in filenames:
            src = os.path.join(dirpath, name)
            rel = os.path.relpath(src, backup_root)
            dst = os.path.join(root, rel)
            os.makedirs(os.path.dirname(dst) or root, exist_ok=True)
            try:
                os.replace(src, dst)
            except OSError:
                shutil.copy2(src, dst)


def _apply_staging(root: str, staging: str, manifest: dict, *, backup_root: str | None) -> None:
    files = set(manifest_files(manifest))
    managed_before = managed_files(root, manifest)
    # Удаляем файлы, которых больше нет в релизе.
    for rel in sorted(managed_before - files):
        if backup_root:
            _backup_file(root, backup_root, rel)
        else:
            full = os.path.join(root, rel.replace("/", os.sep))
            try:
                os.remove(full)
            except OSError:
                pass
    for dirpath, _dirnames, filenames in os.walk(staging, followlinks=False):
        for name in filenames:
            src = os.path.join(dirpath, name)
            rel = os.path.relpath(src, staging).replace("\\", "/")
            if rel not in files:
                continue
            dst = os.path.join(root, rel.replace("/", os.sep))
            os.makedirs(os.path.dirname(dst) or root, exist_ok=True)
            if backup_root:
                _backup_file(root, backup_root, rel)
            tmp = f"{dst}.update-tmp"
            shutil.copy2(src, tmp)
            os.replace(tmp, dst)


def apply_archive(
    *,
    install_dir: str,
    archive: str,
    manifest: dict,
    asset_kind: str,
    work_dir: str,
) -> None:
    """Применяет full/delta архив с проверкой и откатом для delta."""
    validate_manifest(manifest)
    staging = os.path.join(work_dir, "staging")
    extracted = set(extract_archive(archive, staging))
    expected = manifest_files(manifest)
    unknown = extracted - set(expected)
    if unknown:
        raise ValueError(f"archive contains files not in manifest: {sorted(unknown)[:5]}")

    if asset_kind == "full":
        missing = set(expected) - extracted
        if missing:
            raise ValueError(f"full archive is incomplete: {sorted(missing)[:5]}")

    bad = []
    for rel in extracted:
        item = expected[rel]
        full = os.path.join(staging, rel.replace("/", os.sep))
        if file_sha256(full) != str(item.get("sha256") or "").lower():
            bad.append(rel)
    if bad:
        raise ValueError(f"staging checksum mismatch: {bad[:5]}")

    backup_root = None
    if asset_kind != "full":
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_root = os.path.join(work_dir, f"backup-{stamp}")
        os.makedirs(backup_root, exist_ok=True)
    try:
        _apply_staging(install_dir, staging, manifest, backup_root=backup_root)
        bad = verify_files(install_dir, manifest)
        if bad:
            raise ValueError(f"installed files failed verification: {bad[:5]}")
    except Exception:
        if backup_root:
            logger.exception("update: apply failed, restoring backup")
            _restore_backup(backup_root, install_dir)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
        if backup_root and os.path.isdir(backup_root):
            shutil.rmtree(backup_root, ignore_errors=True)


def wait_for_pid(pid: int, timeout_s: float = 60.0) -> None:
    if not pid or pid <= 0:
        return
    if os.name == "nt":
        try:
            import ctypes

            SYNCHRONIZE = 0x00100000
            handle = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
            if handle:
                try:
                    ctypes.windll.kernel32.WaitForSingleObject(handle, int(timeout_s * 1000))
                finally:
                    ctypes.windll.kernel32.CloseHandle(handle)
                return
        except Exception:
            logger.exception("update: wait_for_pid failed")
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            return
        time.sleep(0.5)


def choose_asset(manifest: dict, current_version: str) -> tuple[str, dict]:
    assets = manifest.get("assets") or {}
    deltas = assets.get("deltas") or {}
    delta = deltas.get(current_version)
    if isinstance(delta, dict) and delta.get("url"):
        return "delta", delta
    full = assets.get("full")
    if isinstance(full, dict) and full.get("url"):
        return "full", full
    raise ValueError("manifest has no download asset for this version")


def run_job(job_path: str) -> int:
    with open(job_path, "r", encoding="utf-8") as f:
        job = json.load(f)
    install_dir = os.path.abspath(str(job["install_dir"]))
    manifest_path = str(job["manifest_path"])
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    validate_manifest(manifest)

    wait_for_pid(int(job.get("wait_pid") or 0))
    kind, asset = choose_asset(manifest, str(job.get("current_version") or ""))
    work_dir = os.path.join(install_dir, "updates")
    os.makedirs(work_dir, exist_ok=True)
    archive = os.path.join(work_dir, f"download-{manifest.get('version')}.zip")

    print(f"Обновление до {manifest.get('version')} ({kind})")
    try:
        download_file(
            str(asset["url"]),
            archive,
            expected_sha256=str(asset["sha256"]),
            expected_size=int(asset.get("size") or 0) or None,
        )
        apply_archive(
            install_dir=install_dir,
            archive=archive,
            manifest=manifest,
            asset_kind=kind,
            work_dir=work_dir,
        )
    except Exception:
        if kind != "delta":
            raise
        logger.exception("updater: delta failed, falling back to full archive")
        print("\nДельта не применилась, скачиваю полный архив...")
        full = (manifest.get("assets") or {}).get("full")
        if not isinstance(full, dict) or not full.get("url"):
            raise
        download_file(
            str(full["url"]),
            archive,
            expected_sha256=str(full["sha256"]),
            expected_size=int(full.get("size") or 0) or None,
        )
        apply_archive(
            install_dir=install_dir,
            archive=archive,
            manifest=manifest,
            asset_kind="full",
            work_dir=work_dir,
        )
    try:
        os.remove(archive)
    except OSError:
        pass
    try:
        os.remove(job_path)
    except OSError:
        pass

    restart = str(job.get("restart_exe") or os.path.join(install_dir, "ADL-Smart-Search.exe"))
    print(f"Готово. Запускаю {restart}")
    subprocess.Popen([restart], cwd=install_dir)
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="ADL-Smart-Search-Updater")
    parser.add_argument("--job", required=True, help="Путь к job.json")
    args = parser.parse_args(argv)
    try:
        return run_job(args.job)
    except Exception as e:
        print(f"\n[ERROR] Обновление не удалось: {e}")
        logger.exception("updater failed")
        return 1
