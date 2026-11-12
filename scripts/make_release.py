"""Создание манифеста релиза и дельта-архивов.

Пример:

    python scripts/make_release.py `
      --dist dist/ADL-Smart-Search-20261105-120000 `
      --full-zip dist/ADL-Smart-Search-20261105-120000.zip `
      --version 0.9.1 `
      --channel beta `
      --github-repo dmtrplv/ADL-Smart-Search

Скрипт пишет:

* update/<channel>.json — манифест для приложения (feed);
* update/manifests/<version>.json — история манифестов для будущих дельт;
* update/assets/<...>.zip — полный архив/дельты (если нужен стабильный asset name).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from adl_smart_search.update_manifest import (  # noqa: E402
    build_manifest,
    create_delta_zip,
    file_sha256,
    load_manifest,
    save_manifest,
    version_parts,
)


def _link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if os.path.exists(dst):
        os.remove(dst)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _asset(url: str, path: str) -> dict:
    return {
        "url": url,
        "size": os.path.getsize(path),
        "sha256": file_sha256(path),
    }


def _zip_names(path: str) -> set[str]:
    import zipfile

    with zipfile.ZipFile(path) as zf:
        return {
            info.filename.replace("\\", "/")
            for info in zf.infolist()
            if not info.is_dir() and not info.filename.endswith("/")
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Собрать манифест релиза и дельты.")
    parser.add_argument("--dist", required=True, help="Папка дистрибутива.")
    parser.add_argument("--full-zip", required=True, help="Полный zip релиза.")
    parser.add_argument("--version", default="", help="Версия (по умолчанию из version.json).")
    parser.add_argument("--channel", default="stable", choices=["stable", "beta"])
    parser.add_argument("--notes", default="", help="Текст release notes.")
    parser.add_argument("--notes-file", default="", help="Файл с release notes.")
    parser.add_argument("--github-repo", default="dmtrplv/ADL-Smart-Search")
    parser.add_argument("--tag", default="", help="Тег релиза (по умолчанию v<version>).")
    parser.add_argument("--asset-base-url", default="", help="Базовый URL для assets.")
    parser.add_argument("--out", default="update", help="Куда писать update/.")
    parser.add_argument("--max-deltas", type=int, default=3)
    args = parser.parse_args(argv)

    version = (args.version or "").strip()
    if not version:
        version_path = os.path.join(args.dist, "version.json")
        with open(version_path, "r", encoding="utf-8") as f:
            version = str(json.load(f).get("version") or "").strip()
    if not version:
        print("[ERROR] Не удалось определить версию.")
        return 1

    notes = args.notes
    if args.notes_file:
        with open(args.notes_file, "r", encoding="utf-8") as f:
            notes = f.read().strip()

    tag = (args.tag or f"v{version}").strip()
    if args.asset_base_url:
        base_url = args.asset_base_url.rstrip("/")
    else:
        base_url = f"https://github.com/{args.github_repo}/releases/download/{tag}"

    out_dir = os.path.abspath(args.out)
    assets_dir = os.path.join(out_dir, "assets")
    manifest_dir = os.path.join(out_dir, "manifests")
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)

    full_asset_name = f"ADL-Smart-Search-{version}-full.zip"
    full_asset_path = os.path.join(assets_dir, full_asset_name)
    _link_or_copy(args.full_zip, full_asset_path)

    previous: list[tuple[str, dict]] = []
    for name in os.listdir(manifest_dir):
        if not name.endswith(".json"):
            continue
        old_version = name[:-5]
        if old_version == version:
            continue
        try:
            previous.append((old_version, load_manifest(os.path.join(manifest_dir, name))))
        except Exception as e:
            print(f"[WARN] пропущен старый манифест {name}: {e}")
    previous.sort(key=lambda item: version_parts(item[0]))
    previous = previous[-max(0, args.max_deltas) :]

    manifest = build_manifest(
        args.dist,
        version=version,
        channel=args.channel,
        notes=notes,
        published_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        assets={
            "full": _asset(f"{base_url}/{full_asset_name}", full_asset_path),
            "deltas": {},
        },
    )

    expected = {item["path"] for item in manifest["files"]}
    actual = _zip_names(full_asset_path)
    extra = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extra or missing:
        print("[ERROR] Полный zip не совпадает с манифестом.")
        if extra:
            print("  Лишние файлы (не должны попадать в релиз):", ", ".join(extra[:10]))
        if missing:
            print("  Отсутствуют в zip:", ", ".join(missing[:10]))
        return 1

    for old_version, old_manifest in previous:
        delta_name = f"delta-{old_version}-to-{version}.zip"
        delta_path = os.path.join(assets_dir, delta_name)
        count, size = create_delta_zip(args.dist, old_manifest, manifest, delta_path)
        manifest["assets"]["deltas"][old_version] = _asset(f"{base_url}/{delta_name}", delta_path)
        print(f"[OK] delta {old_version} -> {version}: {count} файлов, {size / (1024 * 1024):.1f} МБ")

    manifest_path = os.path.join(out_dir, f"{args.channel}.json")
    save_manifest(manifest_path, manifest)
    save_manifest(os.path.join(manifest_dir, f"{version}.json"), manifest)
    print(f"[OK] manifest: {manifest_path}")
    print(f"[OK] full asset: {full_asset_path}")
    print("")
    print("Загрузите в GitHub Release:")
    print(f"  {full_asset_path}")
    for name in sorted(os.listdir(assets_dir)):
        if name.startswith("delta-"):
            print(f"  {os.path.join(assets_dir, name)}")
    print("")
    print(f"Опубликуйте {manifest_path} как update/{args.channel}.json в публичном")
    print("репозитории (или укажите путь через ADL_UPDATE_FEED).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
