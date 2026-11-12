# Канал обновлений приложения

Здесь лежат манифесты релизов, которые читает приложение.

- `stable.json` — текущий манифест стабильного канала (его скачивает приложение).
- `manifests/<version>.json` — история манифестов для генерации дельт.
- `assets/` — полный zip и дельта-архивы. В Git не коммитится: загружайте их в
  GitHub Release.

## Сборка релиза

```powershell
# 1. Собрать приложение (без data/)
.\build_exe.ps1 -Mode onedir -Zip

# 2. Сгенерировать манифест и дельты
python scripts/make_release.py `
  --dist dist\ADL-Smart-Search-<дата> `
  --full-zip dist\ADL-Smart-Search-<дата>.zip `
  --version 0.9.0 `
  --channel stable `
  --notes-file release-notes.md `
  --github-repo dmtrplv/ADL-Smart-Search
```

Скрипт создаст:

- `update/stable.json` — манифест канала;
- `update/manifests/0.9.0.json` — история;
- `update/assets/ADL-Smart-Search-0.9.0-full.zip` — полный архив;
- `update/assets/delta-<old>-to-0.9.0.zip` — дельты от предыдущих версий.

Загрузите `update/assets/*` в GitHub Release с тегом `v0.9.0`, затем
закоммитьте `update/stable.json` и `update/manifests/0.9.0.json`.

## Настройка клиента

По умолчанию приложение читает:

```text
https://raw.githubusercontent.com/dmtrplv/ADL-Smart-Search/main/update/stable.json
```

Если манифест лежит в другом месте:

```text
ADL_UPDATE_FEED=https://example.test/update/stable.json
```

Полностью отключить проверку:

```text
ADL_UPDATE_CHECK=0
```
