# Сборка exe и артефактов

## Один раз на машине сборки

1. Python **3.11 или 3.12 x64**.
2. Создать окружение и поставить зависимости:

   ```powershell
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Один раз запустить dev-версию с интернетом, чтобы скачалась embedding-модель:

   ```powershell
   streamlit run app.py
   ```

   Модель должна появиться в пользовательском кэше HuggingFace. При сборке она
   копируется в `dist\<сборка>\models`.

## Сборка приложения

```powershell
.\build_exe.ps1 -Mode onedir -Zip
```

- `-Zip` — упаковать готовую папку в один архив.
- `-IncludeModels` (по умолчанию включён) — положить embedding-модель в `models\`.

`data/` и `config/` в дистрибутив не попадают никогда. Индексы собираются
отдельно и коммитятся в приватный репозиторий с данными.

## Данные и конфиги

Приложение больше не хранит `data/` и `config/` рядом с exe. Пользователь
подключает отдельный приватный репозиторий:

```text
<repo-root>/                 ← Git-репозиторий, например C:\emiastemplates
  Soft/
    ADL Smart Search/        ← data_root
      config/
        settings.json
        llm.json
        prompts.json
        corp_catalog.json
        .env
      data/
        local/
        ckm/
        build_info.json
```

- `config/` — базовые настройки, которые обновляете вы.
- `data/` — индексы; `build_info.json` хранит даты сборки по каждому источнику.
- `.env` содержит общие ключи LLM и лежит только в приватном репозитории.
- `prompts.json` — все тексты промптов. Нужен только для запросов к LLM:
  без него поиск работает, а ошибка появляется при обращении к ИИ. Промптов в коде нет.

Пользовательские переопределения и путь к репозиторию хранятся вне его:

```text
%LOCALAPPDATA%\ADL-Smart-Search\
  config\
    bootstrap.json
  logs\
```

В пользовательской сборке сохраняется только путь к репозиторию с `data/` и
`config/`. Поисковые настройки, LLM и промпты в пользовательских файлах не хранятся.
Выбор провайдера ИИ в чате действует только в текущей сессии; при запуске всегда
используется `config/llm.json` (по умолчанию — МЭШ.ИИ.Хаб).

При первом запуске exe проверяет `C:\emiastemplates\Soft\ADL Smart Search`.
Если папки нет, приложение предлагает выбрать корень репозитория `emiastemplates`
через **«Обзор»**; путь внутри репозитория (`Soft\ADL Smart Search`) уже известен.
Автопоиска по диску нет. В `bootstrap.json` сохраняются `repo_root`
(корень Git-репозитория) и `data_root` (папка с `config/` и `data/`);
обновления их не перезаписывают.

При запуске из папки проекта (`streamlit run app.py`, `python launcher.py`) пути
другие: `<project>/data` + `<project>/config`, а overrides/bootstrap/логи — в
`<project>/.adl-local`. В этом режиме dev-панель пишет изменения (включая промпты
в `config/prompts.json`) как в админский конфиг. В стандартной exe-сборке
dev-режима нет; он доступен только в админской сборке с флагом `-WithDevMode`
(маркер `dev_mode.enabled` рядом с exe).

## Что получится

`dist\ADL-Smart-Search-<дата-время>\`:

- `ADL-Smart-Search.exe` — основное приложение (открывает браузер);
- `ADL-Smart-Search-Updater.exe` — небольшой updater, который скачивает и применяет
  обновления;
- `version.json` — версия установленного приложения для updater;
- `models\` — embedding-модель для офлайн-работы;
- `sertificates\` — корневой сертификат Минцифры для GigaChat.

`ADL-Smart-Search-Index.exe` в стандартную сборку не входит. Он собирается
отдельно для администратора:

```powershell
.\build_exe.ps1 -Mode onedir -WithIndexer
```

Админская сборка с dev-режимом:

```powershell
.\build_exe.ps1 -Mode onedir -WithDevMode -WithIndexer
```

Флаг `-WithDevMode` создаёт рядом с exe маркер `dev_mode.enabled`. Без него
переключатель dev-режима и dev-панель в exe не показываются.

`data\` и `config\` в дистрибутив не входят.

`dist\ADL-Smart-Search-<дата-время>.zip` — тот же дистрибутив одним архивом.

## Обновление приложения

Приложение проверяет манифест из `update/stable.json` публичного репозитория.
Если версия новее, в UI появляется кнопка «Обновить сейчас».

- `ADL-Smart-Search-Updater.exe` запускается из временной копии, ждёт выхода
  приложения, скачивает delta/full asset, проверяет SHA-256, применяет и
  перезапускает приложение.
- Полный zip и дельты загружаются в GitHub Release, а не в Git.
- Манифест и история версий лежат в `update/` и коммитятся.

Подробности — в `update/README.md`.

## Публикация релиза

```powershell
python scripts/make_release.py `
  --dist dist\ADL-Smart-Search-<дата> `
  --full-zip dist\ADL-Smart-Search-<дата>.zip `
  --version 0.9.0 `
  --channel stable `
  --github-repo dmtrplv/ADL-Smart-Search
```

Загрузите `update/assets/*` в GitHub Release с тегом `v0.9.0`, затем закоммитьте
`update/stable.json` и `update/manifests/0.9.0.json`.

## Передача пользователю

1. Передать zip.
2. Пользователь распаковывает архив в любую папку и запускает основной exe.
3. Пользователь подтягивает приватный репозиторий с данными через Git.
4. При первом запуске приложение находит `C:\emiastemplates\Soft\ADL Smart Search`
   автоматически или просит указать папку `emiastemplates` / папку с `config/` и `data/`.
5. Ключи LLM лежат в `<data_root>/config/.env`; отдельный `.env` рядом с exe
   больше не нужен.

## Обновить только индексы

```powershell
.\scripts\build-index.ps1 -AdlDir C:\Archetypes -CkmAdlDir C:\CKM-mirror -IndexDir C:\path\to\data-repo\data
```

Скрипт обновит `local/`, `ckm/` и `build_info.json`. После этого закоммитьте и
запушьте изменения в приватный репозиторий.

Или запустить `ADL-Smart-Search-Index.exe` из админской сборки (`-WithIndexer`),
либо основной `ADL-Smart-Search.exe --index`.

## Версия

Версия задаётся в `adl_smart_search\__init__.py` (`__version__`) и показывается
в правом нижнем углу приложения.
