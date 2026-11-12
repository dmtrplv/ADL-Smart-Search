Param(
  [ValidateSet("onedir")]
  [string]$Mode = "onedir",
  # Рекомендуется Python 3.11/3.12 x64. Для PyInstaller/torch это самый стабильный вариант под Windows.
  [string]$Python = "",
  [switch]$AllowAnyPython,
  [switch]$IncludeModels = $true,
  # Собрать ADL-Smart-Search-Index.exe (нужен администратору для индексации).
  # В обычную пользовательскую сборку не входит.
  [switch]$WithIndexer,
  # Админская сборка: разрешить dev-режим (маркер dev_mode.enabled рядом с exe).
  # Без этого флага dev-режим в exe недоступен.
  [switch]$WithDevMode,
  # Упаковать dist\<buildName> в zip одним архивом.
  [switch]$Zip
)

$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

# В _internal/adl_smart_search попадает исходное дерево пакета. Убираем байткод,
# чтобы он не тащился в релиз и не расходился с манифестом обновления.
$packageDir = Join-Path $PSScriptRoot "adl_smart_search"
if (Test-Path $packageDir) {
  Get-ChildItem -LiteralPath $packageDir -Directory -Recurse -Filter "__pycache__" -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
  Get-ChildItem -LiteralPath $packageDir -Recurse -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -in ".pyc", ".pyo" } |
    Remove-Item -Force -ErrorAction SilentlyContinue
}

function Get-BuildPython {
  param([string]$Preferred)

  if ($Preferred -and (Test-Path $Preferred)) {
    return (Resolve-Path $Preferred).Path
  }

  if (Test-Path ".venv\\Scripts\\python.exe") {
    return (Resolve-Path ".venv\\Scripts\\python.exe").Path
  }

  # Попробуем системный launcher: py -3.11
  $py = Get-Command py -ErrorAction SilentlyContinue
  if ($py) {
    return "py -3.11"
  }

  return ""
}

$pythonCmd = Get-BuildPython -Preferred $Python
if (-not $pythonCmd) {
  Write-Host "[ERROR] Build python not found."
  Write-Host "Create .venv manually or install Python 3.11 x64 so 'py -3.11' works."
  exit 1
}

# Если .venv отсутствует и есть py -3.11 — создадим venv автоматически.
if (!(Test-Path ".venv\\Scripts\\python.exe") -and ($pythonCmd -eq "py -3.11")) {
  Write-Host "[INFO] Creating .venv using: py -3.11"
  & py -3.11 -m venv ".venv"
}

$venvPython = (Resolve-Path ".venv\\Scripts\\python.exe").Path

$pyVer = & $venvPython -c "import sys; print('.'.join(map(str, sys.version_info[:3])))"
Write-Host "[INFO] Using python: $venvPython ($pyVer)"

$majorMinor = & $venvPython -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"
if ($majorMinor -notin @("3.11","3.12")) {
  $msg = "Detected Python $majorMinor in .venv. For Windows builds, Python 3.11/3.12 is strongly recommended (torch/sentence-transformers wheels)."
  if ($AllowAnyPython) {
    Write-Host "[WARN] $msg"
  } else {
    Write-Host "[ERROR] $msg"
    Write-Host "Fix: delete .venv and run: .\\build_exe.ps1 (it will try 'py -3.11 -m venv')."
    Write-Host "Or pass -AllowAnyPython to proceed at your own risk."
    exit 1
  }
}

& $venvPython -m pip install -U pip "pyinstaller==6.22.3" "setuptools<82"

$reqFile = "requirements.txt"
if (Test-Path "requirements.lock.txt") {
  $reqFile = "requirements.lock.txt"
  Write-Host "[INFO] Using locked requirements: $reqFile"
} else {
  Write-Host "[INFO] Using requirements: $reqFile"
}
& $venvPython -m pip install -r $reqFile

# Streamlit static assets path is installation-specific. Compute it from current venv to avoid hardcoded paths.
$streamlitStatic = (& $venvPython -c "import os, streamlit; print(os.path.join(os.path.dirname(streamlit.__file__), 'static'))").Trim()
if (!(Test-Path $streamlitStatic)) {
  Write-Host "[ERROR] Streamlit static dir not found: $streamlitStatic"
  exit 1
}

$baseName = "ADL-Smart-Search"
$buildName = "$baseName-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
$distOut = Join-Path $PSScriptRoot "dist\\$buildName"
Write-Host "[INFO] Build name: $buildName"

function Normalize-EmbedModelId {
  param([string]$ModelId)
  $m = $ModelId
  if ($null -eq $m) { $m = "" }
  $m = $m.Trim()
  if (-not $m) { return "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" }
  if ($m -notmatch "/") { return "sentence-transformers/$m" }
  return $m
}

function Get-HfHubCacheDir {
  param([string]$PythonExe)

  # Preferred: ask huggingface_hub for the actual cache location.
  try {
    $p = (& $PythonExe -c "from huggingface_hub import constants; print(constants.HF_HUB_CACHE)" 2>$null).Trim()
    if ($p -and (Test-Path $p)) { return $p }
  } catch {}

  # Fallbacks for Windows defaults
  $candidates = @()
  if ($env:HF_HOME) {
    $candidates += (Join-Path $env:HF_HOME "hub")
  }
  if ($env:USERPROFILE) {
    $candidates += (Join-Path $env:USERPROFILE ".cache\\huggingface\\hub")
  }
  if ($env:LOCALAPPDATA) {
    $candidates += (Join-Path $env:LOCALAPPDATA "huggingface\\hub")
  }

  foreach ($c in $candidates) {
    if ($c -and (Test-Path $c)) { return $c }
  }
  return ""
}

function Get-HfModelFolderName {
  param([string]$RepoId)
  $id = Normalize-EmbedModelId -ModelId $RepoId
  $parts = $id.Split("/", 2)
  if ($parts.Count -ne 2) { return "" }
  return "models--$($parts[0])--$($parts[1])"
}

function Test-HfModelCacheComplete {
  param([string]$ModelDir)

  if (-not (Test-Path $ModelDir)) { return $false }

  # Preferred HF cache layout: blobs contain the actual big files.
  $blobs = Join-Path $ModelDir "blobs"
  if (Test-Path $blobs) {
    $blobCount = (Get-ChildItem -LiteralPath $blobs -File -Force -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($blobCount -ge 1) { return $true }
  }

  # On Windows without symlinks, the cache can still be valid with weights stored directly in snapshots.
  $snap = Join-Path $ModelDir "snapshots"
  if (Test-Path $snap) {
    $w = Get-ChildItem -LiteralPath $snap -Recurse -File -Force -Include model.safetensors,pytorch_model.bin,model.bin -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($w) { return $true }
  }

  return $false
}

$args = @(
  "--noconfirm",
  "--clean",
  "--name", $buildName,
  "--noconsole",
  "--copy-metadata", "streamlit",
  "--hidden-import", "streamlit",
  "--hidden-import", "streamlit.web.cli",
  "--collect-submodules", "streamlit",
  "--collect-data", "streamlit",
  # sentence-transformers is imported lazily (inside Embedder._ensure_model), so PyInstaller can miss it.
  "--hidden-import", "sentence_transformers",
  "--collect-submodules", "sentence_transformers",
  "--collect-data", "sentence_transformers",
  # NumPy/SciPy/sklearn DLLs: явно забираем бинарники, чтобы в onedir не терялись
  # зависимости compiled-модулей (numpy._core.multiarray и т.п.).
  "--collect-binaries", "numpy",
  "--collect-submodules", "numpy",
  "--collect-binaries", "scipy",
  "--collect-binaries", "sklearn",
  # Индексатор запускается из этого же exe через --index.
  "--hidden-import", "adl_smart_search.rebuild_index",
  # Windows: пути к DLL из numpy.libs/scipy.libs/sklearn/.libs должны быть
  # зарегистрированы до импорта compiled-модулей.
  "--runtime-hook", "rthook_numpy.py",
  # GigaChat SDK is imported lazily inside the provider.
  "--hidden-import", "gigachat",
  "--collect-submodules", "gigachat",
  "--copy-metadata", "gigachat",
  "--add-data", "$streamlitStatic;streamlit\\static",
  "--add-data", "app.py;.",
  "--add-data", "adl_smart_search;adl_smart_search",
  "--add-data", "requirements.txt;.",
  "--add-data", "README.md;."
)

$args += "--onedir"

& $venvPython -m PyInstaller @args launcher.py
if ($LASTEXITCODE -ne 0) {
  Write-Host "[ERROR] Сборка основного приложения завершилась с ошибкой."
  exit $LASTEXITCODE
}

# Фиксированное имя основного exe: папка и zip остаются timestamped, exe — без даты.
$mainExeOld = Join-Path $distOut "$buildName.exe"
if (-not (Test-Path $mainExeOld)) {
  Write-Host "[ERROR] PyInstaller не создал основной exe: $mainExeOld"
  exit 1
}
Move-Item -LiteralPath $mainExeOld -Destination (Join-Path $distOut "ADL-Smart-Search.exe") -Force

# Версия приложения для updater и release-манифеста.
$appVersion = (& $venvPython -c "from adl_smart_search import __version__; print(__version__)").Trim()
$versionJson = @{
  version = $appVersion
  built_at = (Get-Date).ToUniversalTime().ToString("o")
} | ConvertTo-Json
[System.IO.File]::WriteAllText(
  (Join-Path $distOut "version.json"),
  $versionJson,
  (New-Object System.Text.UTF8Encoding($false))
)
Write-Host "[INFO] version.json: $appVersion"

# dev-режим: только в админской сборке. Маркер рядом с exe — признак,
# что переключатель dev-режима можно показывать.
if ($WithDevMode) {
  $markerPath = Join-Path $distOut "dev_mode.enabled"
  [System.IO.File]::WriteAllText(
    $markerPath,
    "ADL Smart Search: dev mode enabled for this build.`r`n",
    (New-Object System.Text.UTF8Encoding($false))
  )
  Write-Host "[INFO] dev-режим включён (dev_mode.enabled)."
} else {
  Write-Host "[INFO] Стандартная сборка: dev-режим отключён."
}

# Маленький updater. Запускается из временной копии и умеет заменять
# собственный exe в установленной папке.
Write-Host "[INFO] Сборка updater (ADL-Smart-Search-Updater.exe)..."
$updaterArgs = @(
  "--noconfirm",
  "--clean",
  "--onefile",
  "--console",
  "--name", "ADL-Smart-Search-Updater",
  "--distpath", $distOut
)
& $venvPython -m PyInstaller @updaterArgs updater_main.py
if ($LASTEXITCODE -ne 0) {
  Write-Host "[ERROR] Сборка updater завершилась с ошибкой."
  exit $LASTEXITCODE
}

# Тонкий консольный лаунчер индексатора: запускает основной exe с --index.
# Тяжёлые библиотеки не дублируются — используется _internal основного приложения.
# В пользовательскую сборку не входит; нужен только администратору.
if ($WithIndexer) {
  Write-Host "[INFO] Сборка лаунчера индексатора (ADL-Smart-Search-Index.exe)..."
  $indexArgs = @(
    "--noconfirm",
    "--clean",
    "--onefile",
    "--console",
    "--name", "ADL-Smart-Search-Index",
    "--distpath", $distOut
  )
  & $venvPython -m PyInstaller @indexArgs index_launcher.py
  if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Сборка лаунчера индексатора завершилась с ошибкой."
    exit $LASTEXITCODE
  }
} else {
  Write-Host "[INFO] ADL-Smart-Search-Index.exe не собран (используйте -WithIndexer для админской сборки)."
}

# После сборки (onedir) кладём модели и сертификаты рядом с exe.
# data и config в дистрибутив не входят: они живут во внешнем репозитории.
if (Test-Path $distOut) {
  $srcModels = Join-Path $PSScriptRoot "models"

  if ($IncludeModels -and (Test-Path $srcModels)) {
    $dstModels = Join-Path $distOut "models"
    New-Item -ItemType Directory -Path $dstModels -Force | Out-Null
    & robocopy $srcModels $dstModels /E /COPY:DAT /R:2 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) {
      Write-Host "[ERROR] robocopy models failed with exit code $LASTEXITCODE"
      exit 1
    }
  }

  # Корневой сертификат Минцифры для GigaChat (нужен Python/httpx).
  $certSrc = Join-Path $PSScriptRoot "sertificates\Russian_Trusted_Root_CA.cer"
  if (Test-Path $certSrc) {
    $dstCertDir = Join-Path $distOut "sertificates"
    New-Item -ItemType Directory -Path $dstCertDir -Force | Out-Null
    Copy-Item -LiteralPath $certSrc -Destination (Join-Path $dstCertDir "Russian_Trusted_Root_CA.cer") -Force
  }

  # Каталог моделей МЭШ.ИИ.Хаб живёт во внешнем config/ и здесь не копируется.

  # Prefer copying the embedding model from the user's HuggingFace hub cache (it contains blobs/weights).
  if ($IncludeModels) {
    $embedModelId = Normalize-EmbedModelId -ModelId $env:ADL_EMBED_MODEL
    $hfHubCache = Get-HfHubCacheDir -PythonExe $venvPython
    if (-not $hfHubCache) {
      Write-Host "[ERROR] Could not locate HuggingFace hub cache directory (HF_HUB_CACHE)."
      Write-Host "Fix: run dev once so the model is downloaded, or set HF_HOME before build."
      exit 1
    }

    $modelFolder = Get-HfModelFolderName -RepoId $embedModelId
    if (-not $modelFolder) {
      Write-Host "[ERROR] Could not compute HF model folder name for: $embedModelId"
      exit 1
    }

    $srcHfModelDir = Join-Path $hfHubCache $modelFolder
    if (!(Test-Path $srcHfModelDir)) {
      Write-Host "[ERROR] Model not found in HF hub cache:"
      Write-Host "  $srcHfModelDir"
      Write-Host "Fix: run dev once to download the model into your user cache, then rebuild."
      exit 1
    }
    if (!(Test-HfModelCacheComplete -ModelDir $srcHfModelDir)) {
      Write-Host "[ERROR] HF cache for model exists but looks incomplete (missing blobs/weights):"
      Write-Host "  $srcHfModelDir"
      Write-Host "Fix: run dev once with internet so the embedding model downloads fully, then rebuild."
      exit 1
    }

    $dstModels = Join-Path $distOut "models"
    # Project-style layout: keep hub cache directly under models/
    $dstHfModelDir = Join-Path $dstModels $modelFolder
    New-Item -ItemType Directory -Path $dstHfModelDir -Force | Out-Null
    & robocopy $srcHfModelDir $dstHfModelDir /E /COPY:DAT /R:2 /W:1 /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -ge 8) {
      Write-Host "[ERROR] robocopy HF model cache failed with exit code $LASTEXITCODE"
      exit 1
    }

    if (!(Test-HfModelCacheComplete -ModelDir $dstHfModelDir)) {
      Write-Host "[ERROR] Copied HF model cache is incomplete in dist (missing weights):"
      Write-Host "  $dstHfModelDir"
      exit 1
    }
  }

  # data/ и config/ в дистрибутив не входят: они подключаются из внешнего репозитория.
}

if ($Zip -and (Test-Path $distOut)) {
  $zipPath = Join-Path $PSScriptRoot "dist\$buildName.zip"
  if (Test-Path $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
  }
  Write-Host "[INFO] Упаковка в zip: $zipPath"
  Compress-Archive -Path (Join-Path $distOut "*") -DestinationPath $zipPath -Force
}

Write-Host ""
Write-Host "=== Сводка артефакта ==="
Write-Host "  Папка: $distOut"
$sizeItems = @()
foreach ($sub in @("_internal", "models")) {
  $p = Join-Path $distOut $sub
  if (Test-Path $p) {
    $bytes = (Get-ChildItem -LiteralPath $p -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    if ($null -eq $bytes) { $bytes = 0 }
    $sizeItems += [pscustomobject]@{ Name = $sub; SizeMB = [math]::Round($bytes / 1MB, 1) }
  }
}
if ($sizeItems.Count -gt 0) {
  $sizeItems | Format-Table Name, SizeMB -AutoSize | Out-String | Write-Host
}
if ($Zip -and (Test-Path (Join-Path $PSScriptRoot "dist\$buildName.zip"))) {
  Write-Host "  Zip: dist\$buildName.zip"
} else {
  Write-Host "  Zip: не создан (передайте -Zip для упаковки)"
}

Write-Host ""
Write-Host "Done. See .\$distOut"

