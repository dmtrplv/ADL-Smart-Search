param(
  [string]$AdlDir = "",
  [string]$CkmAdlDir = "",
  [string]$IndexDir = "data",
  [string]$Python = ""
)

$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

if (-not $Python) {
  if (Test-Path ".venv\Scripts\python.exe") {
    $Python = (Resolve-Path ".venv\Scripts\python.exe").Path
  } else {
    Write-Host "[ERROR] Python не найден. Запустите из .venv либо передайте -Python <path>."
    exit 1
  }
}
$venvPython = (Resolve-Path $Python).Path

# Жёстко офлайн: модель должна быть уже в локальном кэше (dev-машина).
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
if (-not $env:ADL_EMBED_MODEL) {
  $env:ADL_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
}

if (-not $AdlDir) { $AdlDir = $env:ADL_ADL_DIR }
if (-not $AdlDir) {
  Write-Host "[ERROR] Укажите папку с .adl: -AdlDir <path> или ADL_ADL_DIR."
  exit 1
}
if (-not (Test-Path $AdlDir)) {
  Write-Host "[ERROR] Папка с .adl не существует: $AdlDir"
  exit 1
}

Write-Host "[INFO] Сборка индекса локальной базы: $AdlDir -> $IndexDir\local"
Write-Host "[INFO] Модель: $env:ADL_EMBED_MODEL"
$env:ADL_CKM_DIR = "" # CKM собирается отдельным шагом ниже
& $venvPython -m adl_smart_search.rebuild_index --adl-dir $AdlDir --index-dir $IndexDir --source local
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $CkmAdlDir) { $CkmAdlDir = $env:ADL_CKM_ADL_DIR }
if ($CkmAdlDir) {
  if (-not (Test-Path $CkmAdlDir)) {
    Write-Host "[ERROR] Папка CKM .adl не существует: $CkmAdlDir"
    exit 1
  }
  Write-Host "[INFO] Сборка CKM-индекса: $CkmAdlDir -> $IndexDir\ckm"
  $env:ADL_DIR = "" # чтобы случайный ADL_DIR не собрал «локальную» базу
  & $venvPython -m adl_smart_search.rebuild_index --ckm-adl-dir $CkmAdlDir --index-dir $IndexDir --source ckm
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
  Write-Host "[INFO] CKM не задан (пропуск)."
}

Write-Host ""
Write-Host "Готово. Проверка артефакта:"
& $venvPython -m adl_smart_search.rebuild_index --verify $IndexDir
exit $LASTEXITCODE