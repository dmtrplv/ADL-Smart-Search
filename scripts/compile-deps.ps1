# Регенерация requirements.lock.txt из requirements.txt (pip-compile).
#
# Требуется: установленный pip-tools (requirements-dev.txt).
# Запуск:    scripts\compile-deps.ps1
# Флаг -Upgrade пересчитывает все пакеты до актуальных версий (по умолчанию
# pip-compile сохраняет версии из текущего окружения).

param(
    [switch]$Upgrade
)

$ErrorActionPreference = "Stop"

# Определяем python из .venv проекта, если есть.
$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
if (!(Test-Path $python)) {
    $python = "python"
}

& $python -m piptools compile "requirements.txt" --output-file="requirements.lock.txt" @(if ($Upgrade) { "--upgrade" }) --strip-extras
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] pip-compile failed (проверьте, что установлен pip-tools)."
    exit $LASTEXITCODE
}
Write-Host "[OK] requirements.lock.txt перезаписан."