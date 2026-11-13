# One command to go from a fresh clone to a running API (Windows PowerShell).
#
#   .\run.ps1            set up, train, test, then start the API
#   .\run.ps1 -NoServe   same, but stop after the tests
#
# If PowerShell blocks the script, run it like this instead:
#   powershell -ExecutionPolicy Bypass -File .\run.ps1

param([switch]$NoServe)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$py = ".\.venv\Scripts\python.exe"

# PowerShell doesn't stop on a failing python command by itself, so we check.
function Step($title, [scriptblock]$cmd) {
    Write-Host "`n== $title" -ForegroundColor Cyan
    & $cmd
    if ($LASTEXITCODE -ne 0) { throw "Step failed: $title" }
}

if (-not (Test-Path $py)) {
    Step "Create virtual environment (.venv)" { python -m venv .venv }
}
Step "Install packages" { & $py -m pip install --quiet --upgrade pip; & $py -m pip install --quiet -r requirements.txt -e . }
Step "Download FD001 data" { & $py scripts\download_data.py }
Step "Train and compare models (logged to MLflow)" { & $py -m turbofan_rul.train }
Step "Build website data (docs\data.js)" { & $py scripts\build_site_data.py }
Step "Run tests" { & $py -m pytest }

Write-Host "`nDone. See your runs with:  .\.venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlflow.db" -ForegroundColor Green

if (-not $NoServe) {
    Write-Host "Starting the API. Open http://127.0.0.1:8000/docs  (Ctrl+C to stop)" -ForegroundColor Green
    & $py -m uvicorn turbofan_rul.api:app --port 8000
}
