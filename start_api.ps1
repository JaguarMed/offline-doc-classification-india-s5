# Script PowerShell pour démarrer l'API
$env:PYTHONPATH = $PSScriptRoot
Set-Location $PSScriptRoot
Write-Host "Starting Document Classification API..." -ForegroundColor Green
python -m api.main






