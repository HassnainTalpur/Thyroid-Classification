$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$checkpoint = Join-Path $PSScriptRoot "model\convnext_tiny_seed123_best.pt"

if (-not (Test-Path $py)) {
    throw "Virtual environment not found. Run .\setup_windows.ps1 first."
}

if (-not (Test-Path $checkpoint)) {
    Write-Warning "Checkpoint not found: $checkpoint"
    Write-Warning "The UI will open, but prediction will remain disabled until the .pt file is added."
}

& $py -m streamlit run app.py
