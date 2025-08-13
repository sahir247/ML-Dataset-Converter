# Build script for Windows PowerShell
# Creates a standalone EXE using PyInstaller

param(
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

# Ensure venv exists
if (-not (Test-Path ".venv/Scripts/python.exe")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    py -m venv .venv
}

$py = ".venv/ Scripts/ python.exe" -replace ' ', ''

# Optionally clean old builds
if ($Clean) {
    if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
    if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
}

# Install runtime dependencies (single consolidated file)
Write-Host "Installing project dependencies..." -ForegroundColor Cyan
& $py -m pip install --upgrade pip
if (Test-Path "requirements.txt") { & $py -m pip install -r requirements.txt }

# Ensure PyInstaller
Write-Host "Installing build dependency: pyinstaller" -ForegroundColor Cyan
& $py -m pip install pyinstaller

# Build
Write-Host "Building executable..." -ForegroundColor Cyan

# Collect local model weights (.pt) to include alongside the EXE
$ptFiles = Get-ChildItem -File -Filter *.pt -ErrorAction SilentlyContinue
$addDataArgs = @()
foreach ($f in $ptFiles) {
    $addDataArgs += @('--add-data', "$($f.Name);.")
}

& $py -m PyInstaller `
    --name "DatasetConverter" `
    --noconfirm `
    --windowed `
    --clean `
    --paths "." `
    --add-data "requirements.txt;." `
    $addDataArgs `
    app.py

Write-Host "Build complete. EXE at: dist/DatasetConverter/DatasetConverter.exe" -ForegroundColor Green
