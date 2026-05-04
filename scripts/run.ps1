$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$InstallScript = Join-Path $PSScriptRoot "install.ps1"
$StreamlitExe = Join-Path $ProjectRoot "venv\Scripts\streamlit.exe"

Set-Location $ProjectRoot

& $InstallScript

Write-Host "Launching TinyMozart Streamer..."
& $StreamlitExe run app.py --server.address localhost --server.port 8501
