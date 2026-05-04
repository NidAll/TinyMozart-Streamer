$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$InstallScript = Join-Path $PSScriptRoot "install.ps1"
$StreamlitExe = Join-Path $ProjectRoot "venv\Scripts\streamlit.exe"

Set-Location $ProjectRoot

& $InstallScript

function Read-DotEnv {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return
    }
    Get-Content $Path | ForEach-Object {
        $Line = $_.Trim()
        if (-not $Line -or $Line.StartsWith("#") -or -not $Line.Contains("=")) {
            return
        }
        $Name, $Value = $Line.Split("=", 2)
        $Name = $Name.Trim()
        $Value = $Value.Trim().Trim('"').Trim("'")
        if ($Name) {
            [Environment]::SetEnvironmentVariable($Name, $Value, "Process")
        }
    }
}

function Find-FluidSynth {
    if ($env:FLUIDSYNTH_EXE -and (Test-Path $env:FLUIDSYNTH_EXE)) {
        return $env:FLUIDSYNTH_EXE
    }

    $PathCommand = Get-Command fluidsynth.exe -ErrorAction SilentlyContinue
    if ($PathCommand) {
        return $PathCommand.Source
    }

    $LocalMatch = Get-ChildItem -Path (Join-Path $ProjectRoot "tools") -Filter fluidsynth.exe -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($LocalMatch) {
        return $LocalMatch.FullName
    }

    return $null
}

function Find-SoundFont {
    if ($env:TINYMOZART_SF2 -and (Test-Path $env:TINYMOZART_SF2)) {
        return $env:TINYMOZART_SF2
    }

    $SoundFontMatch = Get-ChildItem -Path (Join-Path $ProjectRoot "soundfonts") -Include *.sf2,*.sf3 -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($SoundFontMatch) {
        return $SoundFontMatch.FullName
    }

    return $null
}

Read-DotEnv (Join-Path $ProjectRoot ".env")

$FluidSynthExe = Find-FluidSynth
$SoundFont = Find-SoundFont

if ($FluidSynthExe) {
    $env:FLUIDSYNTH_EXE = $FluidSynthExe
    Write-Host "Using FluidSynth: $FluidSynthExe"
} else {
    Write-Host "FluidSynth not found, using pygame MIDI fallback."
}

if ($SoundFont) {
    $env:TINYMOZART_SF2 = $SoundFont
    Write-Host "Using SoundFont: $SoundFont"
} else {
    Write-Host "SoundFont not found, using pygame MIDI fallback."
}

Write-Host "Launching TinyMozart Streamer..."
& $StreamlitExe run app.py --server.address localhost --server.port 8501
