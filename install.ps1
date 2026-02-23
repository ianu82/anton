#Requires -Version 5.1
<#
.SYNOPSIS
    Install Anton — autonomous AI coworker.
.DESCRIPTION
    Downloads and installs Anton via uv tool install.
    Run: irm https://raw.githubusercontent.com/mindsdb/anton/main/install.ps1 | iex
#>

$ErrorActionPreference = "Stop"

# ── 1. Branded logo ────────────────────────────────────────────────
Write-Host ""
Write-Host "  ▄▀█ █▄ █ ▀█▀ █▀█ █▄ █" -ForegroundColor Cyan
Write-Host "  █▀█ █ ▀█  █  █▄█ █ ▀█" -ForegroundColor Cyan
Write-Host "  autonomous coworker" -ForegroundColor Cyan
Write-Host ""

# ── 2. Check prerequisites ──────────────────────────────────────────
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "error: git is required but not found." -ForegroundColor Red
    Write-Host "  Install it with:  winget install Git.Git"
    Write-Host "  Or download from: https://git-scm.com/downloads/win"
    exit 1
}

# ── 3. Find or install uv ──────────────────────────────────────────
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue

if (-not $uvCmd) {
    $localUv = Join-Path $HOME ".local\bin\uv.exe"
    $cargoUv = Join-Path $HOME ".cargo\bin\uv.exe"

    if (Test-Path $localUv) {
        $uvBinDir = Join-Path $HOME ".local\bin"
        $env:PATH = "$uvBinDir;$env:PATH"
        Write-Host "  Found uv: $localUv"
    }
    elseif (Test-Path $cargoUv) {
        $cargoBinDir = Join-Path $HOME ".cargo\bin"
        $env:PATH = "$cargoBinDir;$env:PATH"
        Write-Host "  Found uv: $cargoUv"
    }
    else {
        Write-Host "  Installing uv..."
        & ([scriptblock]::Create((Invoke-RestMethod https://astral.sh/uv/install.ps1)))
        # Refresh PATH to pick up uv
        $uvBinDir = Join-Path $HOME ".local\bin"
        $env:PATH = "$uvBinDir;$env:PATH"
        Write-Host "  Installed uv"
    }
}
else {
    Write-Host "  Found uv: $($uvCmd.Source)"
}

# Verify uv is available
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "error: uv installation failed. Please install uv manually:" -ForegroundColor Red
    Write-Host "  https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}

# ── 4. Install anton via uv tool ───────────────────────────────────
Write-Host "  Installing anton..."
uv tool install "git+https://github.com/mindsdb/anton.git" --force
Write-Host "  Installed anton"

# ── 5. Ensure ~/.local/bin is in user PATH ─────────────────────────
$uvBinDir = Join-Path $HOME ".local\bin"
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")

if ($userPath -and ($userPath -split ";" | ForEach-Object { $_.TrimEnd('\') }) -contains $uvBinDir.TrimEnd('\')) {
    # Already in PATH
}
else {
    if ($userPath) {
        $newPath = "$uvBinDir;$userPath"
    }
    else {
        $newPath = $uvBinDir
    }
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    $env:PATH = "$uvBinDir;$env:PATH"
    Write-Host "  Added $uvBinDir to user PATH"
}

# ── 6. Success message ──────────────────────────────────────────────
Write-Host ""
Write-Host "  ✓ anton installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "  Open a new terminal, then:"
Write-Host ""
Write-Host "    anton                                          " -NoNewline
Write-Host "# Dashboard" -ForegroundColor Cyan
Write-Host "    anton run `"analyze last month's sales data`"    " -NoNewline
Write-Host "# Give Anton a task" -ForegroundColor Cyan
Write-Host ""
Write-Host "  Upgrade:    uv tool upgrade anton"
Write-Host "  Uninstall:  uv tool uninstall anton"
Write-Host ""
Write-Host "  Config: ~\.anton\.env"
Write-Host ""
