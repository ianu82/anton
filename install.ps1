#Requires -Version 5.1
<#
.SYNOPSIS
    Install Anton — autonomous AI coworker.
.DESCRIPTION
    Downloads and installs Anton via uv tool install into an isolated virtual environment.
    Run: irm https://raw.githubusercontent.com/mindsdb/anton/main/install.ps1 | iex
.PARAMETER Force
    Skip all confirmation prompts.
#>
param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Confirm-Step {
    param([string]$Message)
    if ($Force) { return $true }
    $reply = Read-Host "  $Message [Y/n]"
    return ($reply -eq '' -or $reply -match '^[yY]')
}

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
$needUv = $false

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
        $needUv = $true
    }
}
else {
    Write-Host "  Found uv: $($uvCmd.Source)"
}

if ($needUv) {
    Write-Host "warning: uv is not installed. It is required to manage anton's isolated environment." -ForegroundColor Yellow
    Write-Host "  uv will be installed to ~\.local\bin via https://astral.sh/uv/install.ps1"
    if (Confirm-Step "Install uv?") {
        Write-Host "  Installing uv..."
        & ([scriptblock]::Create((Invoke-RestMethod https://astral.sh/uv/install.ps1)))
        # Refresh PATH to pick up uv
        $uvBinDir = Join-Path $HOME ".local\bin"
        $env:PATH = "$uvBinDir;$env:PATH"
        Write-Host "  Installed uv"
    }
    else {
        Write-Host "error: uv is required. Install it manually: https://docs.astral.sh/uv/getting-started/installation/" -ForegroundColor Red
        exit 1
    }
}

# Verify uv is available
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "error: uv installation failed. Please install uv manually:" -ForegroundColor Red
    Write-Host "  https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}

# ── 4. Install anton via uv tool ───────────────────────────────────
Write-Host ""
Write-Host "  This will install:"
Write-Host "    - anton (from git+https://github.com/mindsdb/anton.git)"
Write-Host "    - Into an isolated virtual environment managed by uv"
Write-Host "    - Python 3.11+ will be downloaded automatically if not present"
Write-Host ""

if (Confirm-Step "Proceed with installation?") {
    Write-Host "  Installing anton into an isolated venv..."
    uv tool install "git+https://github.com/mindsdb/anton.git" --force
    Write-Host "  Installed anton"
}
else {
    Write-Host "  Installation cancelled."
    exit 0
}

# ── 5. Verify the venv was created ─────────────────────────────────
$uvToolDir = Join-Path $HOME ".local\share\uv\tools\anton"
if (Test-Path $uvToolDir) {
    Write-Host "  Venv: $uvToolDir"
}
else {
    Write-Host "warning: Could not verify anton's virtual environment." -ForegroundColor Yellow
    Write-Host "  Expected at: $uvToolDir"
    Write-Host "  Anton may still work — uv manages the environment internally."
}

# ── 6. Configure firewall for scratchpad internet access ──────────
#    Anton's scratchpads run Python in per-scratchpad venvs under
#    ~/.anton/scratchpad-venvs/<name>/Scripts/python.exe.
#    Windows Firewall blocks new executables by default, so we add a
#    wildcard-style rule covering the entire scratchpad-venvs directory.
$scratchpadVenvsDir = Join-Path $HOME ".anton\scratchpad-venvs"
Write-Host ""
Write-Host "  Anton's scratchpads need internet access (for web scraping, APIs, etc.)."
Write-Host "  This adds a Windows Firewall rule allowing Python executables under:"
Write-Host "    $scratchpadVenvsDir"
Write-Host ""

if (Confirm-Step "Allow scratchpad internet access? (requires admin)") {
    # Create the venvs dir so the path exists
    if (-not (Test-Path $scratchpadVenvsDir)) {
        New-Item -ItemType Directory -Path $scratchpadVenvsDir -Force | Out-Null
    }

    # Windows Firewall doesn't support wildcards in program paths, so we
    # allow the entire directory via a rule for each existing venv, plus
    # a broad rule using the uv-managed tool python as a fallback.
    # The most practical approach: allow any python.exe under scratchpad-venvs.
    # We do this by adding a rule per existing venv and documenting how to
    # add rules for new ones. For a fresh install, we add the directory rule.
    try {
        # Remove old single-venv rule if it exists (from previous installs)
        Start-Process -FilePath "netsh" `
            -ArgumentList "advfirewall firewall delete rule name=`"Anton Scratchpad`"" `
            -Verb RunAs -Wait -WindowStyle Hidden -ErrorAction SilentlyContinue

        # Add rules for any existing scratchpad venvs
        $added = 0
        if (Test-Path $scratchpadVenvsDir) {
            Get-ChildItem -Path $scratchpadVenvsDir -Directory | ForEach-Object {
                $pyExe = Join-Path $_.FullName "Scripts\python.exe"
                if (Test-Path $pyExe) {
                    Start-Process -FilePath "netsh" `
                        -ArgumentList "advfirewall firewall add rule name=`"Anton Scratchpad - $($_.Name)`" dir=out action=allow program=`"$pyExe`"" `
                        -Verb RunAs -Wait -WindowStyle Hidden
                    $added++
                }
            }
        }

        # Also add a rule for the uv tool python (used to create venvs)
        $uvToolPython = Join-Path $HOME ".local\share\uv\tools\anton\Scripts\python.exe"
        if (Test-Path $uvToolPython) {
            Start-Process -FilePath "netsh" `
                -ArgumentList "advfirewall firewall add rule name=`"Anton Tool Python`" dir=out action=allow program=`"$uvToolPython`"" `
                -Verb RunAs -Wait -WindowStyle Hidden
        }

        Write-Host "  Firewall rules added ($added scratchpad venvs)" -ForegroundColor Green
        Write-Host "  Note: New scratchpads will create venvs automatically." -ForegroundColor Yellow
        Write-Host "  If a new scratchpad's internet calls time out, re-run this script or add a rule manually:"
        Write-Host "    netsh advfirewall firewall add rule name=`"Anton Scratchpad`" dir=out action=allow program=`"$scratchpadVenvsDir\<name>\Scripts\python.exe`""
    }
    catch {
        Write-Host "  Could not add firewall rules (admin declined or unavailable)." -ForegroundColor Yellow
        Write-Host "  You can add them manually later for each scratchpad:"
        Write-Host "    netsh advfirewall firewall add rule name=`"Anton Scratchpad`" dir=out action=allow program=`"$scratchpadVenvsDir\<name>\Scripts\python.exe`""
    }
}
else {
    Write-Host "  Skipped. If scratchpad internet calls time out, run this in an admin PowerShell:"
    Write-Host "    netsh advfirewall firewall add rule name=`"Anton Scratchpad`" dir=out action=allow program=`"$scratchpadVenvsDir\<name>\Scripts\python.exe`""
}

# ── 7. Ensure ~/.local/bin is in user PATH ──────────────────────────
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

# ── 8. Success message ──────────────────────────────────────────────
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
