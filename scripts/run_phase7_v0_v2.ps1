# scripts/run_phase7_v0_v2.ps1
#
# Phase 7 grid rows V0 (zero-shot, no retrieval) and V2 ({dense, lexical}, no
# KG): MedQA test n=500, seed 42, code pinned to the grid commit 41f7e14
# (ADR-040). Both are pre-specified grid rows (docs/04 section 0.3) that the
# docs/SCHEDULE.md cut left unrun. Their comparisons form a separate, UNPLANNED
# exploratory family (docs/04 section 1.0b) -- never merged into the m=35
# family.
#
#   cd "C:\Claude\AI Project"
#   .\scripts\run_phase7_v0_v2.ps1 -DryRun    # pin check + print commands only
#   .\scripts\run_phase7_v0_v2.ps1            # run V0 then V2 (~1 LLM call/q each)
#   .\scripts\run_phase7_v0_v2.ps1 -Only V0   # one cell per launch (manifest
#                                             # v0_v2_launch_V0.json)
#
# Pin: refuses to launch unless every run-affecting tracked path (src/,
# configs/, pyproject.toml, uv.lock) is byte-identical to 41f7e14 at HEAD AND
# in the working tree (no staged, unstaged or untracked change under them).
# HEAD itself may be a later commit (docs/scripts-only commits on top of
# 41f7e14); scripts/report.py re-checks the same condition from each run's own
# git provenance before it will tabulate the run.
#
# Sequential; each cell logs to logs/phase7_<cell>_<stamp>.log AND streams to
# console; a failed cell does not abort the other. Writes
# results/phase7_grid/v0_v2_launch.json.
#
# ASCII only (8f37955: the PowerShell 5.1 parser chokes on non-ASCII here).

param(
    [switch]$DryRun,
    [ValidateSet("V0", "V2")][string[]]$Only = @("V0", "V2")
)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"   # ADR-044: stdout reaches the log as it is printed
$env:OLLAMA_KEEP_ALIVE = "-1"

$PinnedCommit = "41f7e14"
$RunPaths = @("src", "configs", "pyproject.toml", "uv.lock")

function Stop-Pin([string]$Msg) {
    Write-Host "[phase7_v0_v2] ABORT (pin check) -- $Msg"
    exit 1
}

# --- pin check ------------------------------------------------------------------
git rev-parse --verify --quiet "$PinnedCommit^{commit}" | Out-Null
if ($LASTEXITCODE -ne 0) { Stop-Pin "commit $PinnedCommit not found in this repository" }

git diff --quiet $PinnedCommit HEAD -- $RunPaths
if ($LASTEXITCODE -ne 0) {
    git diff --stat $PinnedCommit HEAD -- $RunPaths
    Stop-Pin "HEAD run-affecting code differs from $PinnedCommit (stat above)"
}
git diff --quiet HEAD -- $RunPaths
if ($LASTEXITCODE -ne 0) {
    git status --short -- $RunPaths
    Stop-Pin "uncommitted (staged or unstaged) changes under run-affecting paths (above)"
}
$Untracked = @(git ls-files --others --exclude-standard -- $RunPaths)
if ($Untracked.Count -gt 0) {
    Stop-Pin ("untracked files under run-affecting paths -- " + ($Untracked -join ", "))
}

$PinnedShaFull = (git rev-parse $PinnedCommit).Trim()
$LaunchSha = (git rev-parse HEAD).Trim()
$DirtyFiles = @(git status --porcelain | ForEach-Object { $_.Substring(3) })
Write-Host "[phase7_v0_v2] pin check PASSED -- src, configs, pyproject.toml, uv.lock identical to $PinnedShaFull"
Write-Host "[phase7_v0_v2] launch HEAD $LaunchSha (dirty files, none run-affecting = $($DirtyFiles.Count))"

# --- cells ----------------------------------------------------------------------
$H = "uv run python -m src.eval.harness --dataset medqa --split test --limit 500 --seed 42 --config configs/base.yaml"
$AllCells = [ordered]@{
    "V0" = "$H --variant V0"
    "V2" = "$H --variant V2"
}
$Cells = [ordered]@{}
foreach ($name in $AllCells.Keys) { if ($Only -contains $name) { $Cells[$name] = $AllCells[$name] } }
$CellNames = @($Cells.Keys)
$ManifestName = if ($CellNames.Count -eq $AllCells.Count) { "v0_v2_launch.json" } else { "v0_v2_launch_$($CellNames -join '_').json" }

if ($DryRun) {
    Write-Host "[phase7_v0_v2] DRY RUN -- would run, in order"
    foreach ($name in $Cells.Keys) { Write-Host "  [$name]  $($Cells[$name])" }
    exit 0
}

$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$OutDir = Join-Path $RepoRoot "results\phase7_grid"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StartUtc = (Get-Date).ToUniversalTime()
$Results = @()

function Invoke-Cell {
    param([string]$Name, [string]$Command)
    $LogPath = Join-Path $LogDir "phase7_$($Name)_$Stamp.log"
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "[phase7_v0_v2] START  $Name"
    Write-Host "[phase7_v0_v2] cmd:   $Command"
    Write-Host "[phase7_v0_v2] log:   $LogPath"
    Write-Host "========================================================================"
    # ADR-039: stub first, so a zero-stdout failure still leaves a trace.
    "[phase7_v0_v2] STUB   $Name started $((Get-Date).ToString('u'))`ncmd: $Command" |
        Set-Content -Path $LogPath -Encoding utf8
    $start = Get-Date
    # ADR-044: cmd merges the traceback (stderr) into stdout before PowerShell
    # sees it; `Invoke-Expression $Command 2>&1 | Tee-Object` dropped it.
    cmd /c "$Command 2>&1" | Tee-Object -FilePath $LogPath -Append
    $exit = $LASTEXITCODE
    $elapsed = (Get-Date) - $start
    $status = if ($exit -eq 0) { "OK" } else { "FAILED (exit $exit)" }
    Write-Host "[phase7_v0_v2] DONE   $Name -> $status  ($([math]::Round($elapsed.TotalMinutes,1)) min)"
    $script:Results += [pscustomobject]@{
        Cell = $Name; ExitCode = $exit
        Minutes = [math]::Round($elapsed.TotalMinutes, 1); Log = $LogPath
    }
}

foreach ($name in $Cells.Keys) { Invoke-Cell -Name $name -Command $Cells[$name] }

# --- run dirs this launch produced ----------------------------------------------
$RunDirs = [ordered]@{}
foreach ($name in $Cells.Keys) {
    $d = Get-ChildItem -Path (Join-Path $RepoRoot "results") -Directory -Filter "medqa_$($name)_*" |
        Where-Object { $_.LastWriteTimeUtc -ge $StartUtc } |
        Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
    $RunDirs[$name] = if ($d) { $d.Name } else { $null }
}

# --- rescore sweep (ADR-037) ----------------------------------------------------
# Relative paths: an absolute path under "AI Project" contains a space and
# would split inside the cmd /c command line.
$pending = @(
    foreach ($name in $RunDirs.Keys) {
        $rd = $RunDirs[$name]
        if (-not $rd) { continue }
        $mp = Join-Path $RepoRoot "results\$rd\metrics.json"
        if ((Test-Path $mp) -and ((Get-Content -Raw $mp | ConvertFrom-Json).metrics.hallucination_pending -eq $true)) {
            "results/$rd"
        }
    }
)
if ($pending.Count -gt 0) {
    Write-Host "[phase7_v0_v2] $($pending.Count) run(s) still hallucination_pending -- rescoring"
    Invoke-Cell -Name "rescore_v0_v2" -Command ("uv run python -m scripts.analysis.rescore_hallucination " + ($pending -join ' '))
} else {
    Write-Host "[phase7_v0_v2] no hallucination_pending runs."
}

# --- manifest -------------------------------------------------------------------
$manifest = [ordered]@{
    phase                       = 7
    cells                       = $CellNames
    purpose                     = "pre-specified grid rows (docs/04 section 0.3) that had not run; their comparisons are the UNPLANNED family (docs/04 section 1.0b, ADR-040), never merged into the m=35 family"
    launched_utc                = $StartUtc.ToString('u')
    pinned_code_commit          = $PinnedShaFull
    pin_check                   = "passed before launch: git diff --quiet $PinnedCommit HEAD, git diff --quiet HEAD (worktree + index), no untracked files -- over src, configs, pyproject.toml, uv.lock"
    launch_head                 = $LaunchSha
    launch_worktree_dirty_files = $DirtyFiles
    dataset                     = "medqa"
    split                       = "test"
    n                           = 500
    seed                        = 42
    commands                    = $Cells
    run_dirs                    = $RunDirs
    cell_outcomes               = $Results
    next_steps                  = @(
        "uv run python -m scripts.analysis.unplanned_v0_v2_family --v0 <V0 run dir> --v2 <V2 run dir> --v5-refit <V5-refit run dir>",
        "uv run python -m scripts.report"
    )
}
$manifestPath = Join-Path $OutDir $ManifestName
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding utf8
Write-Host ""
Write-Host "[phase7_v0_v2] wrote $manifestPath"
$Results | Format-Table -AutoSize

$failed = $Results | Where-Object { $_.ExitCode -ne 0 }
if ($failed) {
    Write-Host "[phase7_v0_v2] $($failed.Count) cell(s) FAILED -- inspect the logs before any analysis."
    exit 1
}
Write-Host "[phase7_v0_v2] all cells OK: $(($RunDirs.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join '  ')"
Write-Host "  when V0, V2 and V5-refit have all run:"
Write-Host "  uv run python -m scripts.analysis.unplanned_v0_v2_family --v0 <V0 run dir> --v2 <V2 run dir> --v5-refit <V5-refit run dir>"
Write-Host "  uv run python -m scripts.report"
exit 0
