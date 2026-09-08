# scripts/run_phase7_grid.ps1
#
# Phase 7 ablation grid -- MedQA test n=500, seed 42 (docs/SCHEDULE.md cut +
# ADR-035/036). Sequential; no concurrency layer (ADR-029). Each cell logs to
# logs/phase7_<cell>_<stamp>.log AND streams to console; a failed cell does not
# abort the rest. Writes results/phase7_grid/manifest.json recording the launch
# commit SHA (per the launch instruction) and per-cell outcomes.
#
#   cd "C:\Claude\AI Project"
#   .\scripts\run_phase7_grid.ps1
#
# Cells, in run order (essentials first; V4-conflict last so it is the one that
# gets cut if the machine runs out of overnight -- ADR-035 budget contingency):
#   V1  = base.yaml,           --variant V1           (~1 call/q,  fast)
#   V3  = base.yaml,           --variant V3           (~1 call/q,  fast)
#   B3  = run_baselines.py --baselines B3             (~3-5 call/q, the prior-work comparison)
#   V5  = v5.yaml                                     (~5-6 call/q, agents + calibration)
#   V4-conflict = v4_no_conflict.yaml --variant V4-conflict  (~5-6 call/q; drop first if needed)
#
# V4-audit is deferred (ADR-036 -- needs the binary-audit prompt). The post-hoc
# binary variant is derived from the V5 run afterwards (zero GPU).

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$env:PYTHONIOENCODING = "utf-8"
$env:OLLAMA_KEEP_ALIVE = "-1"

$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$OutDir = Join-Path $RepoRoot "results\phase7_grid"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LaunchSha = (git rev-parse HEAD).Trim()
$LaunchShaShort = (git rev-parse --short HEAD).Trim()
$DirtyFiles = @(git status --porcelain | ForEach-Object { $_.Substring(3) })
$Results = @()

Write-Host "[phase7_grid] repo: $RepoRoot"
Write-Host "[phase7_grid] launch commit: $LaunchSha (dirty files: $($DirtyFiles.Count))"
Write-Host "[phase7_grid] start: $((Get-Date).ToString('u'))"

function Invoke-Cell {
    param([string]$Name, [string]$Command)
    $LogPath = Join-Path $LogDir "phase7_$($Name)_$Stamp.log"
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "[phase7_grid] START  $Name"
    Write-Host "[phase7_grid] cmd:   $Command"
    Write-Host "[phase7_grid] log:   $LogPath"
    Write-Host "========================================================================"
    $start = Get-Date
    Invoke-Expression $Command 2>&1 | Tee-Object -FilePath $LogPath
    $exit = $LASTEXITCODE
    $elapsed = (Get-Date) - $start
    $status = if ($exit -eq 0) { "OK" } else { "FAILED (exit $exit)" }
    Write-Host "[phase7_grid] DONE   $Name -> $status  ($([math]::Round($elapsed.TotalMinutes,1)) min)"
    $script:Results += [pscustomobject]@{
        Cell = $Name; ExitCode = $exit
        Minutes = [math]::Round($elapsed.TotalMinutes, 1); Log = $LogPath
    }
}

$H = "uv run python -m src.eval.harness --dataset medqa --split test --limit 500"
$B = "uv run python -m scripts.run_baselines --dataset medqa --split test --limit 500"

Invoke-Cell -Name "V1"          -Command "$H --config configs/base.yaml --variant V1"
Invoke-Cell -Name "V3"          -Command "$H --config configs/base.yaml --variant V3"
Invoke-Cell -Name "B3"          -Command "$B --baselines B3"
Invoke-Cell -Name "V5"          -Command "$H --config configs/v5.yaml"
Invoke-Cell -Name "V4-conflict" -Command "$H --config configs/v4_no_conflict.yaml --variant V4-conflict"

# --- rescore sweep (ADR-037) ------------------------------------------------
# The harness scores hallucination in-process after freeing the retrieval
# models. If a cell's NLI pass still OOM'd, its metrics.json reads
# hallucination_pending:true and the run is otherwise complete -- finish it
# here (NLI model only, no retrieval models, no LLM calls).
Write-Host ""
Write-Host "[phase7_grid] rescore sweep for hallucination_pending runs ..."
$pending = @(
    Get-ChildItem -Path "$RepoRoot\results" -Directory -Filter "medqa_*_$LaunchShaShort*" |
    Where-Object {
        $mp = Join-Path $_.FullName "metrics.json"
        (Test-Path $mp) -and ((Get-Content -Raw $mp | ConvertFrom-Json).metrics.hallucination_pending -eq $true)
    } | ForEach-Object { $_.FullName }
)
if ($pending.Count -gt 0) {
    Write-Host "[phase7_grid]   $($pending.Count) run(s) still pending: $($pending -join ', ')"
    Invoke-Cell -Name "rescore" -Command ("uv run python -m scripts.analysis.rescore_hallucination " + ($pending -join ' '))
} else {
    Write-Host "[phase7_grid]   none pending -- every cell scored hallucination in-process."
}

# --- manifest -----------------------------------------------------------------
$manifest = [ordered]@{
    phase                 = 7
    launched_utc          = (Get-Date).ToString('u')
    launch_commit         = $LaunchSha
    launch_commit_short   = $LaunchShaShort
    launch_worktree_dirty = ($DirtyFiles.Count -gt 0)
    launch_dirty_files    = $DirtyFiles
    dataset               = "medqa"
    split                 = "test"
    n                     = 500
    seed                  = 42
    cells_run             = @("V1", "V3", "B3", "V5", "V4-conflict")
    cells_deferred        = @("V4-audit (ADR-036)")
    derived_analyses      = @("post-hoc-binary from the V5 run (ADR-036 Constraint B)")
    pre_registration      = "docs/04_EVALUATION.md section 1.0 (m=35), committed before this run"
    cell_outcomes         = $Results
}
$manifestPath = Join-Path $OutDir "manifest.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding utf8
Write-Host ""
Write-Host "[phase7_grid] wrote $manifestPath"

Write-Host ""
Write-Host "========================================================================"
Write-Host "[phase7_grid] SUMMARY  $((Get-Date).ToString('u'))"
Write-Host "========================================================================"
$Results | Format-Table -AutoSize
$failed = $Results | Where-Object { $_.ExitCode -ne 0 }
if ($failed) {
    Write-Host "[phase7_grid] $($failed.Count) cell(s) FAILED -- inspect the logs before running scripts.report or trusting any number."
    exit 1
} else {
    Write-Host "[phase7_grid] all cells OK. Next: derive the post-hoc-binary analysis, then scripts.report."
    exit 0
}
