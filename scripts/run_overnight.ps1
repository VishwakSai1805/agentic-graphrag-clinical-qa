# scripts/run_overnight.ps1
#
# Sequential overnight batch unlocked by today's session's four code changes:
#   1. AUDIT 2026-09-06 SS1.1/SS1.2 -- nfcorpus_significance.py and
#      eval_retrieval.py now write results/phase1_*/metrics.json instead of
#      stdout-only. This job regenerates all of it for real.
#   2. AUDIT SS5.4 -- pubmedqa is now wired into scripts/run_baselines.py
#      --dataset. This job smoke-tests that path for real.
#   3. AUDIT SS4.3 -- scripts/run_baselines.py --equal-budget trims B2/B3's
#      evidence bundle to agents.max_evidence via the same modality-reserved
#      split V4/V5 use. This job runs the equal-budget B2 comparator the
#      docs/04_EVALUATION.md SS5.5 finding said must exist before its gap is
#      read as anything.
#
# Nothing here needs a Claude Code session -- every job is `uv run python -m
# scripts.<name>` exactly as documented in each script's own module
# docstring/PROJECT_RULES.md SS6. Run this from PowerShell:
#
#   cd "C:\Claude\AI Project"
#   .\scripts\run_overnight.ps1
#
# Each job's full stdout+stderr goes to its own logs/<job>_<timestamp>.log
# AND streams to the console. A job failing does not stop the remaining
# jobs -- this runs unattended, so a bad job should not cost the rest of the
# night. Exit codes are collected and printed as a summary at the end.

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot

$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$Results = @()

function Invoke-Job {
    param(
        [string]$Name,
        [string]$Command
    )
    $LogPath = Join-Path $LogDir "$($Name)_$Stamp.log"
    Write-Host ""
    Write-Host "========================================================================"
    Write-Host "[run_overnight] START  $Name"
    Write-Host "[run_overnight] cmd:   $Command"
    Write-Host "[run_overnight] log:   $LogPath"
    Write-Host "========================================================================"
    $start = Get-Date

    # Invoke-Expression so $Command can be a plain string built below;
    # Tee-Object mirrors to the console AND the per-job log file.
    Invoke-Expression $Command 2>&1 | Tee-Object -FilePath $LogPath
    $exitCode = $LASTEXITCODE
    $elapsed = (Get-Date) - $start

    $status = if ($exitCode -eq 0) { "OK" } else { "FAILED (exit $exitCode)" }
    Write-Host "[run_overnight] DONE   $Name -> $status  ($([math]::Round($elapsed.TotalMinutes, 1)) min)"

    $script:Results += [pscustomobject]@{
        Job      = $Name
        ExitCode = $exitCode
        Minutes  = [math]::Round($elapsed.TotalMinutes, 1)
        Log      = $LogPath
    }
}

Write-Host "[run_overnight] repo root: $RepoRoot"
Write-Host "[run_overnight] starting $((Get-Date).ToString('u'))"

# --------------------------------------------------------------------------
# 1. Phase 1 metrics regeneration (AUDIT SS1.1 / SS1.2)
# --------------------------------------------------------------------------

Invoke-Job -Name "phase1_nfcorpus_significance" `
    -Command "uv run python -m scripts.analysis.nfcorpus_significance"

Invoke-Job -Name "phase1_eval_retrieval_nfcorpus" `
    -Command "uv run python -m scripts.eval_retrieval --dataset nfcorpus --limit 323"

Invoke-Job -Name "phase1_eval_retrieval_pubmedqa" `
    -Command "uv run python -m scripts.eval_retrieval --dataset pubmedqa --limit 200"

Invoke-Job -Name "phase1_eval_retrieval_medqa" `
    -Command "uv run python -m scripts.eval_retrieval --dataset medqa --limit 200"

# --------------------------------------------------------------------------
# 2. Equal-budget B2 on the MedQA dev-20 (AUDIT SS4.3)
# --------------------------------------------------------------------------

Invoke-Job -Name "medqa_dev20_B2_equal_budget" `
    -Command "uv run python -m scripts.run_baselines --dataset medqa --split train --limit 20 --baselines B2 --equal-budget"

# --------------------------------------------------------------------------
# 3. PubMedQA baseline smoke run (AUDIT SS5.4 -- wiring pubmedqa in for the
#    first time; default --baselines runs all of B0-B3)
# --------------------------------------------------------------------------

Invoke-Job -Name "pubmedqa_dev20_smoke" `
    -Command "uv run python -m scripts.run_baselines --dataset pubmedqa --split dev --limit 20"

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

Write-Host ""
Write-Host "========================================================================"
Write-Host "[run_overnight] SUMMARY  $((Get-Date).ToString('u'))"
Write-Host "========================================================================"
$Results | Format-Table -AutoSize

$failed = $Results | Where-Object { $_.ExitCode -ne 0 }
if ($failed) {
    Write-Host "[run_overnight] $($failed.Count) job(s) FAILED -- check the logs above before trusting any new results/ artifact."
    exit 1
} else {
    Write-Host "[run_overnight] all jobs OK."
    exit 0
}
