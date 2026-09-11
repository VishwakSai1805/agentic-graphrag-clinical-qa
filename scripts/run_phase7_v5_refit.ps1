# scripts/run_phase7_v5_refit.ps1
#
# V5-refit (ADR-043): the disclosed post-hoc correction of V5's calibrated
# ranking, applied to the grid V5 run's saved SupportMatrices (MedQA test,
# n=500, seed 42). Re-scoring only -- zero LLM calls, zero GPU, seconds. Its
# tests are the UNPLANNED family's U6-U8 (docs/04 section 1.0b), never merged
# into the m=35 family.
#
#   cd "C:\Claude\AI Project"
#   .\scripts\run_phase7_v5_refit.ps1 -DryRun   # every check, then stop
#   .\scripts\run_phase7_v5_refit.ps1
#
# Refuses to launch unless:
#   1. src/, configs/, pyproject.toml, uv.lock are byte-identical to 41f7e14
#      at HEAD and in the working tree -- the same pin as
#      run_phase7_v0_v2.ps1, so the imported src.calibration code is the
#      grid's own.
#   2. the frozen spec (results/phase7_v5_refit/refit_spec.json) and every
#      script the rule runs through are committed at HEAD and unmodified:
#      the rule that runs is the rule that was registered.
#   3. ADR-043 is in docs/05_DECISIONS.md and U6 in docs/04_EVALUATION.md at
#      HEAD -- registered before the run, verifiably.
#   4. no V5-refit run exists yet. One shot: a re-run is deterministic and
#      would only duplicate the row.
#
# Logs to logs/phase7_V5-refit_<stamp>.log AND the console. Writes
# results/phase7_grid/v5_refit_launch.json.
#
# ASCII only (8f37955: the PowerShell 5.1 parser chokes on non-ASCII here).

param([switch]$DryRun)

$ErrorActionPreference = "Continue"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $RepoRoot
$env:PYTHONIOENCODING = "utf-8"

$PinnedCommit = "41f7e14"
$RunPaths = @("src", "configs", "pyproject.toml", "uv.lock")
$SpecPath = "results/phase7_v5_refit/refit_spec.json"
$RuleFiles = @(
    $SpecPath,
    "scripts/analysis/v5_refit.py",
    "scripts/analysis/spos_consistency.py",
    "scripts/analysis/exploratory_family.py",
    "scripts/analysis/posthoc_binary.py",
    "scripts/fit_calibration.py",
    "scripts/report.py"
)

function Stop-Check([string]$Msg) {
    Write-Host "[phase7_v5_refit] ABORT -- $Msg"
    exit 1
}

# --- 1. code pin ----------------------------------------------------------------
git rev-parse --verify --quiet "$PinnedCommit^{commit}" | Out-Null
if ($LASTEXITCODE -ne 0) { Stop-Check "commit $PinnedCommit not found in this repository" }
git diff --quiet $PinnedCommit HEAD -- $RunPaths
if ($LASTEXITCODE -ne 0) {
    git diff --stat $PinnedCommit HEAD -- $RunPaths
    Stop-Check "HEAD run-affecting code differs from $PinnedCommit (stat above)"
}
git diff --quiet HEAD -- $RunPaths
if ($LASTEXITCODE -ne 0) {
    git status --short -- $RunPaths
    Stop-Check "uncommitted (staged or unstaged) changes under run-affecting paths (above)"
}
$Untracked = @(git ls-files --others --exclude-standard -- $RunPaths)
if ($Untracked.Count -gt 0) {
    Stop-Check ("untracked files under run-affecting paths -- " + ($Untracked -join ", "))
}

# --- 2. frozen spec + rule code committed and unmodified --------------------------
foreach ($f in $RuleFiles) {
    git ls-files --error-unmatch -- $f 2>$null | Out-Null
    if ($LASTEXITCODE -ne 0) { Stop-Check "$f is not committed -- commit the registration first" }
    git diff --quiet HEAD -- $f
    if ($LASTEXITCODE -ne 0) { Stop-Check "$f differs from HEAD -- the registered rule must run unmodified" }
}
$SpecBlob = (git rev-parse "HEAD:$SpecPath").Trim()
$SpecSha = (uv run python -c "import json; print(json.load(open('$SpecPath', encoding='utf-8'))['spec_sha256'])").Trim()

# --- 3. registration committed ---------------------------------------------------
git grep -q "ADR-043" HEAD -- docs/05_DECISIONS.md
if ($LASTEXITCODE -ne 0) { Stop-Check "ADR-043 is not in docs/05_DECISIONS.md at HEAD" }
git grep -q "U6" HEAD -- docs/04_EVALUATION.md
if ($LASTEXITCODE -ne 0) { Stop-Check "U6 is not registered in docs/04_EVALUATION.md at HEAD" }

# --- 4. one shot -----------------------------------------------------------------
$Existing = @(Get-ChildItem -Path (Join-Path $RepoRoot "results") -Directory -Filter "medqa_V5-refit_*")
if ($Existing.Count -gt 0) {
    Stop-Check ("a V5-refit run already exists: " + (($Existing | ForEach-Object { $_.Name }) -join ", "))
}

$PinnedShaFull = (git rev-parse $PinnedCommit).Trim()
$LaunchSha = (git rev-parse HEAD).Trim()
$DirtyFiles = @(git status --porcelain | ForEach-Object { $_.Substring(3) })
Write-Host "[phase7_v5_refit] all checks PASSED"
Write-Host "  code pin:  src, configs, pyproject.toml, uv.lock identical to $PinnedShaFull"
Write-Host "  spec:      $SpecPath blob $SpecBlob, spec_sha256 $SpecSha"
Write-Host "  launch HEAD $LaunchSha (dirty files, none run-affecting or rule = $($DirtyFiles.Count))"

$Command = "uv run python -m scripts.analysis.v5_refit apply"
if ($DryRun) {
    Write-Host "[phase7_v5_refit] DRY RUN -- would run: $Command"
    exit 0
}

# --- run -------------------------------------------------------------------------
$LogDir = Join-Path $RepoRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$OutDir = Join-Path $RepoRoot "results\phase7_grid"
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$StartUtc = (Get-Date).ToUniversalTime()
$LogPath = Join-Path $LogDir "phase7_V5-refit_$Stamp.log"

Write-Host "========================================================================"
Write-Host "[phase7_v5_refit] START  V5-refit"
Write-Host "[phase7_v5_refit] cmd:   $Command"
Write-Host "[phase7_v5_refit] log:   $LogPath"
Write-Host "========================================================================"
# ADR-039: stub first, so a zero-stdout failure still leaves a trace.
"[phase7_v5_refit] STUB   V5-refit started $((Get-Date).ToString('u'))`ncmd: $Command" |
    Set-Content -Path $LogPath -Encoding utf8
$start = Get-Date
Invoke-Expression $Command 2>&1 | Tee-Object -FilePath $LogPath -Append
$exit = $LASTEXITCODE
$minutes = [math]::Round(((Get-Date) - $start).TotalMinutes, 1)

$RunDir = Get-ChildItem -Path (Join-Path $RepoRoot "results") -Directory -Filter "medqa_V5-refit_*" |
    Where-Object { $_.LastWriteTimeUtc -ge $StartUtc } |
    Sort-Object LastWriteTimeUtc -Descending | Select-Object -First 1
$RunDirName = if ($RunDir) { $RunDir.Name } else { $null }

$manifest = [ordered]@{
    phase                       = 7
    cell                        = "V5-refit"
    purpose                     = "disclosed post-hoc correction of V5's scoring rule (ADR-043); tests are the UNPLANNED family's U6-U8 (docs/04 section 1.0b), never merged into the m=35 family"
    launched_utc                = $StartUtc.ToString('u')
    pinned_code_commit          = $PinnedShaFull
    pin_check                   = "passed before launch: git diff --quiet $PinnedCommit HEAD, git diff --quiet HEAD (worktree + index), no untracked files -- over src, configs, pyproject.toml, uv.lock"
    rule_files_committed        = $RuleFiles
    refit_spec                  = $SpecPath
    refit_spec_git_blob         = $SpecBlob
    refit_spec_sha256           = $SpecSha
    launch_head                 = $LaunchSha
    launch_worktree_dirty_files = $DirtyFiles
    command                     = $Command
    run_dir                     = $RunDirName
    exit_code                   = $exit
    minutes                     = $minutes
    log                         = $LogPath
    next_steps                  = @(
        "uv run python -m scripts.analysis.unplanned_v0_v2_family --v0 <V0 run dir> --v2 <V2 run dir> --v5-refit $RunDirName",
        "uv run python -m scripts.report"
    )
}
$manifestPath = Join-Path $OutDir "v5_refit_launch.json"
$manifest | ConvertTo-Json -Depth 6 | Set-Content -Path $manifestPath -Encoding utf8
Write-Host "[phase7_v5_refit] wrote $manifestPath"

if ($exit -ne 0) {
    Write-Host "[phase7_v5_refit] FAILED (exit $exit) -- inspect $LogPath before any analysis."
    exit 1
}
Write-Host "[phase7_v5_refit] OK -> $RunDirName  ($minutes min). Next: V0, then V2."
exit 0
