# Installs the Claude Code configuration into .claude/
# Run once from the project root:  powershell -ExecutionPolicy Bypass -File claude-config\install.ps1

$root = Split-Path -Parent $PSScriptRoot
$src  = Join-Path $root "claude-config"
$dst  = Join-Path $root ".claude"

New-Item -ItemType Directory -Force -Path (Join-Path $dst "commands") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $dst "agents")   | Out-Null

Copy-Item (Join-Path $src "settings.json") $dst -Force
Copy-Item (Join-Path $src "commands\*.md")  (Join-Path $dst "commands") -Force
Copy-Item (Join-Path $src "agents\*.md")    (Join-Path $dst "agents")   -Force

Write-Host "Installed into $dst"
Get-ChildItem -Recurse $dst | Select-Object FullName
Write-Host "`nNow run:  claude   then type /phase 0"
