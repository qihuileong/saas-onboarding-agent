# SessionStart hook - inject the newest session summary so a new Claude Code
# session knows where the last one left off. Outputs Claude Code hook JSON.
# NOTE: keep this file ASCII-only; Windows PowerShell 5.1 mis-parses .ps1
# files that contain characters like a real em-dash.
$ErrorActionPreference = 'Stop'
try {
    $root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
    $sessions = Join-Path $root 'sessions'
    if (-not (Test-Path $sessions)) { exit 0 }

    $latest = Get-ChildItem -Path $sessions -Filter '*.md' -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) { exit 0 }

    $content = Get-Content -Path $latest.FullName -Raw
    if ($null -eq $content) { exit 0 }
    if ($content.Length -gt 6000) { $content = $content.Substring(0, 6000) + "`n...(truncated)" }

    $context = "Notes from the previous session ($($latest.Name)). Pick up from here:`n`n$content"
    $out = @{ hookSpecificOutput = @{ hookEventName = 'SessionStart'; additionalContext = $context } }
    $out | ConvertTo-Json -Depth 5 -Compress
} catch {
    exit 0   # never block a session start
}
