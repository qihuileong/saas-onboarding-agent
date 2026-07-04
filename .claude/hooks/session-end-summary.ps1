# SessionEnd hook - write a dated record of this session into ./sessions/
# so the next SessionStart can pick up where we left off.
# NOTE: keep this file ASCII-only (Windows PowerShell 5.1 mis-parses non-ASCII).
#
# Default is a factual raw excerpt of the conversation (fast, no LLM, no
# hallucination). Set $UseOllama = $true to instead ask local Ollama
# (qwen2.5:7b) for a bullet summary -- but note that model tends to
# fabricate details, so the raw excerpt is the trustworthy default.
$UseOllama = $false

$ErrorActionPreference = 'Stop'
try {
    $raw = [Console]::In.ReadToEnd()
    $data = $null
    if ($raw) { try { $data = $raw | ConvertFrom-Json } catch { } }

    $root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
    $sessions = Join-Path $root 'sessions'
    if (-not (Test-Path $sessions)) { New-Item -ItemType Directory -Path $sessions | Out-Null }

    $stamp = Get-Date -Format 'yyyy-MM-dd_HHmmss'
    $outFile = Join-Path $sessions "session_$stamp.md"

    # Extract readable text from the transcript (JSONL - one message per line).
    $convo = ''
    $tpath = if ($data -and $data.transcript_path) { $data.transcript_path } else { $null }
    if ($tpath -and (Test-Path $tpath)) {
        $lines = Get-Content -Path $tpath -ErrorAction SilentlyContinue | Select-Object -Last 60
        $sb = New-Object System.Text.StringBuilder
        foreach ($line in $lines) {
            try {
                $msg = $line | ConvertFrom-Json
                $role = $msg.message.role
                $c = $msg.message.content
                if (-not $role -or -not $c) { continue }
                $text = ''
                if ($c -is [string]) { $text = $c }
                else { foreach ($b in $c) { if ($b.type -eq 'text') { $text += "$($b.text) " } } }
                $text = $text.Trim()
                if ($text) { [void]$sb.AppendLine("[$role] $text") }
            } catch { }
        }
        $convo = $sb.ToString()
    }
    if (-not $convo) { $convo = '(No transcript text was available for this session.)' }
    if ($convo.Length -gt 8000) { $convo = $convo.Substring($convo.Length - 8000) }

    # Opt-in: summarize with local Ollama; fall back to the raw excerpt.
    $summary = $null
    if ($UseOllama -and (Get-Command ollama -ErrorAction SilentlyContinue)) {
        try {
            $prompt = "Summarize this coding session in up to 12 bullet points: what we worked on, decisions made, current state, and clear next steps. Use ONLY facts present below; do not invent anything.`n`n$convo"
            $summary = ($prompt | & ollama run qwen2.5:7b) -join "`n"
            if ($summary) {
                # strip ANSI/terminal control codes ollama may emit
                $summary = ($summary -replace "\x1b\[[0-9;?]*[a-zA-Z]", '').Trim()
            }
        } catch { $summary = $null }
    }

    $header = "# Session $stamp`n`n"
    if ($summary) {
        Set-Content -Path $outFile -Value ($header + $summary) -Encoding UTF8
    } else {
        Set-Content -Path $outFile -Value ($header + "## Recent conversation (raw excerpt)`n`n" + $convo) -Encoding UTF8
    }
    exit 0
} catch {
    exit 0   # never block session shutdown
}
