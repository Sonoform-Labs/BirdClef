$ErrorActionPreference = 'Continue'
$env:HF_HOME = 'D:\hf_cache'; $env:TMP = 'D:\tmp'; $env:TEMP = 'D:\tmp'
$py     = 'D:\Python\BirdClef\.venv\Scripts\python.exe'
$script = 'D:\Python\BirdClef\paper_analysis\train_joint.py'
$agg    = 'D:\Python\BirdClef\paper_analysis\gate1_5fold_aggregate.py'
$j      = 'D:\Python\BirdClef\paper_analysis\joint'
$log    = 'D:\Python\BirdClef\OVERNIGHT_LOG.md'
Add-Content -Path $log -Encoding utf8 -Value "`n## P6 RECOVERY (per-fold orchestration, folds 1-4; fold 0 banked). bf16, batch 32, retry-once. start $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Write-Output "[orch] start $(Get-Date -Format 'HH:mm:ss')"

foreach ($fold in 1,2,3,4) {
  $marker = Join-Path $j "DONE_fold$fold.txt"
  if (Test-Path $marker) { Write-Output "[orch] fold $fold already done, skip"; continue }
  $ok = $false
  foreach ($attempt in 1,2) {
    Write-Output "[orch] fold $fold attempt $attempt at $(Get-Date -Format 'HH:mm:ss')"
    $out = Join-Path $j "fold$fold.a$attempt.out.log"
    $err = Join-Path $j "fold$fold.a$attempt.err.log"
    Start-Process -FilePath $py -ArgumentList '-u', $script, '--folds', "$fold", '--epochs', '6', '--batch', '32' `
        -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -Wait
    if (Test-Path $marker) { Write-Output "[orch] fold $fold DONE on attempt $attempt"; $ok = $true; break }
    Write-Output "[orch] fold $fold attempt $attempt DIED (no marker). err tail:"
    Get-Content $err -Tail 10 -ErrorAction SilentlyContinue | ForEach-Object { Write-Output "    $_" }
  }
  if (-not $ok) {
    Write-Output "[orch] fold $fold FAILED x2 -> continuing to next fold"
    Add-Content -Path $log -Encoding utf8 -Value "  [orch] fold $fold FAILED x2 (see fold$fold.a*.err.log)"
  }
}

Write-Output "[orch] aggregating $(Get-Date -Format 'HH:mm:ss')"
Start-Process -FilePath $py -ArgumentList '-u', $agg `
    -RedirectStandardOutput (Join-Path $j 'aggregate.out.log') -RedirectStandardError (Join-Path $j 'aggregate.err.log') -NoNewWindow -Wait
"done" | Out-File (Join-Path $j 'ORCH_DONE.txt')
Write-Output "[orch] complete $(Get-Date -Format 'HH:mm:ss')"
