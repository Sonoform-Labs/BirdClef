$ErrorActionPreference = 'Continue'
$env:HF_HOME='D:\hf_cache'; $env:TMP='D:\tmp'; $env:TEMP='D:\tmp'
$py='D:\Python\BirdClef\.venv\Scripts\python.exe'
$script='D:\Python\BirdClef\paper_analysis\train_faithful.py'
$agg='D:\Python\BirdClef\paper_analysis\gate1_faithful_aggregate.py'
$j='D:\Python\BirdClef\paper_analysis\joint_faithful'
$log='D:\Python\BirdClef\OVERNIGHT_LOG.md'
New-Item -ItemType Directory -Force $j | Out-Null
Add-Content -Path $log -Encoding utf8 -Value "`n## FAITHFUL 5-FOLD RUN START (lifted AudioProtoPNet head; AsymLoss+cluster+sep+orth; bf16 b32 6ep) $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
Write-Output "[faith-orch] start $(Get-Date -Format 'HH:mm:ss')"
foreach($fold in 0,1,2,3,4){
  $marker = Join-Path $j "DONE_fold$fold.txt"
  if(Test-Path $marker){ Write-Output "[faith-orch] fold $fold already done, skip"; continue }
  $ok=$false
  foreach($attempt in 1,2){
    Write-Output "[faith-orch] fold $fold attempt $attempt $(Get-Date -Format 'HH:mm:ss')"
    $out=Join-Path $j "fold$fold.a$attempt.out.log"; $err=Join-Path $j "fold$fold.a$attempt.err.log"
    Start-Process -FilePath $py -ArgumentList '-u',$script,'--folds',"$fold",'--epochs','6','--batch','32' `
        -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -Wait
    if(Test-Path $marker){ Write-Output "[faith-orch] fold $fold DONE attempt $attempt"; $ok=$true; break }
    Write-Output "[faith-orch] fold $fold attempt $attempt DIED. err tail:"; Get-Content $err -Tail 10 -EA SilentlyContinue | ForEach-Object { Write-Output "    $_" }
  }
  if(-not $ok){ Write-Output "[faith-orch] fold $fold FAILED x2, continuing"; Add-Content -Path $log -Encoding utf8 -Value "  [faith-orch] fold $fold FAILED x2" }
}
Write-Output "[faith-orch] aggregating $(Get-Date -Format 'HH:mm:ss')"
Start-Process -FilePath $py -ArgumentList '-u',$agg -RedirectStandardOutput (Join-Path $j 'aggregate.out.log') -RedirectStandardError (Join-Path $j 'aggregate.err.log') -NoNewWindow -Wait
"done" | Out-File (Join-Path $j 'FAITH_ORCH_DONE.txt')
Write-Output "[faith-orch] complete $(Get-Date -Format 'HH:mm:ss')"
