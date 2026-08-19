$ErrorActionPreference = 'Continue'
$env:HF_HOME='D:\hf_cache'; $env:TMP='D:\tmp'; $env:TEMP='D:\tmp'; $env:PYTHONUTF8='1'
$py='D:\Python\BirdClef\.venv\Scripts\python.exe'
$script='D:\Python\BirdClef\paper_analysis\train_faithful.py'
$j='D:\Python\BirdClef\paper_analysis\joint_faithful'
$scales=@(@('0.5','_ce05'), @('1.0','_ce10'), @('2.0','_ce20'))
Write-Output "[ce-orch] start $(Get-Date -Format 'HH:mm:ss')"
foreach($sc in $scales){
  $scale=$sc[0]; $sfx=$sc[1]
  $marker = Join-Path $j "DONE_fold0$sfx.txt"
  if(Test-Path $marker){ Write-Output "[ce-orch] scale $scale already done, skip"; continue }
  $ok=$false
  foreach($attempt in 1,2){
    Write-Output "[ce-orch] scale $scale ($sfx) attempt $attempt $(Get-Date -Format 'HH:mm:ss')"
    $out=Join-Path $j "ce$sfx.a$attempt.out.log"; $err=Join-Path $j "ce$sfx.a$attempt.err.log"
    Start-Process -FilePath $py -ArgumentList '-u',$script,'--folds','0','--seed','42','--ce-scale',$scale,'--save-suffix',$sfx,'--epochs','6','--proto-ep1-min','0.75' `
        -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -Wait
    if(Test-Path $marker){ Write-Output "[ce-orch] scale $scale DONE attempt $attempt"; $ok=$true; break }
    Write-Output "[ce-orch] scale $scale attempt $attempt DIED (no marker). err tail:"; Get-Content $err -Tail 8 -EA SilentlyContinue | ForEach-Object { Write-Output "    $_" }
  }
  if(-not $ok){ Write-Output "[ce-orch] scale $scale FAILED x2, continuing" }
}
Write-Output "[ce-orch] analyzing $(Get-Date -Format 'HH:mm:ss')"
Start-Process -FilePath $py -ArgumentList '-u','D:\Python\BirdClef\paper_analysis\ce_sensitivity_analyze.py' `
    -RedirectStandardOutput (Join-Path $j 'ce_analyze.out.log') -RedirectStandardError (Join-Path $j 'ce_analyze.err.log') -NoNewWindow -Wait
'done' | Out-File (Join-Path $j 'CE_SENS_DONE.txt')
Write-Output "[ce-orch] complete $(Get-Date -Format 'HH:mm:ss')"
