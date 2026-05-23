# Seed-variance armor for Gate 1: fold 0, 3 seeds, both heads. Per-seed process isolation + retry.
# Closes the "1 seed/fold" objection by showing within-fold seed variance << the -0.027 effect.
$ErrorActionPreference = 'Continue'
$env:HF_HOME='D:\hf_cache'; $env:TMP='D:\tmp'; $env:TEMP='D:\tmp'
$py='D:\Python\BirdClef\.venv\Scripts\python.exe'
$script='D:\Python\BirdClef\paper_analysis\train_joint.py'
$j='D:\Python\BirdClef\paper_analysis\joint'
Write-Output "[seed-orch] start $(Get-Date -Format 'HH:mm:ss')"
foreach($seed in 101,202,303){
  $marker = Join-Path $j "DONE_fold0_s$seed.txt"
  if(Test-Path $marker){ Write-Output "[seed-orch] seed $seed already done, skip"; continue }
  $ok=$false
  foreach($attempt in 1,2){
    Write-Output "[seed-orch] seed $seed attempt $attempt $(Get-Date -Format 'HH:mm:ss')"
    $out=Join-Path $j "seed$seed.a$attempt.out.log"; $err=Join-Path $j "seed$seed.a$attempt.err.log"
    Start-Process -FilePath $py -ArgumentList '-u',$script,'--folds','0','--seed',"$seed",'--save-suffix',"_s$seed",'--epochs','6','--batch','32' `
        -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -Wait
    if(Test-Path $marker){ Write-Output "[seed-orch] seed $seed DONE attempt $attempt"; $ok=$true; break }
    Write-Output "[seed-orch] seed $seed attempt $attempt DIED. err tail:"; Get-Content $err -Tail 8 -EA SilentlyContinue | ForEach-Object { Write-Output "    $_" }
  }
  if(-not $ok){ Write-Output "[seed-orch] seed $seed FAILED x2, continuing" }
}
Start-Process -FilePath $py -ArgumentList '-u','D:\Python\BirdClef\paper_analysis\gate1_seed_analysis.py' `
    -RedirectStandardOutput (Join-Path $j 'seed_analysis.out.log') -RedirectStandardError (Join-Path $j 'seed_analysis.err.log') -NoNewWindow -Wait
"done" | Out-File (Join-Path $j 'SEED_DONE.txt')
Write-Output "[seed-orch] complete $(Get-Date -Format 'HH:mm:ss')"
