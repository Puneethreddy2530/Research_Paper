# run_after_cec.ps1
# ==================
# Run this in a NEW terminal window.
# It waits for run_cec2017_parallel.py to finish, then automatically
# runs convergence + overnight sequentially.
#
# Usage (from quantum_benchmark/ dir):
#   powershell -ExecutionPolicy Bypass -File run_after_cec.ps1

$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $dir

$sentinel = Join-Path $dir "results\cec2017_results_D30.csv"

Write-Host ""
Write-Host "============================================================"
Write-Host "  AUTO-CHAIN: Will run after CEC2017 finishes"
Write-Host "  Watching for: results\cec2017_results_D30.csv"
Write-Host "  Then will run:"
Write-Host "    1) experiments\run_convergence_parallel.py"
Write-Host "    2) run_overnight_parallel.py"
Write-Host "============================================================"
Write-Host ""

# ── Wait for CEC to finish ────────────────────────────────────
Write-Host "[$(Get-Date -Format 'HH:mm:ss')]  Waiting for CEC2017 to complete..."

while (-not (Test-Path $sentinel)) {
    Start-Sleep -Seconds 60
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')]  Still waiting... (CEC not done yet)"
}

Write-Host ""
Write-Host "[$(Get-Date -Format 'HH:mm:ss')]  CEC2017 DONE! Starting convergence..."
Write-Host "============================================================"

# ── Step 1: Convergence ───────────────────────────────────────
$t1 = Get-Date
python experiments/run_convergence_parallel.py
$elapsed1 = ((Get-Date) - $t1).TotalMinutes
Write-Host ""
Write-Host "[$(Get-Date -Format 'HH:mm:ss')]  Convergence done in $([math]::Round($elapsed1,1)) min"
Write-Host "============================================================"

# ── Step 2: Overnight (Track 2 + Track 3) ────────────────────
Write-Host "[$(Get-Date -Format 'HH:mm:ss')]  Starting overnight run (Tracks 2 & 3)..."
$t2 = Get-Date
python run_overnight_parallel.py
$elapsed2 = ((Get-Date) - $t2).TotalMinutes
Write-Host ""
Write-Host "[$(Get-Date -Format 'HH:mm:ss')]  Overnight done in $([math]::Round($elapsed2,1)) min"

# ── All Done ─────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================"
Write-Host "  ALL STAGES COMPLETE!"
Write-Host "  Results in: results/"
Write-Host "============================================================"
