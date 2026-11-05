# DSV4.1 single engine, 1M context (999,424 tokens): bf16 context cache, growable.
# Exact-cone prefill keeps its carry in available VRAM; N=1 remains fastest.
# Settings: scripts/dsv41_single.py (preset 1m). Options: -Config, -Model, -WebUI, -DryRun, -Stop.
param([string]$Config = '', [string]$Model = '', [switch]$WebUI, [switch]$Vision, [switch]$DryRun, [switch]$Stop,
      [string]$Python='', [double]$HostBudgetGiB=10.25, [double]$WorkerHostBudgetGiB=10,
      [double]$ProducerHostBudgetGiB=6.25, [string]$ProducerHostMaxGiB='7.75',
      [string]$EngramScanRings='auto', [string]$PrefillSweepTokens='auto', [hashtable]$PerformanceFlags=@{}, [switch]$PerformanceExperiment)
& (Join-Path $PSScriptRoot 'start-dsv41-single.ps1') -Preset 1m -Config $Config -Model $Model -WebUI:$WebUI -Vision:$Vision -DryRun:$DryRun -Stop:$Stop -Python $Python -HostBudgetGiB $HostBudgetGiB -WorkerHostBudgetGiB $WorkerHostBudgetGiB -ProducerHostBudgetGiB $ProducerHostBudgetGiB -ProducerHostMaxGiB $ProducerHostMaxGiB -EngramScanRings $EngramScanRings -PrefillSweepTokens $PrefillSweepTokens -PerformanceFlags $PerformanceFlags -PerformanceExperiment:$PerformanceExperiment
exit $LASTEXITCODE
