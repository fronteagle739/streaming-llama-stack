param(
    [Parameter(Mandatory=$true)][string]$Config,
    [Parameter(Mandatory=$true)][string]$Python,
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$PairArgs
)
& $Python -B "$PSScriptRoot/launch.py" --quant q4 --config $Config @PairArgs
exit $LASTEXITCODE
