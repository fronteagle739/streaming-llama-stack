param([string]$Config='config.local.json', [Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
& "$PSScriptRoot/ds4stack.ps1" --config $Config train_lora @CommandArgs
exit $LASTEXITCODE
