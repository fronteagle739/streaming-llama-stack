param([string]$Config='config.local.json', [Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
& "$PSScriptRoot/slstack.ps1" --config $Config merge_lora @CommandArgs
exit $LASTEXITCODE
