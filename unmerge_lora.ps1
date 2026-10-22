param([string]$Config='config.local.json', [Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
& "$PSScriptRoot/slstack.ps1" --config $Config unmerge_lora @CommandArgs
exit $LASTEXITCODE
