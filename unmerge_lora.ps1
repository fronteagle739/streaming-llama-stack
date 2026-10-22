param([string]$Config='config.local.json', [Parameter(ValueFromRemainingArguments=$true)][string[]]$CommandArgs)
& "$PSScriptRoot/ds4stack.ps1" --config $Config unmerge_lora @CommandArgs
exit $LASTEXITCODE
