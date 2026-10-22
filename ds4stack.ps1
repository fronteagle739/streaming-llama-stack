$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
Push-Location -LiteralPath $PSScriptRoot
try {
    & python -B -m ds4stack @args
    $resultCode = $LASTEXITCODE
} finally { Pop-Location }
exit $resultCode
