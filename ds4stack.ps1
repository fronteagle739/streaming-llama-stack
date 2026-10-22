$ErrorActionPreference = 'Stop'
$env:PYTHONDONTWRITEBYTECODE = '1'
& python -B -m ds4stack @args
exit $LASTEXITCODE
