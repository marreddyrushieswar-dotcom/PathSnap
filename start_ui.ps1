param([int]$Port=8080)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath (Join-Path $PSScriptRoot 'ui')
$pythonPath=Join-Path $PSScriptRoot '..\.venv311\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { $pythonPath=Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
& $pythonPath -m http.server $Port --bind 0.0.0.0
