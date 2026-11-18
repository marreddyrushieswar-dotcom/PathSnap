param([string]$BindAddress='127.0.0.1', [int]$Port=8000)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath=Join-Path $PSScriptRoot '..\.venv311\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { $pythonPath=Join-Path $PSScriptRoot '.venv\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Create the project virtual environment first; see README.md.' }
$env:OMP_NUM_THREADS='2'
& $pythonPath -m uvicorn pathsnap.api:app --host $BindAddress --port $Port
