# One-shot demo on Windows: run the full pipeline, then open the dashboard.
# Usage (from the project root):  .\scripts\run_demo.ps1            all four subsets
#                                 .\scripts\run_demo.ps1 FD001       just one
param([string[]]$Subsets = @("FD001", "FD002", "FD003", "FD004"))

$ErrorActionPreference = "Stop"
$py = Join-Path $PSScriptRoot "..\venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }

& $py -m fleetguard run-all --subsets $Subsets
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $py -m streamlit run (Join-Path $PSScriptRoot "..\dashboard\app.py")
