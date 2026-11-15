#Requires -Version 7
param(
    [string]$BasePython = 'C:\dev\Powershell\srv\python\python.exe',
    [string]$Environment = 'C:\dev\Powershell\srv\state\tools\hosted-llm-train'
)
$ErrorActionPreference = 'Stop'
$requirements = Join-Path $PSScriptRoot 'requirements-hosted-llm-train.txt'
if (-not (Test-Path -LiteralPath $BasePython)) { throw "Python is missing: $BasePython" }
if (-not (Test-Path -LiteralPath (Join-Path $Environment 'Scripts\python.exe'))) {
    & $BasePython -m venv $Environment
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the hosted LLM training environment' }
}
$python = Join-Path $Environment 'Scripts\python.exe'
& $python -m pip install --disable-pip-version-check --index-url https://download.pytorch.org/whl/cu124 'torch==2.5.1+cu124'
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed' }
& $python -m pip install --disable-pip-version-check -r $requirements
if ($LASTEXITCODE -ne 0) { throw 'Hosted LLM training dependencies failed' }
& $python -c 'import torch, transformers, peft, accelerate; assert torch.cuda.is_available(); print(torch.__version__, transformers.__version__, peft.__version__, accelerate.__version__)'
if ($LASTEXITCODE -ne 0) { throw 'Hosted LLM training environment did not validate CUDA and imports' }
