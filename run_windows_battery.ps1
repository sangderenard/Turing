# Turing local Windows battery. Run from PowerShell; every step tees to build\battery\.
cd C:\dev\Powershell\turing
$ErrorActionPreference = 'Continue'; $env:PYTHONUNBUFFERED = '1'; $env:PYTHONIOENCODING = 'utf-8'
New-Item -ItemType Directory -Force build\battery | Out-Null
function Step($name, [scriptblock]$body) {
  $log = "build\battery\$name.log"; $t = Get-Date
  & $body 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
  "== $name finished in $([int]((Get-Date) - $t).TotalMinutes) min" | Tee-Object -FilePath $log -Append
}
 
# 0. Fresh main + deps
Step 00_pull { git checkout main; git pull --ff-only origin main; git log --oneline -1 }
Step 01_deps { python -m pip install -r requirements.txt llvmlite ziglang }
 
# 1. Toolchain facts only Windows can report
Step 02_env {
  python --version
  python -c "import llvmlite, ziglang; print('llvmlite', llvmlite.__version__)"
  python -c "from src.compiler.native_code_retention import host_build_target; print('host target', host_build_target())"
  python -c "from src.compiler.ssa_fortran_backend import fortran_compiler; print('fortran', fortran_compiler())"
  where.exe gcc clang cl gfortran 2>&1
}
 
# 2. Windows-sensitive tests: host target default (#899), app/mobile product, DLL exports, SPIR-V tool ids, the Windows-only hang
Step 03_windows_tests {
  python -m pytest -q --tb=short -p no:cacheprovider --timeout 900 -rf `
    tests/test_native_code_retention.py tests/test_application_product.py tests/test_ssa_optional_values.py `
    tests/test_shader_deployment.py tests/test_glsl_blas_deployment.py `
    "tests/test_loop_carried_producers.py::test_one_value_held_by_two_bindings_computes_the_authored_answer"
}
 
# 3. Bootstrap (fresh output dir), then the false-acceptance check on every product
Step 04_bootstrap { python -u tools/bootstrap_compiler_concordance.py --continue-after-failure --output build\battery\bootstrap }
Step 05_bootstrap_check {
  Get-ChildItem build\battery\bootstrap -Recurse -Directory |
    Where-Object { $_.Name -eq 'product' -and $_.Parent.Name -eq 'growth' } |
    ForEach-Object { python tools/compiler_bootstrap_status.py --check-product $_.FullName }
}
 
# 4. Typed corpus through C, LLVM and Fortran (Fortran runs only if a compiler was found in step 02)
$env:TYPED_CHECK_BACKENDS = 'c,llvm,fortran'
Step 06_corpus { python -u tools/native_corpus_typed_check.py tests/native_semantics_corpus }
Remove-Item Env:TYPED_CHECK_BACKENDS
 
# 5. Differential fuzz, newest tier, all three backends (needs PR #906 merged)
$env:FUZZ_EXT = '3'; $env:FUZZ_BACKENDS = 'llvm,fortran'
Step 07_fuzz { python -u tools/differential_fuzz.py build\battery\fuzz 61 30 }
Remove-Item Env:FUZZ_EXT, Env:FUZZ_BACKENDS
 
# 6. Bundle the logs to hand back
Compress-Archive -Force -Path build\battery\*.log, build\battery\fuzz\results.jsonl -DestinationPath build\battery\battery_logs.zip
"Logs: $(Resolve-Path build\battery\battery_logs.zip)"