# Trial: compile AbstractTensor as ONE class into ONE shared library
# (out\AbstractTensor_dll\...\AbstractTensor.dll), using the existing
# whole-class route in src/compiler/compile_section_to_dll.py.
#
#   .\trial_abstract_tensor_class.ps1                    # -Mode Class (default)
#   .\trial_abstract_tensor_class.ps1 -Mode PublicSurface
#
# Class: compile_class_to_dll. Every method the class body defines seeds the
#   closure; one library=True compile exports all of them.
# PublicSurface: export_public_surface. Retains every public class in
#   abstraction.py (AbstractTensor, AbstractScalar, AbstractF, MeshGrid,
#   TensorShapeError), including methods bound on from abstraction_methods.
#
# Files this leaves behind, even after Ctrl+Break:
#   out\AbstractTensor_dll\        product, plus the log next to it
#   artifacts\identity_logs\       identity-book log for this compile
#   build\compiler-logs\           compiler terminal log (when bars are on)
#   .turing-cache\host-ssa\        host-code cache (reused by later runs)
#   %TEMP%\turing-native-build-*   only if interrupted during the C/Fortran build
param([ValidateSet("Class", "PublicSurface")] [string] $Mode = "Class")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$out = "out\AbstractTensor_dll"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$log = "$out\trial-$Mode.log"

if ($Mode -eq "Class") {
  python -u -m src.compiler.compile_section_to_dll `
    src/common/tensors/abstraction.py AbstractTensor $out 2>&1 | Tee-Object $log
} else {
  python -u -c "from src.compiler.compile_section_to_dll import export_public_surface as e; h=e('src.common.tensors.abstraction', r'$out', progress=lambda s: print(s, flush=True)); print('OK ->', h.executable_path)" 2>&1 | Tee-Object $log
}
# Success ends with "OK ... -> <path>.dll". A refusal ends with a traceback and
# (Class mode) "ERROR after Ns: <Type>: <reason>".
