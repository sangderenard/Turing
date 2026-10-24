# Trial: compile every AbstractTensor method as one project catalogue and emit
# each complete unit as a native shared library (.dll on Windows).
# Run from the repo root: C:\dev\Powershell\turing
$ErrorActionPreference = "Stop"
$out = "out\abstract_tensor_class"
New-Item -ItemType Directory -Force -Path out | Out-Null

# The class's methods, as the product itself discovers them (199 today;
# nested <locals> helpers are pulled in as dependencies automatically).
$methods = python -c "from src.compiler.project_compilation_product import discover_authored_calls as d; s=open('src/common/tensors/abstraction.py',encoding='utf-8').read(); print('\n'.join(c.qualified_name for c in d(s) if c.qualified_name.startswith('AbstractTensor.') and '.<locals>.' not in c.qualified_name))"
$entryArgs = $methods | ForEach-Object { "--entry"; $_ }

python -u -m tools.compile_project_catalogue `
  --source src/common/tensors/abstraction.py `
  --output $out --emit-native --jobs 4 @entryArgs 2>&1 | Tee-Object "$out.log"

# Summary: unit status, native status, and every library produced.
python -c "import json,collections; m=json.load(open(r'$out\manifest.json')); u=m['units']; print('units', collections.Counter(x['status'] for x in u)); print('native', collections.Counter(x.get('native_status','-') for x in u)); print('verified', collections.Counter(x.get('native_verification_status','-') for x in u)); [print('DLL', x['qualified_name'], x['native_library']) for x in u if x.get('native_library')]"
