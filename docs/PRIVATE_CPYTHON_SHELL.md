# Python-module native boundaries in Python and C shells

An extraction rule with `native_abi: cpython-c-api` is executable by two shell
profiles without changing the compiled call occurrence:

- `python`: the Python shell owns the selected interpreter and services the
  physical external-reference request/completion records directly.
- `cpython-c`: the C shell loads an artifact-owned CPython DLL, initializes it
  in isolated mode, and exposes generated C thunks to the Fortran internals.

There is no launcher-installed resolver in the C-shell profile. Fortran calls a
normal `bind(C)` symbol. The C translation unit owns module resolution, the
CPython call ABI, and the object-handle table.

## Build-time runtime selection

`discover_cpython_runtime` selects the first usable runtime in this order:

1. an explicitly supplied executable or CPython home;
2. `.venv` or `venv` in the repository;
3. `VIRTUAL_ENV`;
4. the active build interpreter;
5. `python3` or `python` on `PATH`.

Discovery does not become a runtime dependency. `stage_cpython_runtime` copies
the selected DLL, VC redistributables, license, and a versioned standard-library
zip into the compiled artifact. Generated C loads the DLL beside the executable.
It sets isolated, ignore-environment, no-site, and no-bytecode flags before
initialization, so an installed Python, `PYTHONPATH`, or user site cannot make an
otherwise incomplete artifact appear healthy.

## Dependency harvesting

The compiler supplies the module identities retained by the external-reference
thunks. Harvesting then uses the selected interpreter—not the agent's filesystem
assumptions—to classify them:

- built-in/frozen modules require no additional file;
- standard-library source is already in the versioned zip;
- installed distributions are found through `importlib.metadata`, including
  recursively declared distribution requirements;
- loose modules and packages are copied with their package-relative paths;
- requested extension modules are copied from the runtime `DLLs` directory;
- staged `.pyd` and `.dll` files are parsed with Turing's PE reader, and their
  resolvable non-system import closure is copied alongside the artifact;
- Windows/API-set and System32 dependencies remain operating-system facilities.

The artifact writes `cpython-runtime.json` with source provenance, ABI tag,
requested modules, installed distributions, harvested native dependencies, and
the exact staged-file inventory. This is the durable evidence a deployment
consumer should use; it should never repeat discovery from the build machine.

## Object transport

The common record ABI transports immutable scalars by value and arbitrary
objects as shell-owned signed-64-bit handles. The C shell also exports
`turing_cpython_retain_bytes`, allowing a byte/file broker to seed a CPython
bytes object before Fortran calls `_pickle.loads`. The returned unpickled object
is another handle and may be passed to later CPython-native calls without a
`PyObject *` crossing the Fortran ABI.

Public opaque results are handles whose lifetime is the process. Publishing a
durable graph outside that process still requires an explicit materialization
contract; interpreting a handle as a portable serialized object would be wrong.

## Remaining closure limits

Distribution metadata and PE imports cover declared/static dependencies. They
cannot prove arbitrary imports synthesized from runtime strings, plugin entry
points selected from external configuration, or DLLs opened through computed
paths. Those must arrive as additional compiler dependency seeds or remain named
runtime capability requirements. CPython's complete machine-code ingestion path
is complementary: it retains dynamic machine state exactly, but it does not make
dynamic dependency identities statically knowable.

