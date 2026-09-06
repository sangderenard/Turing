"""Discover and stage an artifact-owned CPython runtime.

Discovery is deliberately a build concern.  A compiled artifact records where
its runtime came from, but execution uses only the staged files beside the
artifact and never searches the developer's repository, venv, or system.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, Mapping
import zipfile


@dataclass(frozen=True)
class CPythonRuntime:
    executable: Path
    home: Path
    dll: Path
    include: Path | None
    import_library: Path | None
    stdlib: Path
    dlls: Path | None
    version: tuple[int, int, int]
    provenance: str

    @property
    def abi_tag(self) -> str:
        return f"cp{self.version[0]}{self.version[1]}"


@dataclass(frozen=True)
class StagedCPythonRuntime:
    directory: Path
    runtime: CPythonRuntime
    files: tuple[Path, ...]
    manifest_path: Path


_PROBE = r"""
import json, pathlib, sys, sysconfig
base = pathlib.Path(sys.base_prefix).resolve()
major, minor, micro = sys.version_info[:3]
dll = base / (f"python{major}{minor}.dll" if sys.platform == "win32" else f"libpython{major}.{minor}.so")
print(json.dumps({
    "executable": str(pathlib.Path(sys.executable).resolve()),
    "home": str(base),
    "dll": str(dll),
    "include": sysconfig.get_path("include"),
    "import_library": str(base / "libs" / f"python{major}{minor}.lib") if sys.platform == "win32" else None,
    "stdlib": sysconfig.get_path("stdlib"),
    "dlls": str(base / "DLLs") if sys.platform == "win32" else None,
    "version": [major, minor, micro],
}))
"""

_DEPENDENCY_PROBE = r"""
import importlib.metadata as md, importlib.util, json, pathlib, re, sys, sysconfig
requested = json.loads(sys.argv[1])
packages = md.packages_distributions()
pending = []
loose = []
for name in requested:
    spec = importlib.util.find_spec(name)
    if spec is None:
        raise SystemExit("module-unavailable:" + name)
    distributions = packages.get(name, ())
    if distributions:
        pending.extend(distributions)
    elif spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            root = pathlib.Path(location).resolve()
            loose.extend(
                [str(path.resolve()), str(pathlib.Path(name) / path.relative_to(root))]
                for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    elif spec.origin not in {None, "built-in", "frozen"}:
        origin = pathlib.Path(spec.origin).resolve()
        loose.append([str(origin), name + origin.suffix])
seen = set(); files = []
while pending:
    name = pending.pop(0)
    folded = name.casefold()
    if folded in seen: continue
    seen.add(folded)
    dist = md.distribution(name)
    for item in dist.files or ():
        path = pathlib.Path(dist.locate_file(item)).resolve()
        if path.is_file(): files.append(str(path))
    for requirement in dist.requires or ():
        match = re.match(r"[A-Za-z0-9_.-]+", requirement)
        if match: pending.append(match.group(0))
print(json.dumps({
    "purelib": sysconfig.get_path("purelib"),
    "platlib": sysconfig.get_path("platlib"),
    "distributions": sorted(seen),
    "files": sorted(set(files)),
    "loose": sorted(set(loose)),
}))
"""


def _python_executable(value: str | Path) -> Path | None:
    path = Path(value).expanduser().resolve()
    if path.is_file():
        return path
    candidates = (
        path / "python.exe",
        path / "Scripts" / "python.exe",
        path / "bin" / "python",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _inspect(executable: Path, provenance: str) -> CPythonRuntime | None:
    completed = subprocess.run(
        [str(executable), "-I", "-S", "-c", _PROBE],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    try:
        value = json.loads(completed.stdout)
        dll = Path(value["dll"]).resolve()
        stdlib = Path(value["stdlib"]).resolve()
        if not dll.is_file() or not stdlib.is_dir():
            return None
        include = Path(value["include"]).resolve() if value.get("include") else None
        import_library = (
            Path(value["import_library"]).resolve()
            if value.get("import_library") else None
        )
        dlls = Path(value["dlls"]).resolve() if value.get("dlls") else None
        return CPythonRuntime(
            executable=Path(value["executable"]).resolve(),
            home=Path(value["home"]).resolve(),
            dll=dll,
            include=include if include and include.is_dir() else None,
            import_library=(
                import_library if import_library and import_library.is_file() else None
            ),
            stdlib=stdlib,
            dlls=dlls if dlls and dlls.is_dir() else None,
            version=tuple(map(int, value["version"])),
            provenance=provenance,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def discover_cpython_runtime(
    supplied: str | Path | None = None,
    *,
    repository: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> CPythonRuntime:
    """Select supplied, repository/venv, active, then PATH CPython."""

    env = dict(os.environ if environment is None else environment)
    candidates: list[tuple[str, str | Path]] = []
    if supplied is not None:
        candidates.append(("supplied", supplied))
    if repository is not None:
        root = Path(repository).resolve()
        for name in (".venv", "venv"):
            candidates.append((f"repository:{name}", root / name))
    if env.get("VIRTUAL_ENV"):
        candidates.append(("environment:VIRTUAL_ENV", env["VIRTUAL_ENV"]))
    candidates.append(("active-interpreter", sys.executable))
    for name in ("python3", "python"):
        found = shutil.which(name, path=env.get("PATH"))
        if found:
            candidates.append((f"PATH:{name}", found))

    seen: set[Path] = set()
    for provenance, value in candidates:
        executable = _python_executable(value)
        if executable is None or executable in seen:
            continue
        seen.add(executable)
        runtime = _inspect(executable, provenance)
        if runtime is not None:
            return runtime
    raise FileNotFoundError("no usable CPython runtime was supplied or discovered")


def _copy(source: Path, destination: Path, files: list[Path]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    files.append(destination)


def _stage_installed_dependencies(
    runtime: CPythonRuntime,
    modules: Iterable[str],
    target: Path,
    files: list[Path],
) -> tuple[str, ...]:
    requested = tuple(sorted(set(map(str, modules))))
    if not requested:
        return ()
    completed = subprocess.run(
        [
            str(runtime.executable), "-I", "-c", _DEPENDENCY_PROBE,
            json.dumps(requested),
        ],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "failed to harvest CPython module dependencies: "
            + (completed.stderr or completed.stdout).strip()
        )
    record = json.loads(completed.stdout)
    roots = tuple(
        Path(record[name]).resolve()
        for name in ("purelib", "platlib")
        if record.get(name)
    )

    def destination(source: Path) -> Path:
        for root in roots:
            try:
                relative = source.relative_to(root)
            except ValueError:
                continue
            return target / "Lib" / relative
        return target / "DLLs" / source.name

    dependency_files = [
        (Path(value).resolve(), None) for value in record["files"]
    ]
    dependency_files.extend(
        (Path(value[0]).resolve(), Path(value[1])) for value in record["loose"]
    )
    for source, loose_relative in dependency_files:
        if not source.is_file():
            continue
        try:
            stdlib_relative = source.relative_to(runtime.stdlib)
        except ValueError:
            stdlib_relative = None
        if (
            stdlib_relative is not None
            and "site-packages" not in stdlib_relative.parts
        ):
            continue
        target_path = (
            target / "Lib" / loose_relative
            if loose_relative is not None else destination(source)
        )
        _copy(source, target_path, files)
    return tuple(map(str, record["distributions"]))


def _stage_pe_dependency_closure(
    runtime: CPythonRuntime,
    target: Path,
    files: list[Path],
) -> tuple[str, ...]:
    """Harvest non-system PE imports of staged extension modules and DLLs."""

    if os.name != "nt":
        return ()
    from .binary_ingestion import BinaryFormatError, parse_pe_image

    windows = Path(os.environ.get("WINDIR", r"C:\Windows")).resolve()
    system_roots = {windows / "System32", windows / "SysWOW64"}
    search_roots = [runtime.home, runtime.executable.parent]
    if runtime.dlls is not None:
        search_roots.append(runtime.dlls)
    search_roots.extend(
        Path(item).resolve()
        for item in os.environ.get("PATH", "").split(os.pathsep)
        if item and Path(item).is_dir()
    )
    search_roots.extend(path.parent for path in files)
    indexes: dict[Path, dict[str, Path]] = {}

    def locate(name: str, requester: Path) -> Path | None:
        folded = str(name).casefold()
        if folded.startswith(("api-ms-win-", "ext-ms-win-")):
            return None
        for root in (requester.parent, *search_roots):
            if not root.is_dir():
                continue
            index = indexes.get(root)
            if index is None:
                try:
                    index = {
                        child.name.casefold(): child
                        for child in root.iterdir() if child.is_file()
                    }
                except OSError:
                    index = {}
                indexes[root] = index
            candidate = index.get(folded)
            if candidate is not None:
                return candidate.resolve()
        return None

    queue = [
        path for path in files
        if path.suffix.casefold() in {".pyd", ".dll"}
    ]
    processed: set[Path] = set()
    harvested: list[str] = []
    while queue:
        staged = queue.pop(0).resolve()
        if staged in processed:
            continue
        processed.add(staged)
        try:
            encoded = staged.read_bytes()
            image, _statistics = parse_pe_image(
                encoded, maximum_file_size=len(encoded),
            )
        except (OSError, BinaryFormatError, ValueError):
            continue
        libraries = sorted({
            reference.library
            for reference in (*image.imports, *image.delay_imports)
        })
        for library in libraries:
            source = locate(library, staged)
            if source is None:
                continue
            if any(root == source.parent or root in source.parents for root in system_roots):
                continue
            destination = target / source.name
            if not destination.exists():
                _copy(source, destination, files)
            queue.append(destination)
            harvested.append(source.name)
    return tuple(sorted(set(harvested)))


def stage_cpython_runtime(
    runtime: CPythonRuntime,
    directory: str | Path,
    *,
    module_identities: Iterable[str] = (),
) -> StagedCPythonRuntime:
    """Copy a relocatable private runtime and its importable standard library.

    The standard library is staged without ``site-packages``.  Requested
    extension modules are copied from the selected runtime's ``DLLs`` folder;
    built-in modules require no separate file.  Third-party package harvesting
    is intentionally a separate dependency input rather than an accidental
    sweep of the developer environment.
    """

    target = Path(directory).resolve()
    target.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    _copy(runtime.dll, target / runtime.dll.name, files)
    license_file = runtime.home / "LICENSE.txt"
    if license_file.is_file():
        _copy(license_file, target / license_file.name, files)
    for name in ("vcruntime140.dll", "vcruntime140_1.dll"):
        candidate = runtime.home / name
        if candidate.is_file():
            _copy(candidate, target / candidate.name, files)

    stdlib_archive = target / f"python{runtime.version[0]}{runtime.version[1]}.zip"
    excluded_parts = {
        "site-packages", "__pycache__", "test", "tests", "tkinter",
        "idlelib", "ensurepip",
    }
    with zipfile.ZipFile(
        stdlib_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
    ) as archive:
        for source in sorted(runtime.stdlib.rglob("*")):
            if not source.is_file():
                continue
            relative = source.relative_to(runtime.stdlib)
            if excluded_parts.intersection(relative.parts):
                continue
            if source.suffix.casefold() in {".pyc", ".pyd", ".dll"}:
                continue
            archive.write(source, relative.as_posix())
    files.append(stdlib_archive)

    requested_modules = sorted({
        str(identity).partition(".")[0] for identity in module_identities
        if str(identity).partition(".")[0]
    })
    installed_distributions = _stage_installed_dependencies(
        runtime, requested_modules, target, files,
    )
    if runtime.dlls is not None:
        for module in requested_modules:
            matches = tuple(runtime.dlls.glob(f"{module}*.pyd"))
            for source in matches:
                _copy(source, target / "DLLs" / source.name, files)
    harvested_native_dependencies = _stage_pe_dependency_closure(
        runtime, target, files,
    )

    manifest = {
        "schema": "turing.private-cpython-runtime.v1",
        "abi_tag": runtime.abi_tag,
        "version": list(runtime.version),
        "provenance": runtime.provenance,
        "source_home": runtime.home.as_posix(),
        "runtime_dll": runtime.dll.name,
        "stdlib": stdlib_archive.name,
        "requested_modules": requested_modules,
        "installed_distributions": list(installed_distributions),
        "harvested_native_dependencies": list(harvested_native_dependencies),
        "files": sorted(path.relative_to(target).as_posix() for path in files),
    }
    manifest_path = target / "cpython-runtime.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    files.append(manifest_path)
    return StagedCPythonRuntime(target, runtime, tuple(files), manifest_path)


def emit_private_cpython_adapter(
    thunks: Iterable[Mapping[str, object]],
    *,
    runtime_dll: str,
) -> str:
    """Emit C thunks that own CPython loading and object handles in-process."""

    planned = tuple(dict(thunk) for thunk in thunks)
    if not planned:
        return ""
    definitions: list[str] = []
    for thunk in planned:
        symbol = str(thunk["symbol"])
        identity = str(thunk["identity"])
        module_name, separator, qualname = identity.partition(".")
        if not separator:
            raise ValueError(f"CPython external identity is not qualified: {identity!r}")
        argument_dtypes = tuple(map(str, thunk.get("argument_dtypes") or ()))
        keyword_names = tuple(map(str, thunk.get("keyword_names") or ()))
        positional_count = len(argument_dtypes) - len(keyword_names)
        if positional_count < 0:
            raise ValueError(f"external thunk {symbol!r} has too many keyword names")
        result_dtype = str(thunk.get("result_dtype") or "opaque_ref")
        parameters = [
            f"{('double' if dtype in {'float32', 'float64'} else 'int64_t')} *a{index}"
            for index, dtype in enumerate(argument_dtypes)
        ]
        result_c = "double" if result_dtype in {"float32", "float64"} else "int64_t"
        parameters.append(f"{result_c} *result")
        argument_lines = []
        for index, dtype in enumerate(argument_dtypes):
            if dtype == "opaque_ref":
                expression = f"turing_python_handle(*a{index})"
            elif dtype in {"float32", "float64"}:
                expression = f"turing_python_float(*a{index})"
            elif dtype == "bool":
                expression = f"turing_python_bool(*a{index} != 0)"
            else:
                expression = f"turing_python_int(*a{index})"
            if index < positional_count:
                argument_lines.extend((
                    f"    item = {expression};",
                    f"    if (item == NULL || py.PyTuple_SetItem(args, {index}, item) != 0) goto failed;",
                ))
            else:
                keyword = keyword_names[index - positional_count]
                argument_lines.extend((
                    f"    item = {expression};",
                    f"    if (item == NULL || py.PyDict_SetItemString(kwargs, {json.dumps(keyword)}, item) != 0) goto failed;",
                    "    py.Py_DecRef(item); item = NULL;",
                ))
        if result_dtype == "opaque_ref":
            result_line = "    *result = turing_python_retain(returned); returned = NULL;"
        elif result_dtype in {"float32", "float64"}:
            result_line = "    *result = py.PyFloat_AsDouble(returned);"
        elif result_dtype == "bool":
            result_line = "    *result = (int64_t)py.PyObject_IsTrue(returned);"
        else:
            result_line = "    *result = py.PyLong_AsLongLong(returned);"
        definitions.append("\n".join((
            f"void {symbol}({', '.join(parameters)}) {{",
            "    TuringPyObject *module = NULL, *callable = NULL, *args = NULL, *kwargs = NULL;",
            "    TuringPyObject *item = NULL, *returned = NULL;",
            "    if (!turing_python_start()) goto failed;",
            f"    module = py.PyImport_ImportModule({json.dumps(module_name)});",
            f"    if (module != NULL) callable = turing_python_resolve(module, {json.dumps(qualname)});",
            f"    args = py.PyTuple_New({positional_count});",
            *(('    kwargs = py.PyDict_New();',) if keyword_names else ()),
            "    if (module == NULL || callable == NULL || args == NULL" + (" || kwargs == NULL" if keyword_names else "") + ") goto failed;",
            *argument_lines,
            "    returned = py.PyObject_Call(callable, args, kwargs);",
            "    if (returned == NULL) goto failed;",
            result_line,
            "    if (returned != NULL) py.Py_DecRef(returned);",
            "    if (kwargs != NULL) py.Py_DecRef(kwargs);",
            "    py.Py_DecRef(args); py.Py_DecRef(callable); py.Py_DecRef(module);",
            "    return;",
            "failed:",
            "    if (py.PyErr_Print != NULL) py.PyErr_Print();",
            "    if (returned != NULL) py.Py_DecRef(returned);",
            "    if (args != NULL) py.Py_DecRef(args);",
            "    if (kwargs != NULL) py.Py_DecRef(kwargs);",
            "    if (callable != NULL) py.Py_DecRef(callable);",
            "    if (module != NULL) py.Py_DecRef(module);",
            "    *result = 0;",
            "}",
        )))
    return "\n".join((
        "#if !defined(_WIN32)",
        '#error "private CPython adapter currently requires the Windows loader ABI"',
        "#endif",
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "#include <windows.h>",
        "typedef struct _object TuringPyObject;",
        "typedef struct {",
        "    void (*Py_Initialize)(void); int (*Py_IsInitialized)(void);",
        "    TuringPyObject *(*PyImport_ImportModule)(const char *);",
        "    TuringPyObject *(*PyObject_GetAttrString)(TuringPyObject *, const char *);",
        "    TuringPyObject *(*PyObject_Call)(TuringPyObject *, TuringPyObject *, TuringPyObject *);",
        "    TuringPyObject *(*PyTuple_New)(size_t);",
        "    int (*PyTuple_SetItem)(TuringPyObject *, size_t, TuringPyObject *);",
        "    TuringPyObject *(*PyDict_New)(void);",
        "    int (*PyDict_SetItemString)(TuringPyObject *, const char *, TuringPyObject *);",
        "    TuringPyObject *(*PyLong_FromLongLong)(long long);",
        "    long long (*PyLong_AsLongLong)(TuringPyObject *);",
        "    TuringPyObject *(*PyFloat_FromDouble)(double);",
        "    double (*PyFloat_AsDouble)(TuringPyObject *);",
        "    TuringPyObject *(*PyBool_FromLong)(long);",
        "    int (*PyObject_IsTrue)(TuringPyObject *);",
        "    TuringPyObject *(*PyBytes_FromStringAndSize)(const char *, size_t);",
        "    void (*Py_DecRef)(TuringPyObject *); void (*Py_IncRef)(TuringPyObject *);",
        "    void (*PyErr_Print)(void);",
        "    int *Py_IsolatedFlag; int *Py_IgnoreEnvironmentFlag;",
        "    int *Py_NoSiteFlag; int *Py_DontWriteBytecodeFlag;",
        "} TuringPythonAPI;",
        "static TuringPythonAPI py = {0};",
        "static HMODULE turing_python_module = NULL;",
        "static TuringPyObject **turing_python_handles = NULL;",
        "static size_t turing_python_handle_count = 0, turing_python_handle_capacity = 0;",
        "static int turing_python_start(void) {",
        "    if (turing_python_module == NULL) {",
        f"        turing_python_module = LoadLibraryW(L{json.dumps(runtime_dll)});",
        "        if (turing_python_module == NULL) return 0;",
        "#define TURING_PY_LOAD(name) *(FARPROC *)&py.name = GetProcAddress(turing_python_module, #name)",
        "        TURING_PY_LOAD(Py_Initialize); TURING_PY_LOAD(Py_IsInitialized);",
        "        TURING_PY_LOAD(PyImport_ImportModule); TURING_PY_LOAD(PyObject_GetAttrString);",
        "        TURING_PY_LOAD(PyObject_Call); TURING_PY_LOAD(PyTuple_New);",
        "        TURING_PY_LOAD(PyTuple_SetItem); TURING_PY_LOAD(PyLong_FromLongLong);",
        "        TURING_PY_LOAD(PyDict_New); TURING_PY_LOAD(PyDict_SetItemString);",
        "        TURING_PY_LOAD(PyLong_AsLongLong); TURING_PY_LOAD(PyFloat_FromDouble);",
        "        TURING_PY_LOAD(PyFloat_AsDouble); TURING_PY_LOAD(PyBool_FromLong);",
        "        TURING_PY_LOAD(PyObject_IsTrue); TURING_PY_LOAD(PyBytes_FromStringAndSize);",
        "        TURING_PY_LOAD(Py_DecRef); TURING_PY_LOAD(Py_IncRef); TURING_PY_LOAD(PyErr_Print);",
        "        TURING_PY_LOAD(Py_IsolatedFlag); TURING_PY_LOAD(Py_IgnoreEnvironmentFlag);",
        "        TURING_PY_LOAD(Py_NoSiteFlag); TURING_PY_LOAD(Py_DontWriteBytecodeFlag);",
        "#undef TURING_PY_LOAD",
        "        if (py.Py_Initialize == NULL || py.Py_IsInitialized == NULL ||",
        "            py.PyImport_ImportModule == NULL || py.PyObject_GetAttrString == NULL ||",
        "            py.PyObject_Call == NULL || py.PyTuple_New == NULL ||",
        "            py.PyTuple_SetItem == NULL || py.PyDict_New == NULL ||",
        "            py.PyDict_SetItemString == NULL || py.Py_DecRef == NULL ||",
        "            py.Py_IncRef == NULL) return 0;",
        "    }",
        "    if (!py.Py_IsInitialized()) {",
        "        if (py.Py_IsolatedFlag != NULL) *py.Py_IsolatedFlag = 1;",
        "        if (py.Py_IgnoreEnvironmentFlag != NULL) *py.Py_IgnoreEnvironmentFlag = 1;",
        "        if (py.Py_NoSiteFlag != NULL) *py.Py_NoSiteFlag = 1;",
        "        if (py.Py_DontWriteBytecodeFlag != NULL) *py.Py_DontWriteBytecodeFlag = 1;",
        "        py.Py_Initialize();",
        "    }",
        "    return py.Py_IsInitialized();",
        "}",
        "static TuringPyObject *turing_python_handle(int64_t handle) {",
        "    if (handle <= 0 || (size_t)handle > turing_python_handle_count) return NULL;",
        "    return turing_python_handles[(size_t)handle - 1];",
        "}",
        "static TuringPyObject *turing_python_resolve(TuringPyObject *root, const char *qualname) {",
        "    TuringPyObject *current = root, *next; const char *start = qualname, *dot;",
        "    char *component; size_t length; py.Py_IncRef(current);",
        "    while (start != NULL && *start != '\\0') {",
        "        dot = strchr(start, '.'); length = dot ? (size_t)(dot - start) : strlen(start);",
        "        component = (char *)malloc(length + 1);",
        "        if (component == NULL) { py.Py_DecRef(current); return NULL; }",
        "        memcpy(component, start, length); component[length] = '\\0';",
        "        next = py.PyObject_GetAttrString(current, component); free(component);",
        "        py.Py_DecRef(current); if (next == NULL) return NULL; current = next;",
        "        start = dot ? dot + 1 : NULL;",
        "    }",
        "    return current;",
        "}",
        "static int64_t turing_python_retain(TuringPyObject *value) {",
        "    TuringPyObject **grown; size_t capacity;",
        "    if (value == NULL) return 0;",
        "    if (turing_python_handle_count == turing_python_handle_capacity) {",
        "        capacity = turing_python_handle_capacity ? turing_python_handle_capacity * 2 : 64;",
        "        grown = (TuringPyObject **)realloc(turing_python_handles, capacity * sizeof(*grown));",
        "        if (grown == NULL) return 0;",
        "        turing_python_handles = grown; turing_python_handle_capacity = capacity;",
        "    }",
        "    turing_python_handles[turing_python_handle_count++] = value;",
        "    return (int64_t)turing_python_handle_count;",
        "}",
        "static TuringPyObject *turing_python_int(int64_t value) { return py.PyLong_FromLongLong(value); }",
        "static TuringPyObject *turing_python_float(double value) { return py.PyFloat_FromDouble(value); }",
        "static TuringPyObject *turing_python_bool(int value) { return py.PyBool_FromLong(value); }",
        "__declspec(dllexport) int64_t turing_cpython_retain_bytes(const uint8_t *data, size_t length) {",
        "    TuringPyObject *value; if (!turing_python_start()) return 0;",
        "    value = py.PyBytes_FromStringAndSize((const char *)data, length);",
        "    return turing_python_retain(value);",
        "}",
        *definitions,
        "",
    ))


__all__ = [
    "CPythonRuntime", "StagedCPythonRuntime", "discover_cpython_runtime",
    "emit_private_cpython_adapter", "stage_cpython_runtime",
]
