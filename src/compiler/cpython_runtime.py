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
    platlibdir: str = "lib"
    abi_thread: str = ""

    @property
    def abi_tag(self) -> str:
        return f"cp{self.version[0]}{self.version[1]}"

    @property
    def stdlib_archive_name(self) -> str:
        return f"python{self.version[0]}{self.version[1]}{self.abi_thread}.zip"

    def staged_layout(self, target: Path) -> tuple[Path, Path, Path]:
        """Where this runtime's own path computation looks under a home.

        Returns (stdlib archive, pure library directory, extension
        directory).  Windows reads the archive beside the DLL and ``Lib`` /
        ``DLLs`` under its home; POSIX getpath reads
        ``<platlibdir>/pythonXY.zip``, ``<platlibdir>/pythonX.Y`` and its
        ``lib-dynload`` under the home the adapter sets.
        """

        if self.dll.suffix.casefold() == ".dll":
            return (
                target / self.stdlib_archive_name, target / "Lib", target / "DLLs",
            )
        library = (
            target / self.platlibdir
            / f"python{self.version[0]}.{self.version[1]}{self.abi_thread}"
        )
        return (
            target / self.platlibdir / self.stdlib_archive_name,
            library,
            library / "lib-dynload",
        )


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
thread = "t" if sysconfig.get_config_var("Py_GIL_DISABLED") else ""
dll_refusal = None
if sys.platform == "win32":
    dll = base / f"python{major}{minor}{thread}.dll"
elif not sysconfig.get_config_var("Py_ENABLE_SHARED"):
    # A static-only interpreter has no shared library to load or stage.
    dll = None
    dll_refusal = "interpreter was built without a shared libpython (Py_ENABLE_SHARED=0)"
else:
    # The shared library is wherever this interpreter's build installed it,
    # not under base_prefix: Debian keeps it in the multiarch LIBDIR.  A
    # relocated build (python-build-standalone, pyenv) keeps a stale LIBDIR,
    # so base_prefix/lib is the second place this interpreter itself names.
    names = [
        name for name in (
            sysconfig.get_config_var("INSTSONAME"),
            sysconfig.get_config_var("LDLIBRARY"),
        ) if name and not name.endswith(".a")
    ]
    roots = [
        root for root in (
            sysconfig.get_config_var("LIBDIR"),
            str(base / sys.platlibdir),
            str(base / "lib"),
        ) if root
    ]
    dll = next((
        pathlib.Path(root) / name
        for name in names for root in roots
        if (pathlib.Path(root) / name).is_file()
    ), None)
    if dll is None:
        dll_refusal = (
            "shared libpython " + "/".join(names) + " not found in "
            + ", ".join(roots)
        )
print(json.dumps({
    "executable": str(pathlib.Path(sys.executable).resolve()),
    "home": str(base),
    "dll": None if dll is None else str(dll),
    "dll_refusal": dll_refusal,
    "include": sysconfig.get_path("include"),
    "import_library": str(base / "libs" / f"python{major}{minor}.lib") if sys.platform == "win32" else None,
    "stdlib": sysconfig.get_path("stdlib"),
    "dlls": (
        str(base / "DLLs") if sys.platform == "win32"
        else sysconfig.get_config_var("DESTSHARED")
    ),
    "platlibdir": sys.platlibdir,
    "abi_thread": thread,
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
    pending.extend(distributions)
    # A distro-installed distribution (Debian's python3-yaml) carries no
    # RECORD, so its file list is empty; the module's own tree is then the
    # only statement of what it needs.
    if any(md.distribution(item).files for item in distributions):
        pass
    elif spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            root = pathlib.Path(location).resolve()
            loose.extend(
                (str(path.resolve()), str(pathlib.Path(name) / path.relative_to(root)))
                for path in root.rglob("*")
                if path.is_file() and "__pycache__" not in path.parts
            )
    elif spec.origin not in {None, "built-in", "frozen"}:
        origin = pathlib.Path(spec.origin).resolve()
        loose.append((str(origin), name + origin.suffix))
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


def _inspect(executable: Path, provenance: str) -> CPythonRuntime | str:
    """The runtime this interpreter describes, or why it cannot be one."""

    completed = subprocess.run(
        [str(executable), "-I", "-S", "-c", _PROBE],
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return f"{executable}: probe failed: {completed.stderr.strip()}"
    try:
        value = json.loads(completed.stdout)
        if value.get("dll") is None:
            return f"{executable}: {value.get('dll_refusal')}"
        dll = Path(value["dll"]).resolve()
        stdlib = Path(value["stdlib"]).resolve()
        if not dll.is_file():
            return f"{executable}: runtime library {dll} is not a file"
        if not stdlib.is_dir():
            return f"{executable}: stdlib {stdlib} is not a directory"
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
            platlibdir=str(value.get("platlibdir") or "lib"),
            abi_thread=str(value.get("abi_thread") or ""),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return f"{executable}: unreadable probe record: {error}"


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
    refusals: list[str] = []
    for provenance, value in candidates:
        executable = _python_executable(value)
        if executable is None or executable in seen:
            continue
        seen.add(executable)
        runtime = _inspect(executable, provenance)
        if isinstance(runtime, CPythonRuntime):
            return runtime
        refusals.append(f"{provenance}: {runtime}")
    raise FileNotFoundError(
        "no usable CPython runtime was supplied or discovered"
        + "".join(f"\n  {refusal}" for refusal in refusals)
    )


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

    _archive, library_directory, extension_directory = runtime.staged_layout(
        target,
    )

    def destination(source: Path) -> Path:
        for root in roots:
            try:
                relative = source.relative_to(root)
            except ValueError:
                continue
            return library_directory / relative
        return extension_directory / source.name

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
            library_directory / loose_relative
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

    stdlib_archive, _library_directory, extension_directory = (
        runtime.staged_layout(target)
    )
    stdlib_archive.parent.mkdir(parents=True, exist_ok=True)
    excluded_parts = {
        "site-packages", "dist-packages", "__pycache__", "test", "tests",
        "tkinter", "idlelib", "ensurepip", "lib-dynload",
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
            if source.suffix.casefold() in {".pyc", ".pyd", ".dll", ".so"}:
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
        extension = ".pyd" if runtime.dll.suffix.casefold() == ".dll" else ".so"
        for module in requested_modules:
            matches = tuple(runtime.dlls.glob(f"{module}*{extension}"))
            for source in matches:
                _copy(source, extension_directory / source.name, files)
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
        "stdlib": stdlib_archive.relative_to(target).as_posix(),
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


def _posix_cpython_loader(runtime_dll: str) -> tuple[str, ...]:
    """The POSIX counterpart of ``LoadLibraryW(L"pythonXY.dll")``.

    Windows resolves a bare module name to the interpreter already mapped
    into the process, else to the copy staged beside the application, whose
    stdlib archive it finds beside the DLL.  POSIX has neither rule, so the
    adapter states both: an interpreter already in the process (the artifact
    loaded through ctypes) is the global scope's ``Py_IsInitialized``; else
    the staged libpython beside this artifact is opened RTLD_GLOBAL, so the
    staged extension modules bind to it, and its home is set to this
    artifact's directory, where ``stage_cpython_runtime`` laid out getpath's
    landmarks.  ``dladdr`` is reached through ``dlsym`` because ``-std=c11``
    hides its declaration; ``Dl_info`` begins with these two members on
    glibc, musl and Darwin.
    """

    return (
        "#if !defined(_WIN32)",
        "typedef struct { const char *dli_fname; void *dli_fbase; const char *dli_sname; void *dli_saddr; } TuringDlInfo;",
        "static int turing_python_start(void);",
        "static int turing_python_open(void) {",
        "    void *process = dlopen(NULL, RTLD_NOW);",
        "    int (*address_info)(const void *, TuringDlInfo *) = NULL;",
        "    void (*set_home)(const wchar_t *) = NULL;",
        "    wchar_t *(*decode)(const char *, size_t *) = NULL;",
        "    char *(*real_path)(const char *, char *) = NULL;",
        "    TuringDlInfo self = {0}; const char *slash; size_t directory_length;",
        "    char *located, *path; wchar_t *home;",
        "    if (process != NULL && dlsym(process, \"Py_IsInitialized\") != NULL) {",
        "        turing_python_module = process; return 1;",
        "    }",
        "    if (process != NULL) {",
        "        *(void **)&address_info = dlsym(process, \"dladdr\");",
        "        *(void **)&real_path = dlsym(process, \"realpath\");",
        "    }",
        "    /* glibc names the main program by its argv[0]; a bare name came",
        "       from PATH, and /proc/self/exe is then the file itself. */",
        "    located = (address_info == NULL || real_path == NULL",
        "        || !address_info((const void *)&turing_python_start, &self)",
        "        || self.dli_fname == NULL) ? NULL",
        "        : real_path(strchr(self.dli_fname, '/') != NULL ? self.dli_fname : \"/proc/self/exe\", NULL);",
        "    if (located == NULL) {",
        "        fprintf(stderr, \"private CPython runtime: cannot locate this artifact (dladdr/realpath)\\n\");",
        "        return 0;",
        "    }",
        "    slash = strrchr(located, '/');",
        "    directory_length = (size_t)(slash - located);",
        f"    path = (char *)malloc(directory_length + {len(runtime_dll) + 2});",
        "    if (path == NULL) { free(located); return 0; }",
        "    memcpy(path, located, directory_length); free(located);",
        f"    path[directory_length] = '/'; memcpy(path + directory_length + 1, {json.dumps(runtime_dll)}, {len(runtime_dll) + 1});",
        "    turing_python_module = dlopen(path, RTLD_NOW | RTLD_GLOBAL);",
        "    if (turing_python_module == NULL) {",
        "        fprintf(stderr, \"private CPython runtime: %s\\n\", dlerror()); free(path); return 0;",
        "    }",
        "    *(void **)&set_home = dlsym(turing_python_module, \"Py_SetPythonHome\");",
        "    *(void **)&decode = dlsym(turing_python_module, \"Py_DecodeLocale\");",
        "    if (set_home == NULL || decode == NULL) {",
        "        fprintf(stderr, \"private CPython runtime: %s exports no Py_SetPythonHome\\n\", path);",
        "        free(path); turing_python_module = NULL; return 0;",
        "    }",
        "    path[directory_length] = '\\0'; home = decode(path, NULL); free(path);",
        "    if (home == NULL) { turing_python_module = NULL; return 0; }",
        "    set_home(home); /* CPython keeps this pointer; it lives with the process. */",
        "    return 1;",
        "}",
        "#endif",
    )


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
        if result_dtype == "utf8_sequence":
            parameters.extend((
                "int64_t *result_data",
                "int64_t *result_length",
                "int64_t *result_capacity",
                "int64_t *result_status",
            ))
        else:
            result_c = (
                "double"
                if result_dtype in {"float32", "float64"}
                else "int64_t"
            )
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
        if result_dtype == "utf8_sequence":
            result_line = "\n".join((
                "    utf8 = py.PyUnicode_AsUTF8AndSize(returned, &utf8_size);",
                "    if (utf8 == NULL) goto failed;",
                "    if (utf8_size < 0 || utf8_size > *result_capacity) {",
                "        *result_length = 0; *result_status = 2; goto complete;",
                "    }",
                "    for (utf8_index = 0; utf8_index < utf8_size; ++utf8_index)",
                "        result_data[utf8_index] = (int64_t)(unsigned char)utf8[utf8_index];",
                "    *result_length = (int64_t)utf8_size; *result_status = 1;",
            ))
        elif result_dtype == "opaque_ref":
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
            *((
                "    const char *utf8 = NULL; TuringPySsizeT utf8_size = 0, utf8_index = 0;",
            ) if result_dtype == "utf8_sequence" else ()),
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
            *(("complete:",) if result_dtype == "utf8_sequence" else ()),
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
            (
                "    *result_length = 0; *result_status = 3;"
                if result_dtype == "utf8_sequence" else "    *result = 0;"
            ),
            "}",
        )))
    return "\n".join((
        "#include <stddef.h>",
        "#include <stdint.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "#if defined(_WIN32)",
        "#include <windows.h>",
        "#define TURING_PY_EXPORT __declspec(dllexport)",
        "typedef HMODULE TuringPyLibrary;",
        "#define TURING_PY_SYMBOL(name) ((void *)GetProcAddress(turing_python_module, name))",
        "#else",
        "#include <dlfcn.h>",
        "#include <wchar.h>",
        "#define TURING_PY_EXPORT __attribute__((visibility(\"default\")))",
        "typedef void *TuringPyLibrary;",
        "#define TURING_PY_SYMBOL(name) dlsym(turing_python_module, name)",
        "#endif",
        "typedef intptr_t TuringPySsizeT;",
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
        "    const char *(*PyUnicode_AsUTF8AndSize)(TuringPyObject *, TuringPySsizeT *);",
        "    void (*Py_DecRef)(TuringPyObject *); void (*Py_IncRef)(TuringPyObject *);",
        "    void (*PyErr_Print)(void);",
        "    int *Py_IsolatedFlag; int *Py_IgnoreEnvironmentFlag;",
        "    int *Py_NoSiteFlag; int *Py_DontWriteBytecodeFlag;",
        "} TuringPythonAPI;",
        "static TuringPythonAPI py = {0};",
        "static TuringPyLibrary turing_python_module = NULL;",
        "static TuringPyObject **turing_python_handles = NULL;",
        "static size_t turing_python_handle_count = 0, turing_python_handle_capacity = 0;",
        *_posix_cpython_loader(runtime_dll),
        "static int turing_python_start(void) {",
        "    if (turing_python_module == NULL) {",
        "#if defined(_WIN32)",
        f"        turing_python_module = LoadLibraryW(L{json.dumps(runtime_dll)});",
        "        if (turing_python_module == NULL) return 0;",
        "#else",
        "        if (!turing_python_open()) return 0;",
        "#endif",
        "#define TURING_PY_LOAD(name) *(void **)&py.name = TURING_PY_SYMBOL(#name)",
        "        TURING_PY_LOAD(Py_Initialize); TURING_PY_LOAD(Py_IsInitialized);",
        "        TURING_PY_LOAD(PyImport_ImportModule); TURING_PY_LOAD(PyObject_GetAttrString);",
        "        TURING_PY_LOAD(PyObject_Call); TURING_PY_LOAD(PyTuple_New);",
        "        TURING_PY_LOAD(PyTuple_SetItem); TURING_PY_LOAD(PyLong_FromLongLong);",
        "        TURING_PY_LOAD(PyDict_New); TURING_PY_LOAD(PyDict_SetItemString);",
        "        TURING_PY_LOAD(PyLong_AsLongLong); TURING_PY_LOAD(PyFloat_FromDouble);",
        "        TURING_PY_LOAD(PyFloat_AsDouble); TURING_PY_LOAD(PyBool_FromLong);",
        "        TURING_PY_LOAD(PyObject_IsTrue); TURING_PY_LOAD(PyBytes_FromStringAndSize);",
        "        TURING_PY_LOAD(PyUnicode_AsUTF8AndSize);",
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
        "TURING_PY_EXPORT int64_t turing_cpython_retain_bytes(const uint8_t *data, size_t length) {",
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
