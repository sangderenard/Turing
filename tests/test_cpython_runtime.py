import json
from pathlib import Path
import sys
import zipfile

import pytest

from src.compiler.cpython_runtime import (
    discover_cpython_runtime, stage_cpython_runtime,
)


def test_active_cpython_can_be_discovered_without_install_assumption():
    runtime = discover_cpython_runtime(environment={"PATH": ""})

    assert runtime.executable == Path(sys.executable).resolve()
    assert runtime.dll.is_file()
    assert runtime.stdlib.is_dir()
    assert runtime.provenance == "active-interpreter"


def test_private_runtime_is_staged_with_manifest_and_no_site_packages(tmp_path):
    runtime = discover_cpython_runtime(supplied=sys.executable)

    staged = stage_cpython_runtime(
        runtime,
        tmp_path / "runtime",
        module_identities=("_pickle.loads", "yaml.safe_load"),
    )

    manifest = json.loads(staged.manifest_path.read_text(encoding="utf-8"))
    assert (staged.directory / runtime.dll.name).is_file()
    stdlib = staged.directory / f"python{runtime.version[0]}{runtime.version[1]}.zip"
    with zipfile.ZipFile(stdlib) as archive:
        assert "encodings/__init__.py" in archive.namelist()
        assert not any("site-packages" in name for name in archive.namelist())
    assert manifest["requested_modules"] == ["_pickle", "yaml"]
    assert any(name.casefold() == "pyyaml" for name in manifest["installed_distributions"])
    assert (staged.directory / "Lib" / "yaml" / "__init__.py").is_file()
    assert manifest["provenance"] == "supplied"


def test_private_cpython_adapter_is_headerless_and_owns_object_handles():
    from src.compiler.cpython_runtime import emit_private_cpython_adapter

    source = emit_private_cpython_adapter(({
        "symbol": "pickle_thunk",
        "identity": "_pickle.loads",
        "argument_dtypes": ["opaque_ref"],
        "result_dtype": "opaque_ref",
    },), runtime_dll="python311.dll")

    assert "Python.h" not in source
    assert 'LoadLibraryW(L"python311.dll")' in source
    assert "void pickle_thunk(int64_t *a0, int64_t *result)" in source
    assert "turing_cpython_retain_bytes" in source


def test_compiled_program_api_uses_integer_transport_for_opaque_refs():
    from src.compiler.compiled_program_api import _c_type_for

    assert _c_type_for("opaque_ref") == ("int64_t", "c_int64")


def test_standalone_fortran_c_shell_calls_private_cpython_runtime(tmp_path):
    from src.compiler.fortran_c_shell import (
        compile_fortran_module_c_shell, lower_ast_source_to_ssa,
    )
    from src.compiler.ssa_fortran_backend import emit_module, fortran_compiler

    if fortran_compiler() is None:
        pytest.skip("no Fortran compiler installed")
    contract = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"
    module, outputs, exports = lower_ast_source_to_ssa(
        "import time\ndef root():\n    time.perf_counter()\n    return 1\n",
        "root",
        name="private_cpython_test",
        extraction_contract=contract,
    )
    emitted = emit_module(
        module,
        name="private_cpython_test",
        outputs=outputs,
        extra_roots=exports,
        progress=lambda _message: None,
    )

    artifact = compile_fortran_module_c_shell(
        emitted,
        {},
        tmp_path,
        entrypoint="private_cpython_test__root",
        name="private_cpython_test",
    )
    completed = artifact.run()
    published = artifact.api_path.read_text(encoding="utf-8")

    assert completed.returncode == 0, completed.stderr
    assert '"status":1' in completed.stdout
    assert (tmp_path / "cpython-runtime.json").is_file()
    assert "private_cpython_runtime:" in published
    assert "Py_Initialize" in artifact.c_source_path.read_text(encoding="utf-8")
