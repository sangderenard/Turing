"""An authored unsigned span must not silently become floating storage."""

from pathlib import Path

import pytest
import yaml

import numpy as np
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tests" / "unsigned_span_corpus" / "c621_unsigned_remainder.py"
BASE_CONTRACT = ROOT / "extraction_contracts" / "program_extraction.yaml"


@pytest.mark.parametrize("backend", ("c", "llvm", "fortran"))
@pytest.mark.parametrize(
    "dtype, storage, shape",
    (("uint64", "span", None), ("u64", "span", (3,)),
     ("uint64", "scalar", ())),
)
def test_c621_unsigned_program_abi_executes_exact_native_remainder(
    tmp_path, dtype, storage, shape, backend,
):
    raw = yaml.safe_load(BASE_CONTRACT.read_text(encoding="utf-8"))
    values = []
    for parameter in ("x", "n"):
        field = {
            "function": "f", "parameter": parameter,
            "storage": storage if parameter == "x" else "scalar",
            "dtype": dtype,
            "python_type": (
                "src.common.tensors.abstraction.AbstractTensor"
                if parameter == "x" and storage == "span" else "builtins.int"
            ),
        }
        if parameter == "x" and storage == "span":
            field["rank"] = 1
            if shape is not None:
                field["shape"] = list(shape)
        values.append(field)
    raw["program_abi"] = {"records": {}, "bindings": [], "values": values}
    contract = tmp_path / "unsigned-program-abi.yaml"
    contract.write_text(yaml.safe_dump(raw), encoding="utf-8")

    module, outputs, exports = lower_ast_source_to_ssa(
        SOURCE.read_text(encoding="utf-8"), "f", name="c621_unsigned_remainder",
        extraction_contract=str(contract), progress=lambda _: None, progress_bars=False,
    )
    root_name = next(name for name in exports if name.endswith("__f"))
    root = module.functions[root_name]
    parameters = dict(root.metadata["parameter_names"])
    cases = [2**53 + 1, 2**63 + 17, 2**64 - 1]
    x = np.array(cases if storage == "span" else [cases[-1]], dtype=np.uint64)
    divisor = 7
    feeds = {parameters["x"]: x, parameters["n"]: np.array([divisor], dtype=np.uint64)}
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c
        artifact = emit_ssa_to_c(module, root_name)
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "native")
        execution = artifact.prepare_execution(feeds).run()
    elif backend == "llvm":
        from llvmlite import binding
        from src.compiler.ssa_llvm_backend import (
            emit_ssa_function_to_llvm, compile_artifact, prepare_artifact_execution,
        )
        artifact = emit_ssa_function_to_llvm(module, root_name)
        assert artifact.complete, artifact.shortfalls
        binding.parse_assembly(artifact.llvm_ir).verify()
        built = compile_artifact(artifact, directory=tmp_path / "native")
        execution = prepare_artifact_execution(built, feeds).run()
    else:
        from src.compiler.ssa_fortran_backend import emit_module, FortranCoreNative, fortran_compiler
        from src.compiler.fortran_c_shell import compile_fortran_module_c_shell
        if fortran_compiler() is None:
            pytest.skip("native Fortran compiler unavailable")
        from src.compiler.fortran_toolchain import native_fortran_flags, aggressive_fortran_flags
        compiler = fortran_compiler()
        try:
            native_fortran_flags(compiler, 'unsigned(c_int64_t) :: capability_probe',
                                 aggressive_fortran_flags(compiler))
        except ValueError as error:
            if str(error).startswith((
                'native unsigned Fortran requires a compiler supporting -funsigned;',
                'selected Fortran compiler/flags do not support exact native unsigned ISO C storage with -funsigned:',
            )):
                pytest.skip('selected Fortran toolchain lacks native unsigned ISO-C storage: ' + str(error))
            raise
        artifact = emit_module(module, name="c621_unsigned_remainder", outputs=outputs, progress=lambda _: None)
        assert artifact.complete, artifact.shortfalls
        built = compile_fortran_module_c_shell(
            artifact, {}, tmp_path / "native", entrypoint=root_name, library=True,
        )
        execution = FortranCoreNative(
            built.executable_path, artifact.api.entry_point(root_name), artifact.api,
        ).prepare_execution(feeds).run()
    result = execution.buffers[outputs[root_name][0].id]
    assert result.dtype == np.dtype("uint64")
    assert result.reshape(-1).tolist() == [int(value) % divisor for value in x]
