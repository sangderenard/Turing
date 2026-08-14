"""The LLVM lane, end to end, on a full real pythonic program.

The input is ``examples/xor_project/train_xor.py`` -- the ordinary
abstract_nn training program (Model/Linear/Adam classes, the actual loop),
exactly as a user wrote it. It travels the compiler's own entries:
source -> dual IR (``compile_ast_aot``, precompile-only, whole-program) ->
repository SSA (``lower_precompile_and_control_to_ssa`` with the C
computational-core reference) -> LLVM (the likeness-table emitter) -> native
artifact (Zig's clang, ahead of time). Every stage's shortfall census must be
empty or name itself; nothing is bypassed and nothing synthetic stands in
for the program.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.common.tensors.accelerator_backends.aot_compile import compile_ast_aot
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)
from src.compiler.precompile_to_ssa import lower_precompile_and_control_to_ssa
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
)

EXAMPLE = (
    Path(__file__).resolve().parents[1]
    / "examples" / "xor_project" / "train_xor.py"
)


@pytest.fixture(scope="module")
def ssa_module():
    source = EXAMPLE.read_text(encoding="utf-8")
    compilation = compile_ast_aot(
        source, "train", {},
        precompile_only=True, bake_mode="whole_program",
        mutable_parameters=("steps", "lr"),
    )
    result = lower_precompile_and_control_to_ssa(
        compilation.compiled_shell_program,
        compilation.shell_control_program,
        region_programs=dict(compilation.region_programs),
        hierarchy_plan=compilation.hierarchy_plan,
        identity_table=compilation.identity_table,
        function_outputs=tuple(compilation.function_outputs),
        function_parameters=tuple(compilation.function_parameters),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
    )
    return result


def _region_functions(module) -> list[str]:
    return sorted(
        name for name in module.functions
        if name.startswith("numerical_region_")
    )


def test_real_program_lowers_to_ssa(ssa_module):
    for shortfall in ssa_module.shortfalls:
        print("lowering shortfall:", shortfall)
    assert ssa_module.shortfalls == ()
    assert _region_functions(ssa_module.module)


def test_real_program_regions_emit_llvm(ssa_module):
    for region in _region_functions(ssa_module.module):
        artifact = emit_ssa_function_to_llvm(
            ssa_module.module, region, entry_name=region,
        )
        for shortfall in artifact.shortfalls:
            print(f"{region} emission shortfall:", shortfall)
        assert artifact.shortfalls == (), region
        # Every local in the emitted entry is defined exactly once.
        seen: set[str] = set()
        in_entry = False
        for line in artifact.llvm_ir.splitlines():
            if line.startswith(f"define void @{region}"):
                in_entry = True
                continue
            if in_entry and line.startswith("}"):
                break
            if in_entry and " = " in line:
                name = line.strip().split(" = ")[0]
                assert name not in seen, f"{region}: duplicate local {name}"
                seen.add(name)


def test_real_program_regions_compile_to_native_artifacts(ssa_module, tmp_path):
    for region in _region_functions(ssa_module.module):
        artifact = emit_ssa_function_to_llvm(
            ssa_module.module, region, entry_name=region,
        )
        if not artifact.complete:
            pytest.fail(f"{region} has emission shortfalls")
        compiled = compile_artifact(artifact, directory=tmp_path / region)
        assert compiled.library_path is not None and compiled.library_path.is_file()
        assert compiled.entry() is not None
