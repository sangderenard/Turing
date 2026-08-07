from __future__ import annotations

import ctypes
import os
from pathlib import Path
import subprocess

import networkx as nx
import pytest

from src.common.tensors.accelerator_backends.native_library import (
    compile_and_load,
    detect_toolchains,
)
from src.compiler.machine_code_lifting import (
    MachineFunction,
    MachineInstruction,
    MachineLiftError,
    c_function_token_multigraph,
    disassemble_gnu_object,
    emit_scalar_c,
    lift_x86_64_affine_function,
    parse_objdump_function,
    quotient_common_subexpressions,
    ssa_dataflow_multigraph,
    topology_similarity,
)
from src.transmogrifier.ssa_registry import Handler


SOURCE = """
#include <stdint.h>
__declspec(dllexport) int32_t doubled_product(int32_t x, int32_t y) {
    return (x * y) + (x * y);
}
"""


def _gnu_toolchain():
    return next((item for item in detect_toolchains() if item.kind == "gnu"), None)


def _compile_object(source: str, path: Path) -> None:
    toolchain = _gnu_toolchain()
    if toolchain is None:
        pytest.skip("a GNU-compatible C toolchain is required")
    environment = dict(os.environ)
    environment["PATH"] = (
        str(Path(toolchain.executable).parent)
        + os.pathsep
        + environment.get("PATH", "")
    )
    completed = subprocess.run(
        [toolchain.executable, "-x", "c", "-O2", "-c", "-o", str(path), "-"],
        input=source,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout


TOKEN_ARGUMENT = (1 << 32) | 0
TOKEN_RETURN = (1 << 32) | 1
TOKEN_ADD = (2 << 32) | 0  # canonical op id 0
TOKEN_MUL = (2 << 32) | 2  # canonical op id 2
OPERATION_TOKENS = {
    "argument": TOKEN_ARGUMENT,
    Handler.Add.value: TOKEN_ADD,
    Handler.Mul.value: TOKEN_MUL,
    Handler.Ret.value: TOKEN_RETURN,
}


def test_machine_reference_vocabulary_fails_closed():
    unsupported = MachineFunction(
        "unsupported",
        (MachineInstruction(0, b"\x90", "mov", "eax,ecx"),),
    )
    with pytest.raises(MachineLiftError, match="unsupported machine instruction mov"):
        lift_x86_64_affine_function(
            unsupported,
            argument_registers=("ecx",),
            argument_names=("x",),
        )


def test_c_machine_ssa_c_roundtrip_is_behavioral_and_topologically_similar(tmp_path):
    original_object = tmp_path / "original.o"
    _compile_object(SOURCE, original_object)
    decoded = parse_objdump_function(
        disassemble_gnu_object(original_object), "doubled_product",
    )

    # GCC performs common-subexpression elimination: the source has two Mul
    # nodes, while the decoded machine program has one IMUL whose result is
    # reused by LEA. The raiser preserves that optimized topology.
    assert [item.mnemonic for item in decoded.instructions] == ["imul", "lea", "ret"]
    lifted = lift_x86_64_affine_function(
        decoded,
        argument_registers=("ecx", "edx"),
        argument_names=("x", "y"),
    )
    raised_graph = ssa_dataflow_multigraph(
        lifted, operation_tokens=OPERATION_TOKENS,
    )
    source_graph, source_atlas = c_function_token_multigraph(
        SOURCE,
        "doubled_product",
        operation_tokens=OPERATION_TOKENS,
    )
    multiplication_tokens = {
        attributes["expression_token"]
        for _, attributes in source_graph.nodes(data=True)
        if attributes["token_id"] == TOKEN_MUL
    }
    assert len(multiplication_tokens) == 1
    assert len(source_atlas.path(next(iter(multiplication_tokens)))) == 3
    assert all(
        isinstance(attributes["token_id"], int)
        for graph in (source_graph, raised_graph)
        for _, attributes in graph.nodes(data=True)
    )
    assert source_graph.number_of_nodes() != raised_graph.number_of_nodes()
    assert not nx.is_isomorphic(source_graph, raised_graph)
    similarity = topology_similarity(source_graph, raised_graph)
    assert 0.60 <= similarity < 1.0

    # A stronger witness than the scalar score: the compiler's observed CSE
    # is one explicit quotient rewrite. Once identical source Mul nodes are
    # contracted, its typed wiring is the raised machine SSA topology.
    source_after_cse = quotient_common_subexpressions(source_graph)
    assert nx.is_isomorphic(
        source_after_cse,
        raised_graph,
        node_match=lambda left, right: left["token_id"] == right["token_id"],
    )

    regenerated_source = emit_scalar_c(lifted, name="raised_doubled_product")
    regenerated_object = tmp_path / "regenerated.o"
    _compile_object(regenerated_source, regenerated_object)
    relifted = lift_x86_64_affine_function(
        parse_objdump_function(
            disassemble_gnu_object(regenerated_object),
            "raised_doubled_product",
        ),
        argument_registers=("ecx", "edx"),
        argument_names=("x", "y"),
    )
    relifted_graph = ssa_dataflow_multigraph(
        relifted, operation_tokens=OPERATION_TOKENS,
    )
    assert topology_similarity(
        raised_graph, relifted_graph,
    ) >= 0.90
    assert nx.is_isomorphic(
        raised_graph,
        relifted_graph,
        node_match=lambda left, right: left["token_id"] == right["token_id"],
    )

    original = compile_and_load(SOURCE, name="original", directory=tmp_path / "original")
    regenerated = compile_and_load(
        regenerated_source,
        name="regenerated",
        directory=tmp_path / "regenerated",
    )
    original_function = original.function(
        "doubled_product",
        restype=ctypes.c_int32,
        argtypes=(ctypes.c_int32, ctypes.c_int32),
    )
    regenerated_function = regenerated.function(
        "raised_doubled_product",
        restype=ctypes.c_int32,
        argtypes=(ctypes.c_int32, ctypes.c_int32),
    )
    for left, right in ((0, 0), (1, 7), (-4, 9), (123, -51), (-300, -200)):
        assert regenerated_function(left, right) == original_function(left, right)
