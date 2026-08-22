"""Graph-only (unfed symbolic parameter) precompile: real structural region
programs, and the WebAssembly backend lowering its own layout/host-boundary
ops.

An unfed mutable parameter (the decoder's ``subject``/``state``) is left a
symbolic SSA input -- the one case an autograd tape cannot observe. That path
builds region programs structurally from the planner's dispatch subgraphs
(``_structural_region_program_from_subgraph`` -> the same
``dispatch_region_to_fused_program`` the tape path uses), not by hand-synthesis.
Layout ops (``reshape``) and host-boundary reinterprets (``tobytes``) ride
through the transcriber under their own names; the WebAssembly backend lowers
them to a linear identity, exactly as the C backend does for itself.

These compile in well under a second each -- the manageable-pace correctness
fixtures for the checkpoint pipeline.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.common.tensors.accelerator_backends.aot_compile import (
    compile_ast_aot,
    _walk_planned_shells,
)
from src.compiler.glsl_deployment_strategy import (
    _structural_region_program_from_subgraph,
)
from src.compiler.fused_program_wasm_backend import emit_wasm_module


def _graph_only_regions(source: str, entrypoint: str, function_name: str):
    """Compile through the graph-only path and return that function's shell's
    structural region programs."""

    aot = compile_ast_aot(
        source,
        entrypoint,
        {"buf": np.asarray([0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0],
                           dtype=np.int64)},
        precompile_only=True,
        remove_loops=True,
        mutable_parameters=("state",),
        checkpoint=False,
    )
    shell = next(
        s for s in _walk_planned_shells(aot.deployment)
        if s.process_graph.G.graph.get("function_name") == function_name
        and s.dispatch_subgraphs
    )
    return [
        _structural_region_program_from_subgraph(subgraph)
        for subgraph in shell.dispatch_subgraphs
    ]


_RESHAPE_SOURCE = (
    "def decode_block(state, buf):\n"
    "    lo = buf & 0x0F\n"
    "    hi = (buf >> 4) & 0x0F\n"
    "    packed = (hi << 4) | lo\n"
    "    state.last = packed\n"
    "    return packed.reshape(2, 4)\n"
)

_TOBYTES_SOURCE = (
    "def emit_state(state, buf):\n"
    "    packed = (buf & 0x0F) | ((buf >> 4) << 4)\n"
    "    state.last = packed\n"
    "    return packed.tobytes()\n"
)


def test_graph_only_reshape_builds_region_and_emits_wasm():
    regions = _graph_only_regions(_RESHAPE_SOURCE, "decode_block", "decode_block")
    assert regions, "graph-only path produced no structural region programs"
    program = regions[0].program
    op_names = [step.op_name for step in program.steps]
    # The bitwise chain plus the faithfully-transcribed reshape (not dropped,
    # not translated in the builder).
    assert "reshape" in op_names
    assert {"bitand", "shr", "shl", "bitor"} <= set(op_names)
    # The backend lowers the reshape (a view) to a complete module.
    module = emit_wasm_module(program, name="decode_block_r0", dtype="int64")
    assert module.complete, module.shortfall_report()


def test_graph_only_region_preserves_deterministic_boundary_identities():
    program = _graph_only_regions(
        _RESHAPE_SOURCE, "decode_block", "decode_block",
    )[0].program

    identity_tokens = (program.extras or {}).get("ssa_identity_tokens", {})

    assert identity_tokens
    assert set(identity_tokens) <= {
        *program.feeds, *program.outputs.values(),
    }
    assert all(tuple(tokens) for tokens in identity_tokens.values())


def test_graph_only_tobytes_builds_region_and_emits_wasm():
    regions = _graph_only_regions(_TOBYTES_SOURCE, "emit_state", "emit_state")
    assert regions, "graph-only path produced no structural region programs"
    program = regions[0].program
    assert "tobytes" in [step.op_name for step in program.steps]
    # tobytes is a host-boundary reinterpret; the backend lowers it to a linear
    # identity of its operand rather than refusing it as non-elementwise.
    module = emit_wasm_module(program, name="emit_state_r0", dtype="int64")
    assert module.complete, module.shortfall_report()
