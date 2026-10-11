"""Bounded parity gates for the serial descriptor's compiled integer law."""
import dataclasses
import json
from pathlib import Path
import pickle
import random
import subprocess
import sys

import numpy as np

from src.compiler.native_compiler_accelerators import (
    BROADCAST_SHAPES, CompilerAcceleratorRegistry, _python_broadcast_shapes,
    broadcast_shape, broadcast_shapes, compile_and_register_broadcast_shapes_accelerator,
)


def _pairs_and_expected():
    pairs = [((), ()), ((), (0,)), ((0,), (1,)), ((1,), (0,)),
             ((2, 1, 3), (4, 3)), ((2,), (3,)), ((2**60, 1), (1, 7))]
    expected = [(), (0,), (0,), (0,), (2, 4, 3), None, (2**60, 7)]
    generator = random.Random(9831)
    for _ in range(400):
        left = tuple(generator.randrange(5) for _ in range(generator.randrange(7)))
        right = tuple(generator.randrange(5) for _ in range(generator.randrange(7)))
        pairs.append((left, right))
        try:
            expected.append(np.broadcast_shapes(left, right))
        except ValueError:
            expected.append(None)
    return pairs, expected


def test_eager_transfer_is_the_existing_exact_broadcast_law():
    pairs, expected = _pairs_and_expected()
    assert _python_broadcast_shapes(pairs) == tuple(expected)
    assert broadcast_shapes(()) == ()
    assert broadcast_shape((0,), (1,)) == (0,)


def test_own_compiler_builds_shape_transfer_and_native_workers_agree(tmp_path, monkeypatch):
    from src.compiler import fortran_c_shell, ssa_fortran_backend
    calls = []
    canonical = fortran_c_shell.lower_ast_source_to_ssa

    def observed(source, entrypoint, **kwargs):
        calls.append(entrypoint)
        return canonical(source, entrypoint, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("no alternate source compiler")

    monkeypatch.setattr(fortran_c_shell, "lower_ast_source_to_ssa", observed)
    monkeypatch.setattr(ssa_fortran_backend, "compile_module", forbidden)
    registry = CompilerAcceleratorRegistry()
    registry.register_fallback(BROADCAST_SHAPES, _python_broadcast_shapes, name="test:eager-shapes")
    provider = compile_and_register_broadcast_shapes_accelerator(
        tmp_path, registry=registry, progress=lambda _message: None,
    )
    assert calls == ["broadcast_shapes"]
    assert provider.metadata["source_compiler"] == "src.compiler.fortran_c_shell.lower_ast_source_to_ssa"
    assert provider.metadata["concordance"].splitlines()[0].endswith(", 0 finding(s)")
    pairs, expected = _pairs_and_expected()
    artifact = dataclasses.replace(provider.metadata["artifact"], emission=None,
        _entry=None, _library=None, _validation_error_reset=None, _validation_error_take=None)
    payload = tmp_path / "native-shapes.pkl"
    payload.write_bytes(pickle.dumps((artifact, provider.metadata["abi"]["parameters"],
                                    ("shape_pairs",), pairs)))
    worker = Path(__file__).resolve().parents[1] / "tools/compiler_probes/graph_kernel_native_worker.py"
    completed = subprocess.run([sys.executable, "-u", str(worker), str(payload)],
                               capture_output=True, text=True, timeout=15)
    print(completed.stdout, end="", flush=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr[-2500:]
    receipt = json.loads(next(line.split("=", 1)[1] for line in completed.stdout.splitlines()
                            if line.startswith("GRAPH_KERNEL_SHAPES=")))
    assert receipt == [list(shape) if shape is not None else None for shape in expected]


def test_descriptor_keeps_unknown_guard_and_original_shape_publication(monkeypatch):
    from src.compiler.glsl_deployment_strategy import _tensor_descriptor
    from src.compiler.identity_concordance import (
        begin_identity_book, end_identity_book, proven_shape_of, shape_scope_of,
    )
    from src.compiler.native_compiler_accelerators import compiler_accelerators
    from src.transmogrifier.graph.graph_express2 import ProcessGraph
    calls = []
    original = compiler_accelerators.resolve(BROADCAST_SHAPES)

    def observed(pairs):
        pairs = tuple(pairs)
        calls.extend(pairs)
        return _python_broadcast_shapes(pairs)

    # Exercise the native dispatch branch without loading unverified code.
    monkeypatch.setitem(compiler_accelerators._native, BROADCAST_SHAPES,
        dataclasses.replace(original, run=observed, tier="native"))
    for unknown in (False, True):
        graph = ProcessGraph(materialize_memory=False)
        graph.G.graph["function_name"] = "shape_admission"
        graph.G.add_node(1, type="Constant", op="const", value_id=1,
                         constant=np.zeros((2, 1)), attributes={}, parents=[])
        if unknown:
            graph.G.add_node(2, type="Call", op="Call", value_id=2,
                             tensor={"shape": (), "dtype": "float64", "rank": 2},
                             attributes={}, parents=[])
        else:
            graph.G.add_node(2, type="Constant", op="const", value_id=2,
                             constant=np.zeros((1, 3)), attributes={}, parents=[])
        graph.G.add_node(3, type="add", op="add", value_id=3, attributes={},
                         parents=[(1, "lhs"), (2, "rhs")])
        graph.G.add_edges_from(((1, 3), (2, 3)))
        _book, token = begin_identity_book()
        try:
            before = len(calls)
            descriptor = _tensor_descriptor(graph, 3)
            if unknown:
                assert len(calls) == before
                assert proven_shape_of(shape_scope_of(graph), 3) != (2, 1)
            else:
                assert calls[-1] == ((2, 1), (1, 3))
                assert descriptor == {"shape": (2, 3), "dtype": "float64", "rank": 2}
                assert proven_shape_of(shape_scope_of(graph), 3) == (2, 3)
        finally:
            end_identity_book(token)
