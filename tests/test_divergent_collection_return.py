"""Return sites that return different lists.

``return [n]`` on one edge and ``return []`` on another are two arenas;
the caller has one result.  A list born in the function dies at its
return, so each edge copies its list into the one result list (lanes plan
A6; corpus c541-c544).  A parameter's list returned on one edge is the
caller's own object: copying it would split one object in two, so that is
still refused by name until the caller-side alias check exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CONTRACT = str(
    Path(__file__).resolve().parents[1]
    / "extraction_contracts" / "program_extraction.yaml"
)


def test_a_returned_parameter_list_beside_a_local_list_is_refused_by_name():
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    source = (
        "def pick(xs: list[int], n: int) -> list[int]:\n"
        "    out = [n, n + 1]\n"
        "    if n > 2:\n"
        "        return out\n"
        "    return xs\n"
        "\n\n"
        "def root(n: int) -> int:\n"
        "    base = [7, 8, 9]\n"
        "    t = pick(base, n)\n"
        "    return len(t) * 100 + t[0] * 10 + t[-1]\n"
    )
    with pytest.raises(ValueError, match="only lists born in the function"):
        lower_ast_source_to_ssa(
            source, "root", name="x", extraction_contract=CONTRACT,
            progress=lambda _: None,
        )


MATERIALIZED_SOURCE = (
    "def pick(n: int):\n"
    "    xs = [n, n + 1, n + 2]\n"
    "    if n > 3:\n"
    "        return 1, []\n"
    "    if n > 1:\n"
    "        return 2, [n * 5]\n"
    "    return 0, list(xs)\n"
    "\n\n"
    "def root(n: int):\n"
    "    status, t = pick(n)\n"
    "    total = status * 1000 + len(t) * 100\n"
    "    for v in t:\n"
    "        total = total * 7 + v\n"
    "    return total\n"
)
MATERIALIZED_INPUTS = (0, 1, 2, 3, 4, 9)


def _materialized_python(n):
    namespace = {}
    exec(MATERIALIZED_SOURCE, namespace)
    return namespace["root"](n)


def _run_materialized(tmp_path, backend):
    import numpy as np

    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    module, _outputs, exports = lower_ast_source_to_ssa(
        MATERIALIZED_SOURCE, "root", name="materialized_divergent_return",
        extraction_contract=CONTRACT, progress=lambda _: None,
    )
    root = module.functions[exports[0]]
    parameters = dict(root.metadata["parameter_names"])
    output = next(
        int(value.id)
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
        and not instruction.attributes.get("fault_exit")
        for value in instruction.args
    )
    if backend == "c":
        from src.compiler.ssa_c_backend import emit_ssa_to_c

        artifact = emit_ssa_to_c(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        artifact.compile(tmp_path / "c")
        prepare = artifact.prepare_execution
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )

        emitted = emit_ssa_function_to_llvm(module, exports[0])
        assert not emitted.shortfalls, emitted.shortfalls
        artifact = compile_artifact(emitted, directory=tmp_path / "llvm")

        def prepare(feeds):
            return prepare_artifact_execution(artifact, feeds)
    return [
        int(prepare(
            {parameters["n"]: np.array([n], dtype=np.int64)}
        ).run().buffers[output].reshape(-1)[0])
        for n in MATERIALIZED_INPUTS
    ]


@pytest.mark.parametrize("backend", ["c", "llvm"])
def test_a_list_materialization_is_born_in_the_function(tmp_path, backend):
    """``return 0, list(xs)`` beside ``return 1, []`` (string core
    ``expand``/``cover`` shape): ``list(...)`` builds a new list, so it is
    copied into the one result like a display, not refused."""

    assert _run_materialized(tmp_path, backend) == [
        _materialized_python(n) for n in MATERIALIZED_INPUTS
    ]
