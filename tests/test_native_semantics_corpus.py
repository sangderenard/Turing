"""Authored programs compiled to native C and executed against Python.

Each file in ``tests/native_semantics_corpus`` defines ``root`` and ends with
``# INPUTS [...]``, the argument sets it is run with.  The program is lowered
through the sanctioned entry, ``lower_ast_source_to_ssa``, under the
repository's extraction contract, emitted by the C module lane, compiled and
run; the native result must equal the Python result for every input.  The
lowered module must also leave the identity concordance with no finding.

The corpus records miscompiles that a clean audit did not catch: ``a / b`` on
two ints truncated, a structural ``Ret`` replaced the if-merge of a returned
name by the pre-``if`` loop result, a region fed a settled ``int64`` was cast
to its cut-time ``float64`` formal, and a list's arena became a double buffer
read back as float64.
"""

from __future__ import annotations

import ast
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

CORPUS = Path(__file__).with_name("native_semantics_corpus")
PROGRAMS = sorted(CORPUS.glob("c*.py"))
CONTRACT = Path(__file__).resolve().parents[1] / "extraction_contracts" / "program_extraction.yaml"

_RUNNER = r'''
import pickle, sys
import numpy as np
artifact, parameters, outputs, cases = pickle.load(open(sys.argv[1], "rb"))
classes = {"builtins.int": int, "builtins.float": float, "builtins.bool": bool,
           "builtins.NoneType": type(None)}
for feeds, expected in cases:
    execution = artifact.prepare_execution({
        parameters[name]: np.array([value]) for name, value in feeds.items()
    }).run()
    got = []
    for output in outputs:
        if isinstance(output, tuple):
            # A value union's frame: the tag names the live arm, and only
            # that arm's payload is the value, with that arm's class.
            tag_id, arms = output
            name, payload_id = arms[int(execution.buffers[tag_id].reshape(-1)[0])]
            got.append((name, None if payload_id is None
                        else execution.buffers[payload_id].reshape(-1)[0].item()))
            continue
        got.extend(float(item) for item in execution.buffers[output].reshape(-1).tolist())
    assert len(got) == len(expected), (feeds, got, expected)
    for left, right in zip(got, expected):
        if isinstance(left, tuple):
            name, payload = left
            assert classes.get(name) is type(right), (feeds, got, expected)
            left = None if right is None else float(payload)
            if right is None:
                continue
        right = float(right)
        assert abs(left - right) <= 1e-12 * max(1.0, abs(right)), (feeds, got, expected)
'''


def _unannotated_formals(source: str) -> list[str]:
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "root":
            arguments = node.args
            return [
                argument.arg
                for argument in (
                    *arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs,
                )
                if argument.annotation is None
            ]
    return []


def _inputs(source: str) -> list[dict]:
    return ast.literal_eval(source.split("# INPUTS", 1)[1].strip())


@pytest.mark.parametrize("program", PROGRAMS, ids=lambda path: path.stem)
def test_native_program_matches_python(program, tmp_path):
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import CorrelationTable
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    source = program.read_text()
    # A class defined with no __name__ gets __module__ "builtins", which
    # sends the static-binding walk through the builtins module itself.
    # ``dont_inherit``: this file's ``from __future__ import annotations``
    # must not turn the program's annotations into strings.
    namespace: dict = {"__name__": f"corpus_{program.stem}"}
    exec(compile(source, str(program), "exec", dont_inherit=True), namespace)
    inputs = _inputs(source)
    cases = []
    for feeds in inputs:
        expected = namespace["root"](**feeds)
        cases.append((feeds, list(expected) if isinstance(expected, tuple) else [expected]))

    # The test IS the caller: for formals the program leaves unannotated,
    # each INPUTS row is a root call record whose argument classes feed the
    # compiler's called inventory, and one exact-typed copy of the entry is
    # lowered per distinct class signature (as tools/native_corpus_typed_check
    # does).  Annotated formals are untouched.
    free = _unannotated_formals(source)
    if free:
        from src.compiler.called_inventory import class_spelling

        groups: dict = {}
        for feeds, case in zip(inputs, cases):
            signature = tuple((name, class_spelling(feeds[name])) for name in sorted(free))
            groups.setdefault(signature, []).append(case)
        calls = [{name: feeds[name] for name in free} for feeds in inputs]
        runs = [
            (group, dict(calls=calls, signature=dict(signature)))
            for signature, group in groups.items()
        ]
    else:
        runs = [(cases, None)]

    for index, (group, inventory) in enumerate(runs):
        extra = (
            {} if inventory is None
            else dict(root_calls=inventory["calls"], root_signature=inventory["signature"])
        )
        module, _outputs, exports = lower_ast_source_to_ssa(
            source, "root", name=f"corpus_{program.stem}_{index}",
            extraction_contract=str(CONTRACT), progress=lambda _message: None,
            **extra,
        )
        findings = CorrelationTable.build(module).findings(module)
        assert findings == [], findings

        root = module.functions[exports[0]]
        parameters = dict(root.metadata["parameter_names"])
        returns = [
            instruction.args
            for block in root.blocks.values()
            for instruction in block.instrs
            if instruction.op == "Ret"
        ]
        # A bounds check or an absent key leaves through a fault-exit return
        # that carries the same output identities as the program's own.
        assert len([
            instruction for block in root.blocks.values()
            for instruction in block.instrs
            if instruction.op == "Ret" and not instruction.attributes.get("fault_exit")
        ]) == 1
        ids = [int(value.id) for value in returns[0]]
        # A value union leaves the program as its frame (an int64 tag and one
        # payload cell per arm, ``module.value_union_tables``); it is read as
        # one value at its first member's position.
        table = getattr(module, "value_union_tables", {}).get(exports[0])
        frames = {}
        for union in ([] if table is None else dict(table.unions).values()):
            members = {int(union.tag_value_id)} | {
                int(value_id) for arm in union.alternatives for value_id in arm.value_ids
            }
            if members <= set(ids):
                frames[min(ids.index(member) for member in members)] = (union, members)
        covered = {member for _union, members in frames.values() for member in members}
        outputs = []
        for position, value_id in enumerate(ids):
            if position in frames:
                union, _members = frames[position]
                outputs.append((int(union.tag_value_id), [
                    (str(arm.name), int(arm.value_ids[0]) if arm.value_ids else None)
                    for arm in union.alternatives
                ]))
            elif value_id not in covered:
                outputs.append(value_id)
        artifact = emit_ssa_to_c(module, exports[0])
        assert artifact.complete, artifact.shortfalls
        run_dir = tmp_path / f"run{index}"
        run_dir.mkdir()
        artifact.compile(run_dir / "native")
        saved = run_dir / "artifact.pkl"
        saved.write_bytes(pickle.dumps((artifact, parameters, outputs, group)))
        result = subprocess.run(
            [sys.executable, "-c", _RUNNER, str(saved)],
            capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_unbounded_local_list_is_an_explicit_boundary():
    """A list grown by a data-bounded ``while`` has no proven capacity.

    It used to be given a one-element private arena whose ``full`` status
    nobody read, so ``len(xs)`` returned 1.  Dynamic growth is the declared
    unsupported boundary; emission must say so instead of computing.
    """
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    source = (
        "def root(n: int) -> int:\n"
        "    xs = []\n"
        "    i = 0\n"
        "    while i < n:\n"
        "        xs.append(i * 2)\n"
        "        i += 1\n"
        "    return len(xs)\n"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="corpus_unbounded_list",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    artifact = emit_ssa_to_c(module, exports[0])
    assert not artifact.complete
    assert any(
        "dynamic-growth-unavailable" in shortfall.reason
        for shortfall in artifact.shortfalls
    )


def test_guard_nested_in_a_loop_arm_is_placed_inside_that_arm():
    """``if i > 20 or not allow: raise`` inside ``if x == 6:`` in a loop.

    The loop composer placed the inner guard at the loop-body level after the
    outer conditional, where ``i > 20`` (computed only inside the arm) does not
    dominate it: audit ``use-not-dominated`` (``_validate_row``'s ``if item is
    NEW:`` arm in the bootstrap).  The guard belongs inside its arm.  The C
    module lane has no validation runtime, so this is checked by the audit.
    """
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import CorrelationTable

    source = (
        "def root(n: int, allow: bool) -> int:\n"
        "    out = 0\n"
        "    for i in range(n):\n"
        "        x = (i * 3) % 7\n"
        "        if x == 6:\n"
        "            if i > 20 or not allow:\n"
        "                raise ValueError('bad')\n"
        "            out += 100\n"
        "            continue\n"
        "        if x > 10:\n"
        "            raise ValueError('never')\n"
        "        out += x\n"
        "    return out\n"
    )
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name="corpus_nested_loop_guard",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    assert CorrelationTable.build(module).findings(module) == []


@pytest.mark.parametrize("statement", ["return d[n]", "del d[n]\n    return len(d)"])
def test_absent_mapping_key_raises_where_python_raises_key_error(statement, tmp_path):
    """``d[n]`` and ``del d[n]`` with ``n`` absent raise ``KeyError``.

    The lookup returned 0 and the delete went on (both helpers knew the key
    was absent and reported it only in a status nobody read).  A present key
    still returns Python's value.
    """
    import numpy as np

    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.ssa_c_backend import emit_ssa_to_c

    source = (
        "def root(n: int) -> int:\n"
        "    d = {1: n, 2: 5}\n"
        f"    {statement}\n"
    )
    namespace: dict = {}
    exec(source, namespace)
    module, _outputs, exports = lower_ast_source_to_ssa(
        source, "root", name="corpus_absent_key",
        extraction_contract=str(CONTRACT), progress=lambda _message: None,
    )
    root = module.functions[exports[0]]
    parameters = dict(root.metadata["parameter_names"])
    outputs = [
        instruction.args
        for block in root.blocks.values()
        for instruction in block.instrs
        if instruction.op == "Ret"
    ][0]
    artifact = emit_ssa_to_c(module, exports[0])
    assert artifact.complete, artifact.shortfalls
    artifact.compile(tmp_path / "native")

    def run(n: int):
        execution = artifact.prepare_execution(
            {parameters["n"]: np.array([n])}
        ).run()
        return float(execution.buffers[outputs[0].id].reshape(-1)[0])

    assert run(1) == float(namespace["root"](1))
    with pytest.raises(KeyError):
        namespace["root"](7)
    with pytest.raises(Exception):
        run(7)
