"""``x[...] = v`` ingests as ONE full-extent store, not an expanded index.

The dt system's generated ``PieceState.restore`` spells one ``self.col[...] =
col`` per column (839 of them in a 35-piece system).  Ingestion used to rewrite
every ``obj[...]`` into ``obj[tuple([slice(None)] * obj.ndim)]`` (about 25
graph nodes of index arithmetic per occurrence), and the structural fold then
folded that arithmetic once per column again: ``PieceState.restore`` alone was
26,855 nodes.

A subscript whose WHOLE index is the ellipsis is a declared basic-index
literal: ``normalize_basic_index`` resolves it against the parent's declared
shape, exactly as it resolves ``:`` there.  The ellipsis constant stays the one
``index`` operand of the ``Indexed``/``IndexedStore`` node, the store versions
the field's one storage, and the compiled program is the same full-extent
write the explicit expansion produced.

Recognition points:
* ``node_special_cases._EllipsisExpander.visit_Subscript`` -- a sole ``...``
  is left as the authored Subscript (ingestion);
* ``glsl_deployment_strategy._dispatch_metadata_rule`` -- ``Ellipsis`` is a
  basic-index literal, so the ``IndexedStore`` stays NUMERICAL_WORK instead of
  being classified coordinator metadata (``non_numeric_constant_operand``),
  which silently dropped the store.
"""
from __future__ import annotations

import pickle
import subprocess
import sys
import warnings

import numpy as np
import pytest

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.ssa_c_backend import emit_ssa_module_to_c
from src.compiler.ssa_llvm_backend import (
    compile_artifact,
    emit_ssa_function_to_llvm,
    prepare_artifact_execution,
)
from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
    c_backend_repository_ssa_reference,
)

EXTENT = 4
ELLIPSIS = "..."
#: What the ingestion rewrite used to produce for ``self.x[...]``, spelled
#: out in the source (no flag): ``() + tuple([slice(None)] * (x.ndim - 0)) + ()``.
EXPLICIT = "() + tuple([slice(None)] * (self.{name}.ndim - 0)) + ()"
NO_STORE = None


def _source(columns: int, index: str, *, copy: bool = False) -> str:
    names = [f"c{i}" for i in range(columns)]
    lines = ["class PieceState:"]
    lines.append("    def restore(self, snapshot):")
    lines.append(f"        {', '.join(names)}, = snapshot")
    for name in names:
        if index is not NO_STORE:
            lines.append(
                f"        self.{name}[{index.format(name=name)}] = {name}"
            )
    if index is NO_STORE:
        lines.append("        pass")
    if copy:
        lines.append("    def copy_shallow(self):")
        lines.append("        return (")
        for name in names:
            lines.append(f"            self.{name}.copy(),")
        lines.append("        )")
    snapshots = ", ".join(f"s{i}" for i in range(columns))
    lines.append(f"def root(state, {snapshots}):")
    lines.append(f"    state.restore(({snapshots},))")
    lines.append("    return 0.0")
    return "\n".join(lines) + "\n"


def _policy(columns: int) -> ExtractionContract:
    span = {
        "storage": "span", "dtype": "float64", "rank": 1,
        "shape": [EXTENT], "mutable": True,
    }
    return ExtractionContract(
        "extraction_contracts/program_extraction.yaml"
    ).with_execution_file(
        "extraction_contracts/vehicle_full_native_execution.yaml"
    ).with_program_abi({
        "records": {"PieceState": {
            "identity": "PieceState",
            "fields": {f"c{i}": dict(span) for i in range(columns)},
        }},
        "bindings": [
            {"function": "root", "parameter": "state", "record": "PieceState"},
        ],
        "values": [
            {**span, "function": "root", "parameter": f"s{i}",
             "python_type": "numpy.ndarray", "mutable": False}
            for i in range(columns)
        ],
    })


class _Ingested(Exception):
    pass


def _ingested_graph(source: str, columns: int):
    """The ProcessGraph the real lowering builds, stopped before the reducer."""
    import src.common.tensors.topological_reducer as reducer

    captured = {}
    original = reducer.reduce_abstract_tensor_topology

    def stop(graph, *args, **kwargs):
        captured["graph"] = graph
        raise _Ingested

    reducer.reduce_abstract_tensor_topology = stop
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lower_ast_source_to_ssa(
                source, "root", name="ellipsis_count",
                extraction_contract=_policy(columns),
                progress=lambda _message: None,
            )
    except _Ingested:
        pass
    finally:
        reducer.reduce_abstract_tensor_topology = original
    return captured["graph"]


def _nodes_per_column(index: str, *, copy: bool = False) -> float:
    few = _ingested_graph(_source(3, index, copy=copy), 3).G.number_of_nodes()
    many = _ingested_graph(_source(5, index, copy=copy), 5).G.number_of_nodes()
    return (many - few) / 2


def test_full_extent_store_ingests_as_one_indexed_store_per_column():
    # Graph nodes attributable to one store: the store statement's own nodes
    # (Assign, target Subscript, GetAttr self.x, receiver, index, value).
    base = _nodes_per_column(NO_STORE)
    old = _nodes_per_column(EXPLICIT) - base
    new = _nodes_per_column(ELLIPSIS) - base
    colon = _nodes_per_column(":") - base
    # The store costs what the plain ``[:]`` store costs, not the expansion.
    assert new == colon
    assert new <= 7, new
    assert old - new >= 15, (old, new)


def test_copy_shallow_column_cost_is_one_call_per_column():
    assert _nodes_per_column(ELLIPSIS, copy=True) - _nodes_per_column(
        ELLIPSIS
    ) <= 6


def _lower(columns: int, index: str, name: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, _, _exports = lower_ast_source_to_ssa(
            _source(columns, index), "root", name=name,
            extraction_contract=_policy(columns),
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            progress=lambda _message: None,
        )
    return module, module.functions[f"{name}__root"]


def _store_chain_rows(module, columns: int) -> dict:
    """The store-chain rows the lowering posted for the field stores.

    Every ``self.cN[...] = sN`` versions the field's ONE storage:
    * ``reducer_field_state``: an ELEMENT_WRITTEN row per field whose value and
      effect are both the store's one cell;
    * ``record_storage_alias``: the store versions the record field's storage;
    * ``dispatch_store``: the planner's store row, one per column.
    """
    from src.compiler.identity_concordance import identity_book

    book = identity_book(module)
    state = book.page("reducer_field_state")
    written = {}
    for row in state.rows():
        for _column, fact in state.history(row):
            kind = getattr(getattr(fact, "kind", None), "value", None)
            if kind == "element_written":
                # value and effect are both the store: one cell.
                written[row[-1]] = fact.value == fact.effect
    return {
        "element_written": written,
        "record_storage_alias": len(book.page("record_storage_alias").rows()),
        "dispatch_store": len(book.page("dispatch_store").rows()),
    }


@pytest.mark.parametrize("columns", [3, 5])
def test_ellipsis_store_posts_the_store_chain_rows_of_the_explicit_expansion(
    columns,
):
    posted = {
        label: _store_chain_rows(
            _lower(columns, index, f"ell_rows_{label}_{columns}")[0], columns
        )
        for label, index in (
            ("ellipsis", ELLIPSIS), ("colon", ":"), ("explicit", EXPLICIT),
        )
    }
    rows = posted["ellipsis"]
    assert set(rows["element_written"]) == {f"c{i}" for i in range(columns)}
    assert all(rows["element_written"].values()), rows
    assert rows["record_storage_alias"] == columns
    assert rows["dispatch_store"] >= columns
    assert rows == posted["colon"] == posted["explicit"], posted


def _eager(columns: int, snapshots, states):
    namespace: dict = {}
    exec(compile(_source(columns, ELLIPSIS), "<eager>", "exec"), namespace)
    state = namespace["PieceState"].__new__(namespace["PieceState"])
    for i, array in enumerate(states):
        setattr(state, f"c{i}", array.copy())
    namespace["root"](state, *snapshots)
    return [getattr(state, f"c{i}") for i in range(columns)]


def _inputs(columns: int):
    rng = np.random.default_rng(20261007)
    snapshots = [
        rng.standard_normal(EXTENT) * 10.0 ** i for i in range(columns)
    ]
    states = [np.full(EXTENT, 7.5 + i) for i in range(columns)]
    return snapshots, states


def _bits(array) -> np.ndarray:
    return np.ascontiguousarray(array, dtype=np.float64).view(np.uint64)


def _ids(root, columns):
    fields = {
        argument.accounting.get("program_abi_field"): int(argument.id)
        for argument in root.args
        if argument.accounting.get("program_abi_field")
    }
    parameters = dict(root.metadata["parameter_names"])
    return (
        [fields[f"c{i}"] for i in range(columns)],
        [int(parameters[f"s{i}"]) for i in range(columns)],
    )


def _run_llvm(module, root, columns, snapshots, states, directory):
    artifact = emit_ssa_function_to_llvm(module, root.name)
    assert artifact.shortfalls == (), artifact.shortfalls
    native = compile_artifact(artifact, directory=directory)
    state_ids, snapshot_ids = _ids(root, columns)
    feed = {}
    for identifier, array in zip(state_ids, states):
        feed[identifier] = array.copy()
    for identifier, array in zip(snapshot_ids, snapshots):
        feed[identifier] = array.copy()
    execution = prepare_artifact_execution(native, feed)
    execution.run()
    return [np.asarray(execution.buffers[i]).copy() for i in state_ids]


_C_PROBE = """
import pickle, sys
import numpy as np
artifact, state_ids, snapshot_ids, states, snapshots = pickle.load(
    open(sys.argv[1], 'rb'))
feed = {i: a.copy() for i, a in zip(state_ids, states)}
feed.update({i: a.copy() for i, a in zip(snapshot_ids, snapshots)})
result = artifact.prepare_execution(feed).run()
pickle.dump([np.asarray(result.buffers[i]).copy() for i in state_ids],
            open(sys.argv[2], 'wb'))
"""


def _run_c(module, root, columns, snapshots, states, directory):
    artifact = emit_ssa_module_to_c(module, root.name)
    assert artifact.complete, artifact.shortfalls
    artifact.compile(directory)
    state_ids, snapshot_ids = _ids(root, columns)
    payload = directory.parent / (directory.name + ".pkl")
    payload.write_bytes(pickle.dumps((
        artifact, state_ids, snapshot_ids, states, snapshots,
    )))
    probe = subprocess.run(
        [sys.executable, "-c", _C_PROBE, str(payload), str(payload) + ".out"],
        capture_output=True, text=True, timeout=60,
    )
    assert probe.returncode == 0, probe.stdout + probe.stderr
    with open(str(payload) + ".out", "rb") as stream:
        return pickle.load(stream)


@pytest.mark.parametrize("columns", [3, 5])
def test_ellipsis_store_runs_natively_and_matches_eager_and_old_expansion(
    columns, tmp_path,
):
    snapshots, states = _inputs(columns)
    expected = _eager(columns, snapshots, states)
    for column, snapshot in zip(expected, snapshots):
        assert np.array_equal(_bits(column), _bits(snapshot))

    results = {}
    for label, index in (("ellipsis", ELLIPSIS), ("explicit", EXPLICIT)):
        module, root = _lower(columns, index, f"ell_{label}_{columns}")
        results[label, "llvm"] = _run_llvm(
            module, root, columns, snapshots, states,
            tmp_path / f"llvm_{label}",
        )
        results[label, "c"] = _run_c(
            module, root, columns, snapshots, states, tmp_path / f"c_{label}",
        )
    for key, produced in results.items():
        for column in range(columns):
            assert np.array_equal(
                _bits(produced[column]), _bits(expected[column])
            ), (key, column, produced[column], expected[column])
