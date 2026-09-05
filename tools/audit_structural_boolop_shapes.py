"""Fast, real audit: which BoolOp-operand shapes ``ensure_structural_value``
(fortran_c_shell.recover_structural_source_outputs) handles.

Modelled exactly on tools/repro_no_exchange_observed.py: the REAL functions
from dt_controller.py/dt_scaler.py via ``inspect.getsource`` and the REAL
program ABI from ``balloon_tire_managed_extraction_contract``.  Never the
vehicle build, never ``lower_balloon_tire_managed_python_ssa``.

Usage::

    python tools/audit_structural_boolop_shapes.py no_exchange
    python tools/audit_structural_boolop_shapes.py energy_time_limit
    python tools/audit_structural_boolop_shapes.py shadow_dt_limit
    python tools/audit_structural_boolop_shapes.py all

For every case it prints LOWERED/FAILED, then every function's
``metadata["structural_output_shortfalls"]`` from the returned module, and a
dump (via ``resolved_process_graph_sink``) of every graph node whose
``expr_obj`` is a Compare/BoolOp/UnaryOp/IfExp/Call/Attribute/Constant --
its ``type``/``op``/``source_type`` and whether it carries the
``coordinator_short_circuit`` tag -- so the op each shape ARRIVES with is
evidence, not inference.
"""

from __future__ import annotations

import ast
import inspect
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.vehicle_python_compilation import (  # noqa: E402
    balloon_tire_managed_extraction_contract, BalloonTireManagedState,
)
from src.common.dt_system import dt_controller  # noqa: E402
from src.common.dt_system.dt_scaler import _scalar  # noqa: E402

CONTRACTS = Path(__file__).resolve().parents[1] / "extraction_contracts"


def _base_records():
    import numpy as np

    stub = BalloonTireManagedState.__new__(BalloonTireManagedState)
    for name in (
        "inputs", "state", "output", "wheel_input_indices", "rest",
        "face_vertices", "face_rest", "face_scatter", "bending_incidence",
        "bending_scatter", "bending_weight", "vertex_area", "bead_mask",
        "face_material", "telemetry",
    ):
        setattr(stub, name, np.zeros((1,), dtype=np.float64))
    return balloon_tire_managed_extraction_contract(stub).program_abi.receipt()


CASES = {
    # name: (functions whose real source is included, root source, extra
    # bindings beyond metrics/targets)
    "no_exchange": (
        (_scalar, dt_controller._no_exchange_observed),
        "def root(metrics, targets):\n"
        "    return _no_exchange_observed(metrics, targets)\n",
    ),
    "energy_time_limit": (
        (_scalar, dt_controller._energy_time_limit),
        "def root(metrics, targets):\n"
        "    return _energy_time_limit(metrics, targets)\n",
    ),
    "shadow_dt_limit": (
        (_scalar, dt_controller._shadow_dt_limit),
        "def root(dt_tensor, metrics, targets):\n"
        "    return _shadow_dt_limit(dt_tensor, metrics, targets)\n",
    ),
}

INTERESTING = (
    ast.Compare, ast.BoolOp, ast.UnaryOp, ast.IfExp, ast.Call,
    ast.Attribute, ast.Constant, ast.Dict,
)


def _dump_graph(graph, functions_of_interest: tuple[str, ...]):
    """Print op/type/source_type/short-circuit tag per interesting node."""

    # The sink hands over the whole-program graph; each authored function's
    # canonical post-reduction graph hangs off its function-table entry.
    graphs = [("<root>", getattr(graph, "G", graph))]
    for entry in getattr(graph, "function_table", ()) or ():
        function_graph = getattr(getattr(entry, "graph", None), "G", None)
        if function_graph is not None:
            graphs.append((
                str(function_graph.graph.get("function_name") or entry.name),
                function_graph,
            ))
    print("--- resolved process graph nodes (expr shape -> op) ---")
    rows = []
    seen = set()
    for owner, G in graphs:
      for node_id, data in G.nodes(data=True):
        if (owner, int(node_id)) in seen:
            continue
        seen.add((owner, int(node_id)))
        expression = data.get("expr_obj")
        # The root (whole-program) graph is huge; filter it to the shapes
        # under audit.  Per-function graphs are small: dump every node so
        # rewritten nodes (expr_obj=None) are visible too.
        if owner == "<root>" and not isinstance(expression, INTERESTING):
            continue
        attributes = data.get("attributes") or {}
        try:
            text = ast.unparse(expression) if expression is not None else (
                f"label={data.get('label')!r} parents={data.get('parents')!r}"
            )
        except Exception:  # pragma: no cover - defensive
            text = type(expression).__name__
        if len(text) > 90:
            text = text[:87] + "..."
        shape = type(expression).__name__ if expression is not None else "-"
        if isinstance(expression, ast.Compare):
            shape += "[" + ",".join(
                type(op).__name__ for op in expression.ops
            ) + "]"
        elif isinstance(expression, ast.BoolOp):
            shape += "[" + type(expression.op).__name__ + "]"
        elif isinstance(expression, ast.UnaryOp):
            shape += "[" + type(expression.op).__name__ + "]"
        rows.append((
            owner, int(node_id), shape, str(data.get("type")), str(data.get("op")),
            str(attributes.get("source_type")),
            bool(attributes.get("coordinator_short_circuit")),
            str(attributes.get("static_python_reference") or ""),
            text,
        ))
    rows.sort()
    for row in rows:
        owner, node_id, shape, ntype, op, source_type, tagged, static_ref, text = row
        print(
            f"  {owner:<22} node={node_id:<5} {shape:<22} type={ntype:<14} op={op:<14} "
            f"source_type={source_type:<10} short_circuit={str(tagged):<5} "
            f"{('static=' + static_ref + ' ') if static_ref else ''}"
            f"| {text}"
        )


class _ArrivalTrace:
    """Log what ``ensure_structural_value`` sees, without touching src/.

    ``ensure_structural_value`` is a closure nested inside
    ``recover_structural_source_outputs``; a profile hook keyed on that code
    name reads its ``value_id`` argument and its free variable ``graph`` on
    entry, so the op/type/source_type each node ARRIVES with is recorded
    verbatim (this is how the "disguised as call" question is settled).
    """

    def __init__(self):
        self.rows = []
        self._seen = set()

    def __call__(self, frame, event, arg):
        if event != "call" or frame.f_code.co_name != "ensure_structural_value":
            return
        locals_ = frame.f_locals
        value_id = locals_.get("value_id")
        graph = locals_.get("graph")
        # ``symbol`` is not a free variable of the closure; ``function`` is.
        symbol = getattr(locals_.get("function"), "name", None)
        if value_id is None or graph is None:
            return
        key = (symbol, int(value_id))
        if key in self._seen:
            return
        self._seen.add(key)
        data = graph.nodes.get(int(value_id), {})
        attributes = data.get("attributes") or {}
        expression = data.get("expr_obj")
        try:
            text = ast.unparse(expression) if expression is not None else (
                f"label={data.get('label')!r}"
            )
        except Exception:  # pragma: no cover - defensive
            text = type(expression).__name__
        self.rows.append((
            str(symbol), int(value_id), str(data.get("type")),
            str(data.get("op")), str(attributes.get("source_type")),
            bool(attributes.get("coordinator_short_circuit")),
            bool(int(value_id) in (locals_.get("values") or {})),
            text[:80],
        ))


def _declare_optional_targets(targets_record: dict) -> dict:
    """The real ``Targets`` receipt plus its two optional float fields.

    ``balloon_tire_managed_extraction_contract`` declares only the four
    ``Targets`` fields the tire program reads.  Declaring
    ``energy_exchange_fraction``/``shadow_growth_max`` the exact way the same
    receipt already declares ``Metrics.dt_limit`` (``scalar float64``,
    ``default None``) is what a program using the energy side-chain would
    carry; without it every ``getattr(targets, <field>, None)`` folds to the
    ``None`` default and the functions under test collapse to ``return None``
    before any BoolOp operand exists.
    """

    import copy

    record = copy.deepcopy(targets_record)
    template = None
    for name, field in record["fields"].items():
        if field.get("storage") == "scalar" and field.get("dtype") == "float64":
            template = copy.deepcopy(field)
            template.pop("default", None)
            break
    assert template is not None
    for name in ("energy_exchange_fraction", "shadow_growth_max"):
        field = copy.deepcopy(template)
        field["default"] = None
        record["fields"][name] = field
    return record


def _print_trace(trace: _ArrivalTrace) -> None:
    print("--- ensure_structural_value arrivals (op as seen INSIDE the closure) ---")
    if not trace.rows:
        print("  (never entered)")
    for symbol, value_id, ntype, op, source_type, tagged, present, text in trace.rows:
        print(
            f"  {symbol.rsplit('__', 1)[-1]:<22} node={value_id:<4} type={ntype:<14} "
            f"op={op:<14} source_type={source_type:<10} short_circuit={str(tagged):<5} "
            f"already_in_values={str(present):<5} | {text}"
        )


def run_case(case: str, *, declare_optional_targets: bool = False) -> int:
    functions, root_source = CASES[case]
    real_source = "\n\n".join(inspect.getsource(f) for f in functions)
    # dt_controller.py imports ``math`` at module scope; the authored
    # functions reference it, so the compilation unit gets the same import.
    source = "import math\n\n" + real_source + "\n\n" + root_source
    base = _base_records()
    targets_record = base["records"]["Targets"]
    if declare_optional_targets:
        targets_record = _declare_optional_targets(targets_record)
    contract_abi = {
        "records": {
            "Metrics": base["records"]["Metrics"],
            "Targets": targets_record,
        },
        "bindings": [
            {"function": "*", "parameter": "metrics", "record": "Metrics"},
            {"function": "*", "parameter": "targets", "record": "Targets"},
        ],
        "values": [],
    }
    policy = ExtractionContract(
        CONTRACTS / "program_extraction.yaml"
    ).with_program_abi(contract_abi)

    captured = {}

    def sink(graph):
        captured["graph"] = graph

    print(
        f"=== case {case}: {', '.join(f.__name__ for f in functions)}"
        f"{' [Targets ABI + optional fields]' if declare_optional_targets else ''}"
        " ==="
    )
    trace = _ArrivalTrace()
    t0 = time.time()
    sys.setprofile(trace)
    try:
        module, outputs, exports = lower_ast_source_to_ssa(
            source, "root", name=f"audit_{case}", extraction_contract=policy,
            resolved_process_graph_sink=sink,
        )
    except Exception as error:
        sys.setprofile(None)
        print(f"FAILED after {time.time()-t0:.2f}s: {type(error).__name__}: "
              f"{str(error)[:800]}", flush=True)
        _print_trace(trace)
        if "graph" in captured:
            _dump_graph(captured["graph"], ())
        return 1
    finally:
        sys.setprofile(None)
    print(f"LOWERED in {time.time()-t0:.2f}s", flush=True)
    _print_trace(trace)
    function_table = getattr(module, "functions", None) or {}
    if isinstance(function_table, dict):
        items = function_table.items()
    else:
        items = ((getattr(f, "name", "?"), f) for f in function_table)
    print("--- structural_output_shortfalls per function ---")
    any_shortfall = False
    for name, function in items:
        metadata = getattr(function, "metadata", {}) or {}
        shortfalls = metadata.get("structural_output_shortfalls")
        recovered = metadata.get("recovered_structural_outputs")
        if shortfalls or recovered:
            print(f"  {name}: shortfalls={shortfalls!r} recovered={recovered!r}")
        any_shortfall = any_shortfall or bool(shortfalls)
    if not any_shortfall:
        print("  (none)")
    if "graph" in captured:
        _dump_graph(captured["graph"], ())
    return 0


def main(argv) -> int:
    os.environ.setdefault("TURING_DEBUG_STRUCTURAL_OUTPUTS", "1")
    arguments = list(argv[1:])
    declare = "--declare-optional-targets" in arguments
    arguments = [a for a in arguments if a != "--declare-optional-targets"]
    selected = arguments or ["no_exchange"]
    if selected == ["all"]:
        selected = list(CASES)
    status = 0
    for case in selected:
        status |= run_case(case, declare_optional_targets=declare)
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
