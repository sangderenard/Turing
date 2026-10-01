"""A record returned from three sites inside ``while True:``.

The shape of ``dt_controller.step_with_dt_control_used`` cut to seconds:
the returns sit inside a ``while True:`` loop (the top-level guard rewrite
skips loops), one return follows ``m.hard_failure = True``, one follows a
write on another branch, one is the loop body's fall-through.  Lowered
through ``lower_ast_source_to_ssa`` with a ``program_abi`` record binding.

The probe prints whether a record ``return_merge`` Phi exists, every
``record_return_field_selection`` row with its fact and inbound edges,
whether any selection row was revised (and whether each revision is
sourced), then emits C and runs it against CPython for both values of the
branch.  Each native run is a child process with a timeout, because a
lost return edge makes the native loop spin.

Pass = lowers, at least one selection row, every revision sourced,
native == Python.

    python -u tools/compiler_probes/probe_record_return_merge.py

Diagnosis (2026-10-01, HEAD 8462a2a9; read-only hooks, no compiler edit)
-------------------------------------------------------------------------
Outcome: lowers; a record return-merge Phi exists (expanded into one
``return_merge`` field Phi per field, each carrying ``record_phi``); six
selection rows, all ``Unresolved``, each with one inbound edge (the record
Phi's ``ssa_value`` cell); no row revised.  Native disagrees with Python:
(rejected=False, value=2.0) returns hard_failure True (Python False), and
(rejected=False, value=0.5) never returns (Python returns).

Fault 1 -- the fall-through return is lost (control, before SSA):

1. ``LoopComposer.describe`` records three ``return_controls`` for the
   loop, every one anchored at the returned value's node -- the parameter
   record ``m`` -- so all three share ONE anchor id for three authored
   sites.
2. ``expanded_loop_body_nodes`` (loop_composer.py, the ``return`` block
   of ``body_items``) keeps a return only ``if node_id in
   lexical_position``.  The anchor is a parameter, not a body node, so
   all three are dropped (hooked at that line: every anchor
   ``in lexical_position`` is False).  ``planned_root`` builds the
   WhileBlock with no return control and no terminal control.
3. ``arm_return_control`` (glsl_deployment_strategy.py, the conditional
   builder) re-creates a return only when it is the terminal statement of
   an ``if`` arm.  The two arm returns come back that way; the loop-body
   return has no such restorer.
4. SSA: ``if_merge.1`` branches to ``while_latch``; the loop has no exit
   edge.  The record return-merge Phi's third predecessor is the
   post-loop fall-through (``while_exit``), which is unreachable and
   pruned; the finished field Phis have two incoming edges.  Native
   spins whenever neither arm returns.

Fault 2 -- the unwritten field at the second site takes the first site's
write (record return selection):

1. Receipts (``record_return_state_receipts``): site A (``m.hard_failure
   = True; return m``) holds ``(m, hard_failure)``; site B holds only
   ``(m, value)``; site C (lost) holds ``(m, value)``.  An unwritten
   field is not in a site's receipt.
2. ``scalar_return_field_versions.lookup`` maps a predecessor to its
   sites by RETURN-SLOT VALUE IDENTITY (``return_slot_values`` equal to
   the edge's ``return_source_value_ids``).  All three sites return the
   same record id, so every predecessor matches all three sites; any
   site lacking the key exits ``SITE_WITHOUT_FIELD_STATE`` -- for both
   fields at both surviving predecessors.  One identity, three sites.
   No ``return_site_field_state`` cell is read (the ``read`` is the Phi
   cell alone).
3. ``select_return_arguments`` keeps the fallback argument: the record
   descriptor's field value, which is flow-insensitive -- for
   ``hard_failure`` it is the Const True written in site A's arm.  The
   finished ``hard_failure`` Phi selects that Const on both edges.
4. The parameter's incoming ``hard_failure`` never becomes a formal, so
   the native program cannot return it on the non-writing path.

The selection rows are sourced and none is revised; the record selection
is wrong because the predecessor-to-site key is the returned value's
identity, not the return site's.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import warnings

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

CONTRACTS = REPO / "extraction_contracts"
BUILD = REPO / "build" / "record_return_merge"
NAME = "record_return_merge"
TIMEOUT_S = 90

SOURCE = '''
from dataclasses import dataclass


@dataclass
class Metrics:
    hard_failure: bool = False
    value: float = 0.0


def step(m: Metrics, rejected: bool) -> Metrics:
    while True:
        if rejected:
            m.hard_failure = True
            return m
        if m.value > 1.0:
            m.value = m.value * 0.25
            return m
        return m
'''

# (rejected, m.value): both branch values of ``rejected``; with rejected
# False, one value takes the second arm and one the fall-through.
CASES = ((True, 2.0), (False, 2.0), (False, 0.5))


def lower():
    from src.compiler.extraction_contract import ExtractionContract
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa

    contract = (
        ExtractionContract(CONTRACTS / "program_extraction.yaml")
        .with_program_abi({
            "records": {"Metrics": {
                "identity": f"{NAME}.Metrics",
                "fields": {
                    "hard_failure": {"storage": "scalar", "dtype": "bool",
                                     "mutable": True},
                    "value": {"storage": "scalar", "dtype": "float64",
                              "mutable": True},
                },
            }},
            "bindings": [
                {"function": "step", "parameter": "m", "record": "Metrics"},
            ],
            "values": [
                {"function": "step", "parameter": "rejected",
                 "storage": "scalar", "dtype": "bool", "rank": 0,
                 "python_type": "builtins.bool"},
            ],
        })
        .with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return lower_ast_source_to_ssa(
            SOURCE, "step", name=NAME, python_bindings={},
            extraction_contract=contract, runtime_closure_only=True,
        )


def root_function(module):
    return module.functions[f"{NAME}__step"]


def python_result(rejected, value):
    namespace: dict = {}
    exec(SOURCE, namespace)  # noqa: S102 -- the probe's own source
    m = namespace["step"](namespace["Metrics"](False, value), rejected)
    return {"hard_failure": bool(m.hard_failure), "value": float(m.value)}


def native_child(rejected, value) -> int:
    """Lower, emit C, run one case; print one JSON line."""
    import numpy as np

    from src.compiler.ssa_c_backend import emit_ssa_module_to_c

    module, outputs, _ = lower()
    root = root_function(module)
    named = dict(root.metadata.get("parameter_names") or ())
    feeds = {}
    for argument in root.args:
        accounting = dict(argument.accounting or {})
        parameter = accounting.get("program_abi_parameter")
        field = accounting.get("program_abi_field")
        if parameter == "m" and field == "value":
            feeds[int(argument.id)] = np.asarray(value, dtype=np.float64)
        elif parameter == "m" and field == "hard_failure":
            feeds[int(argument.id)] = np.asarray(False, dtype=np.bool_)
        elif int(argument.id) == int(named.get("rejected", -1)):
            feeds[int(argument.id)] = np.asarray(rejected, dtype=np.bool_)
    artifact = emit_ssa_module_to_c(module, root.name)
    if not artifact.complete:
        print(json.dumps({"error": f"C emission incomplete: {artifact.shortfalls}"}))
        return 1
    workdir = BUILD / f"r{int(rejected)}_v{str(value).replace('.', 'p')}"
    workdir.mkdir(parents=True, exist_ok=True)
    artifact.compile(workdir / NAME)
    result = artifact.prepare_execution(feeds).run()
    native = {}
    by_id = {int(item.res.id): item for block in root.blocks.values()
             for item in block.instrs if item.res is not None}
    for output in outputs[root.name]:
        definition = by_id.get(int(output.id))
        field = None if definition is None else (
            (definition.attributes or {}).get("record_field")
        )
        cell = np.asarray(result.buffers[output.id]).reshape(-1)[0]
        native[str(field)] = bool(cell) if field == "hard_failure" else float(cell)
    print("NATIVE " + json.dumps(native), flush=True)
    return 0


failures: list[str] = []


def check(label: str, condition: bool) -> None:
    print(("ok   " if condition else "FAIL ") + label, flush=True)
    if not condition:
        failures.append(label)


def main() -> int:
    from src.compiler.concordance_declarations import (
        RECORD_RETURN_FIELD_SELECTION,
    )
    from src.compiler.identity_concordance import Ref, Unresolved, identity_book

    try:
        module, _outputs, _exports = lower()
    except Exception as error:  # noqa: BLE001 -- a raise is the finding
        check(f"the program lowers ({type(error).__name__}: {error})", False)
        return 1
    check("the program lowers", True)
    book = identity_book(module)
    step = root_function(module)

    # ---- the record return-merge Phi --------------------------------------
    merges = [
        item for block in step.blocks.values() for item in block.instrs
        if item.op == "Phi"
        and (item.attributes or {}).get("binding") == "return_merge"
    ]
    record_merges = [
        item for item in merges
        if (item.attributes or {}).get("record_phi") is not None
    ]
    print(f"return_merge Phis: {len(merges)} "
          f"(expanded from a record Phi: {len(record_merges)})")
    for item in record_merges:
        attributes = item.attributes or {}
        print(f"    field={attributes.get('record_field')!r} "
              f"incoming={tuple(attributes.get('incoming_blocks', ()))!r} "
              f"record_return_scalar={attributes.get('record_return_scalar')!r}")
    check("a record return_merge Phi exists", bool(record_merges))
    return_edges = [
        name for name, block in step.blocks.items()
        if block.instrs and (block.instrs[-1].attributes or {}).get(
            "return_source_value_ids") is not None
    ]
    print(f"return edges in the finished function: {len(return_edges)} "
          f"(authored returns: 3)")

    # ---- the selection rows -------------------------------------------------
    stored = book.pages.get(RECORD_RETURN_FIELD_SELECTION.name)
    rows = [] if stored is None else list(stored.rows())
    print(f"record_return_field_selection rows: {len(rows)}")
    revised = 0
    unsourced_revisions = 0
    for row in rows:
        history = list(stored.history(row))
        print(f"  field={row[2]} position={row[3]} predecessor={row[4]} "
              f"revisions={len(history)}")
        previous_sources = None
        for column, fact in history:
            ref = Ref(RECORD_RETURN_FIELD_SELECTION, row, column)
            if isinstance(fact, Unresolved):
                print(f"    [{column}] Unresolved({fact.reason.name})")
            else:
                print(f"    [{column}] {fact!r}")
            edges = book.edges_into(ref)
            if not edges:
                print("        (no inbound edge)")
            for source, stage in edges:
                print(f"        <- {source!r} [{stage.name}]")
            sources = {source for source, _stage in edges}
            if previous_sources is not None:
                revised += 1
                # A revision the api admitted names a changed source cell.
                if not sources or sources == previous_sources:
                    unsourced_revisions += 1
            previous_sources = sources
    check("at least one selection row", bool(rows))
    check("every selection row has an inbound edge", all(
        bool(book.edges_into(Ref(RECORD_RETURN_FIELD_SELECTION, row, column)))
        for row in rows for column, _fact in stored.history(row)
    ))
    print(f"revised selection rows: {revised}")
    check("every revision derives from a changed cell", unsourced_revisions == 0)

    # ---- native vs CPython --------------------------------------------------
    for rejected, value in CASES:
        expected = python_result(rejected, value)
        label = f"rejected={rejected} value={value}"
        try:
            run = subprocess.run(
                [sys.executable, "-u", __file__, "--native",
                 str(int(rejected)), repr(value)],
                capture_output=True, text=True, timeout=TIMEOUT_S, cwd=str(REPO),
            )
        except subprocess.TimeoutExpired:
            check(f"native == python  {label}  python={expected} "
                  f"native=(no return within {TIMEOUT_S}s)", False)
            continue
        line = next((text for text in run.stdout.splitlines()
                     if text.startswith("NATIVE ")), None)
        if line is None:
            tail = (run.stdout + run.stderr).strip().splitlines()[-3:]
            check(f"native == python  {label}  python={expected} "
                  f"native=(failed: {tail})", False)
            continue
        native = json.loads(line[len("NATIVE "):])
        check(f"native == python  {label}  python={expected} native={native}",
              native == expected)

    print(f"failures: {len(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--native":
        sys.exit(native_child(bool(int(sys.argv[2])), float(sys.argv[3])))
    sys.exit(main())
