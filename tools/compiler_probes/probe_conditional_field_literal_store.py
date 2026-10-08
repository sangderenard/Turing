"""A literal stored to a record field on ONE arm must be stored on that arm.

    python -u tools/compiler_probes/probe_conditional_field_literal_store.py [c|llvm|both] [variant ...]

Each variant is a small class whose method ``step(self, x)`` stores a literal
under ``if x > self.threshold``:

    scalar        if x > self.threshold: self.phase = 1.0
    element       if x > self.threshold: self.cells[1] = 1.0
    increment     if x > self.threshold: self.phase = self.phase + 1.0   (control)
    scalar_else   if x > self.threshold: self.phase = 1.0 / else: self.phase = 0.0

The method is lowered through ``lower_ast_source_to_ssa`` with a real
extraction contract and a declared ABI (the receiver is a record), run
natively on the C lane and the LLVM lane for five steps with the condition
false for the first three and true after, and compared bit-for-bit with the
same Python object stepped the same way.  The defect: the literal's store was
placed after the constant's producer (the entry block), so the field changed
on step 1 although the condition was false.

Exit status is 0 when every (variant, lane) matches.
"""
from __future__ import annotations

import math
import pathlib
import sys
import warnings

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (  # noqa: E402
    c_backend_repository_ssa_reference,
)
from src.compiler.extraction_contract import ExtractionContract  # noqa: E402
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa  # noqa: E402

CONTRACTS = REPO / "extraction_contracts"
XS = (0.0, 0.0, 0.0, 2.0, 2.0)          # threshold 1.0: false x3, then true
THRESHOLD = 1.0

CLASS_HEAD = '''
class Machine:
    def __init__(self):
        self.threshold = 1.0
        self.phase = {phase}
        self.cells = [0.0, 0.0, 0.0]
{extra_init}
    def step(self, x):
'''

BODIES = {
    "scalar": ("0.0", '''\
        if x > self.threshold:
            self.phase = 1.0
        return x
'''),
    "element": ("0.0", '''\
        if x > self.threshold:
            self.cells[1] = 1.0
        return x
'''),
    "increment": ("0.0", '''\
        if x > self.threshold:
            self.phase = self.phase + 1.0
        return x
'''),
    # a value computed BEFORE the ``if`` and stored inside it: its producer
    # dominates the arm, so "right after the producer" is every path
    "outside_value": ("0.0", '''\
        y = x * 2.0
        if x > self.threshold:
            self.phase = y
        return x
'''),
    # an int literal into a float field
    "int_literal": ("0.0", '''\
        if x > self.threshold:
            self.phase = 1
        return x
'''),
    # the same field stored on two arms of a chain
    "elif_chain": ("5.0", '''\
        if x > 3.0:
            self.phase = 3.0
        elif x > self.threshold:
            self.phase = 1.0
        return x
'''),
    "scalar_else": ("5.0", '''\
        if x > self.threshold:
            self.phase = 1.0
        else:
            self.phase = 0.0
        return x
'''),
    # the field is READ after the conditional write and returned: the read is
    # the merge of the arm's version and the incoming one, not the arm's Load
    "return_field": ("0.0", '''\
        if x > self.threshold:
            self.phase = self.phase + 1.0
        return self.phase
'''),
    # control: no conditional at all -- the read after the write
    "return_field_straight": ("0.0", '''        self.phase = self.phase + x
        return self.phase
'''),
    # a REFERENCE field written inside the ``if``: it has no arm Store (its
    # source is a StaticRef, not a scalar), so where it is stored is the
    # placement row, not the source's producer
    "ref_none": ("0.0", '''        if x > self.threshold:
            self.link = None
        return x
'''),
    # a STATIC reference (a compile-time Python object) into the reference
    # field inside the ``if``
    "ref_static": ("0.0", '''        if x > self.threshold:
            self.link = math.sqrt
        return x
'''),
    "return_field_literal": ("5.0", '''\
        if x > self.threshold:
            self.phase = 1.0
        return self.phase
'''),
}


# variants that also declare a reference field ``link`` (optional, mutable).
# Python starts it at a non-None object; None-ness is what is compared.
REFERENCE_VARIANTS = {"ref_none", "ref_static"}
LINK_ANCHOR = 123456789          # the native handle fed for the initial object


def source_of(variant):
    phase, body = BODIES[variant]
    extra = (
        "        self.link = 'anchor'" + chr(10)
        if variant in REFERENCE_VARIANTS else ""
    )
    return CLASS_HEAD.format(phase=phase, extra_init=extra) + body


def contract(variant=None):
    fields = {
        "threshold": {"storage": "scalar", "dtype": "float64",
                      "rank": 0, "mutable": True},
        "phase": {"storage": "scalar", "dtype": "float64",
                  "rank": 0, "mutable": True},
        "cells": {"storage": "span", "dtype": "float64",
                  "rank": 1, "shape": [3], "mutable": True},
    }
    if variant in REFERENCE_VARIANTS:
        fields["link"] = {"storage": "reference", "mutable": True}
    return (
        ExtractionContract(CONTRACTS / "program_extraction.yaml")
        .with_program_abi({
            "records": {"Machine": {
                "identity": "Machine",
                "fields": fields,
            }},
            "bindings": [
                {"function": "*step", "parameter": "self",
                 "record": "Machine"},
            ],
            "values": [
                {"function": "*step", "parameter": "x",
                 "storage": "scalar", "dtype": "float64", "rank": 0,
                 "python_type": "builtins.float"},
            ],
        })
        .with_execution_file(CONTRACTS / "vehicle_full_native_execution.yaml")
    )


def link_state(is_none, is_anchor):
    return "none" if is_none else "anchor" if is_anchor else "reference"


def python_run(variant):
    namespace: dict = {"math": math}
    exec(compile(source_of(variant), "<authored>", "exec"), namespace)
    machine = namespace["Machine"]()
    trace = []
    for x in XS:
        result = machine.step(x)
        row = (float(result), float(machine.phase),
               tuple(float(c) for c in machine.cells))
        if variant in REFERENCE_VARIANTS:
            row += (link_state(machine.link is None,
                               machine.link == 'anchor'),)
        trace.append(row)
    return trace


def native_run(variant, lane):
    name = f"cfls_{variant}_{lane}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        module, outputs, _exports = lower_ast_source_to_ssa(
            source_of(variant), "Machine.step", name=name,
            extraction_contract=contract(variant),
            tensor_ssa_reference=c_backend_repository_ssa_reference(),
            python_bindings={"math": math},
        )
    qualified = next(
        symbol for symbol in module.functions
        if symbol.endswith("__step") and "planned_region" not in symbol
    )
    function = module.functions[qualified]
    # A write-only field may be declared by more than one formal (its
    # incoming slot and the receiver column); every one is fed, and every one
    # must show the Python object's value.
    fields: dict = {}
    for value in function.args:
        field_name = (value.accounting or {}).get("program_abi_field")
        if field_name:
            fields.setdefault(field_name, []).append(int(value.id))
    parameters = dict(function.metadata["parameter_names"])
    published = int(outputs[qualified][0].id)
    if lane == "c":
        from src.compiler.ssa_c_backend import emit_ssa_module_to_c
        artifact = emit_ssa_module_to_c(module, qualified)
        if not artifact.complete:
            raise RuntimeError(f"C shortfalls: {artifact.shortfalls}")
        artifact.compile(REPO / "build" / name, optimization="O0")
        prepare = artifact.prepare_execution
    else:
        from src.compiler.ssa_llvm_backend import (
            compile_artifact, emit_ssa_function_to_llvm,
            prepare_artifact_execution,
        )
        artifact = emit_ssa_function_to_llvm(module, qualified,
                                             entry_name=name)
        if artifact.shortfalls:
            raise RuntimeError(f"LLVM shortfalls: {artifact.shortfalls}")
        native = compile_artifact(artifact, directory=REPO / "build" / name,
                                  optimization="O0")
        prepare = lambda feeds: prepare_artifact_execution(native, feeds)  # noqa: E731
    initial_phase = float(BODIES[variant][0])
    initial = {"threshold": [THRESHOLD], "phase": [initial_phase],
               "cells": [0.0, 0.0, 0.0], "link": [LINK_ANCHOR]}
    # structural ``None`` is the native zero sentinel (a NoneValue operation)
    none_handles = {0}
    # A field the method never touches is not an argument of its function.
    feeds = {
        value_id: np.array(initial[field_name])
        for field_name, ids in fields.items() for value_id in ids
    }
    feeds[parameters["x"]] = np.array([0.0])
    feeds[published] = np.zeros(1)
    execution = prepare(feeds)
    trace = []
    for x in XS:
        execution.buffers[parameters["x"]][0] = x
        execution.run()
        shown = {}
        for field_name in ("phase", "cells"):
            copies = [
                tuple(float(c) for c in execution.buffers[value_id])
                for value_id in fields.get(field_name, ())
            ] or [tuple(float(c) for c in initial[field_name])]
            # all formals of one field must agree; disagreement is reported
            shown[field_name] = copies[0] if len(set(copies)) == 1 else ("DISAGREE", *copies)
        row = (
            float(np.asarray(execution.buffers[published]).reshape(-1)[0]),
            shown["phase"][0] if shown["phase"][0] != "DISAGREE" else shown["phase"],
            shown["cells"],
        )
        if variant in REFERENCE_VARIANTS:
            links = {
                int(np.asarray(execution.buffers[value_id]).reshape(-1)[0])
                for value_id in fields.get("link", ())
            }
            row += (link_state(links <= none_handles,
                               links == {LINK_ANCHOR}),)
        trace.append(row)
    return trace


def main(argv):
    lanes = {"c": ("c",), "llvm": ("llvm",)}.get(
        argv[0] if argv else "both", ("c", "llvm"))
    variants = [a for a in argv if a in BODIES] or list(BODIES)
    bad = 0
    for variant in variants:
        expected = python_run(variant)
        print(f"{variant}: python  (return, phase, cells) per step:")
        for step, row in enumerate(expected):
            print(f"    {step} x={XS[step]} {row}")
        for lane in lanes:
            try:
                actual = native_run(variant, lane)
            except Exception as error:  # noqa: BLE001
                print(f"{variant}/{lane}: FAILED {type(error).__name__}: "
                      f"{str(error)[:300]}")
                bad += 1
                continue
            ok = actual == expected
            print(f"{variant}/{lane}: {'MATCH' if ok else 'MISMATCH'}")
            if not ok:
                bad += 1
                for step, (a, e) in enumerate(zip(actual, expected)):
                    print(f"    {step} native={a} python={e}"
                          + ("" if a == e else "   <-- differs"))
    print("ALL_MATCH" if not bad else f"{bad} MISMATCH(ES)")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
