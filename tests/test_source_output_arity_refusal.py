"""A Ret that carries fewer values than the source returns is refused.

Program: ``return math.floor(x), int(x)`` (corpus c922).  The region
computing ``math.floor`` was retired, the root's Ret carried one value for
the source's two, and the program returned a 1-tuple with a clean audit.
The final concordance compares the function's ``source_output_value_ids``
with each Ret: fewer values is a named finding.  More is the ABI's
expansion of a None, a union frame or a forwarded tuple, and is admitted.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.compiler.identity_concordance import CorrelationTable
from src.transmogrifier.ssa import Instr, SSAValue


def _module(outputs, returned):
    ret = Instr("Ret", [SSAValue(value_id, dtype="int64") for value_id in returned], None)
    function = SimpleNamespace(
        metadata={"source_output_value_ids": tuple(outputs)},
        blocks={"entry": SimpleNamespace(name="entry", instrs=[ret])},
    )
    return SimpleNamespace(functions={"m__root": function})


def _kinds(outputs, returned):
    return [
        (finding.kind, finding.value_id)
        for finding in CorrelationTable._unreturned_output_findings(
            _module(outputs, returned)
        )
    ]


def test_ret_shorter_than_source_outputs_is_refused_by_name():
    assert _kinds((1, 3), (3,)) == [("source-output-arity-short", 1)]


def test_empty_ret_keeps_its_own_finding():
    assert _kinds((1, 3), ()) == [("source-output-unreturned", 1)]


def test_equal_and_expanded_rets_are_admitted():
    assert _kinds((1, 3), (1, 3)) == []
    # A union frame or forwarded tuple occupies more than one slot.
    assert _kinds((2,), (4, 6, 8)) == []
