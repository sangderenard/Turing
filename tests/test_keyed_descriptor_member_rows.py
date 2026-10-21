"""A keyed mapping parameter's descriptor members post physical fields.

Program: a root taking ``d: dict[int, float]``.  Wrong record: the members'
int64 length/capacity words and column spans got accounting by role only; their
``ssa_value`` rows carried no storage/rank/arena (audit
``ssa-physical-unposted``).  Rule: the member's own row is posted, derived from
its sequence cells, and accounting follows the fact.
"""
from __future__ import annotations

from pathlib import Path

from src.compiler.concordance_declarations import SSA_VALUE
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import CorrelationTable
from src.compiler.ssa_record_return_state import function_scope_of

ROOT = Path(__file__).resolve().parents[1]

# A captured keyed memo: ``visit`` is linked to ``root``'s keyed parameter.
SOURCE = """
def root(memo, key):
    def visit(value):
        return value in memo
    return visit(key)
"""


def _contract():
    return ExtractionContract(
        ROOT / "extraction_contracts/program_extraction.yaml"
    ).with_program_abi({
        "records": {}, "bindings": [], "values": [
            {"function": "root", "parameter": "memo", "storage": "keyed",
             "dtype": "opaque_ref", "mutable": True,
             "key_encoding": "integer_identity", "python_type": "builtins.dict"},
            {"function": "root", "parameter": "key", "storage": "scalar",
             "dtype": "int64", "python_type": "builtins.int"},
        ],
    })


def test_keyed_descriptor_members_carry_physical_rows():
    module, _, _ = lower_ast_source_to_ssa(
        SOURCE, "root", name="keyed_members",
        extraction_contract=_contract(), runtime_closure_only=True,
        progress=lambda message: None, progress_bars=False,
    )
    book = module.metadata["identity_book"]
    page = book.page(SSA_VALUE)
    checked = 0
    for fname, fn in module.functions.items():
        sequences = getattr(module.sequence_tables.get(fname), "sequences", {})
        formals = {int(f.id): f for f in fn.args}
        scope = function_scope_of(fn)
        for descriptor in sequences.values():
            for member in (
                *descriptor.column_value_ids, descriptor.length_address_id,
                descriptor.capacity_value_id, descriptor.status_address_id,
                descriptor.live_flags_value_id,
            ):
                formal = formals.get(member)
                if formal is None or (formal.accounting or {}).get(
                    "linked_parameter_provenance"
                ) != "exact_call_argument_sequence_member":
                    continue
                fact = page.latest((scope, int(member)))
                capacity = member == descriptor.capacity_value_id
                assert fact.storage == ("scalar" if capacity else "span")
                assert fact.rank == (0 if capacity else 1)
                assert fact.arena == (member in descriptor.column_value_ids)
                assert formal.accounting["program_abi_storage"] == fact.storage
                assert formal.accounting["program_abi_rank"] == fact.rank
                checked += 1
    assert checked
    findings = CorrelationTable.build(module).findings(module)
    assert not [f for f in findings if f.kind == "ssa-physical-unposted"]
    assert not [f for f in findings if "unsourced" in f.kind], findings
