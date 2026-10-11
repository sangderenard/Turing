"""Canonical expansions survive bounded residency without reminting values."""
from pathlib import Path

import pytest

from src.compiler import hierarchical_plan as plans
from src.compiler.identity_concordance import begin_identity_book, end_identity_book


def test_expansion_and_planning_views_share_one_resident_bound(monkeypatch):
    calls = []
    original = plans.plan_region_to_ssa_instrs
    def observe(region, **options):
        calls.append(region.closure_id)
        return original(region, **options)
    monkeypatch.setattr(plans, "plan_region_to_ssa_instrs", observe)
    regions = tuple(plans.PlanClosure(
        f"region_{index}", (index * 4, index * 4 + 1, index * 4 + 2),
        (plans.PlanLine.create("max", inputs=(index * 4, index * 4 + 1, index * 4 + 2),
                               outputs=(index * 4 + 3,)),),
        closure_id=index, value_shapes=tuple((index * 4 + n, (), "float64") for n in range(4)),
    ) for index in range(12))
    book, token = begin_identity_book()
    try:
        expansion = plans.expand_plan_regions(plans.PlanClosure("root", (), regions),
                                              first_free_value_id=48, resident_limit=1)
        pool = expansion.pool
        planning = pool.view("planned")
        baseline = {}
        for region in regions:
            instructions = expansion[(region.name, region.closure_id)]
            baseline[region.closure_id] = tuple((i.op, tuple(a.id for a in i.args), i.res.id) for i in instructions)
            planning[region.closure_id] = instructions
        for index in reversed(range(12)):
            instructions = planning[index]
            assert tuple((i.op, tuple(a.id for a in i.args), i.res.id) for i in instructions) == baseline[index]
        assert calls == list(range(12))
        assert len({row[0][2] for row in baseline.values()}) == 12
        assert all(row[0][2] >= 48 for row in baseline.values())
        assert pool.stats["snapshots"] == 24
        assert pool.stats["resident_peak"] == len(pool.resident) == 1
        assert pool.stats["spills"] == 24 and pool.stats["reloads"] > 12
        assert book.page("compile_policy").latest(("ssa_resident_regions",)) == "1"
        paths = (pool.store.path, pool.store.index_path)
        pool.close()
        assert all(not Path(path).exists() for path in paths)
        with pytest.raises(RuntimeError, match="closed"):
            expansion[("region_0", 0)]
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_instruction_residency_rejects_invalid_admission(limit):
    with pytest.raises(ValueError, match="positive region count"):
        plans.RegionInstructionStore(limit)


def test_copied_occurrences_cannot_retype_the_canonical_snapshot():
    from src.transmogrifier.ssa import Instr, SSAValue
    _book, token = begin_identity_book()
    try:
        pool = plans.RegionInstructionStore(1)
        view = pool.view("expanded")
        view[0] = (Instr("Add", [SSAValue(0, "float64", accounting={"rank": 0})],
                         SSAValue(1, "float64", accounting={"rank": 0})),)
        copy = plans.copy_region_instructions(view[0])
        copy[0].args[0].accounting["rank"] = 1
        copy[0].res.accounting["rank"] = 1
        view[1] = (Instr("Ret", [], None),)
        assert view[0][0].args[0].accounting == {"rank": 0}
        assert view[0][0].res.accounting == {"rank": 0}
        pool.close()
    finally:
        end_identity_book(token)


def test_concordance_detects_a_false_residency_receipt():
    from src.transmogrifier.ssa import Function, IRModule
    from src.compiler.identity_concordance import CorrelationTable
    _book, token = begin_identity_book()
    try:
        pool = plans.RegionInstructionStore(1)
        function = Function("root", [], {}, metadata={"region_instruction_residency": {
            "schema": "turing.region-instruction-residency.v1", "limit": 1,
            "resident_peak": 2, "resident_instructions_peak": 2,
            "snapshots": 2, "spills": 1, "reloads": 0,
        }})
        module = IRModule({function.name: function})
        assert CorrelationTable._region_instruction_residency_findings(module)[0].kind == (
            "region-instruction-residency-disagreement")
        pool.close()
    finally:
        end_identity_book(token)
