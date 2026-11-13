"""Structural preflight cannot move a selected owner twice."""
from pathlib import Path
from tests.test_owned_record_return_source import SOURCE
from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.source_owned_record_transfer import owned_record_transfer_plans


def _lower(source):
    module, _, exports = lower_ast_source_to_ssa(
        source, 'root', extraction_contract=ExtractionContract(
            Path(__file__).resolve().parents[1]/'extraction_contracts/program_extraction.yaml'),
        progress=lambda _: None, progress_bars=False)
    return module, exports[0]


def test_selected_local_owners_plan_without_mutation():
    module, name = _lower(SOURCE)
    function = module.functions[name]
    before = tuple((label, tuple(id(op) for op in block.instrs))
                   for label, block in function.blocks.items())
    plans = owned_record_transfer_plans(module, name)
    assert len(plans) == 1
    assert len(plans[0].lifetimes.original_owner_ids) == 3
    assert tuple(cell.row[-1] for cell in plans[0].member_cells) == plans[0].lifetimes.selection.member_ids
    assert tuple((label, tuple(id(op) for op in block.instrs))
                 for label, block in function.blocks.items()) == before


def test_two_record_fields_cannot_move_one_owner_twice():
    source = SOURCE.replace('    items: list[int]', '    items: list[int]\n    other: list[int]').replace(
        'return Result(items=out)', 'return Result(items=out, other=out)')
    module, name = _lower(source)
    assert owned_record_transfer_plans(module, name) == ()


def test_borrowed_record_field_cannot_be_transferred():
    source = '''from dataclasses import dataclass
@dataclass
class Result:
    items: list[int]
def root(items: list[int]) -> Result:
    return Result(items=items)
'''
    module, name = _lower(source)
    assert owned_record_transfer_plans(module, name) == ()
