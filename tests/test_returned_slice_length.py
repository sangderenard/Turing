"""A returned slice keeps the exact collection that its store chain writes."""
from pathlib import Path

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import concordance_report

ROOT = Path(__file__).resolve().parents[1]


def test_returned_mutated_slice_length_is_defined_before_typed_append():
    source = (ROOT / "tests/record_view_corpus/c636_returned_mutated_slice_length.py").read_text()
    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name="returned_mutated_slice",
        extraction_contract=ROOT / "extraction_contracts/program_extraction.yaml",
        progress=lambda _: None, progress_bars=False,
    )
    put_name = next(name for name in module.functions
                    if "__put__specialized" in name
                    and "__planned_region_" not in name and "__ssa_sequence_" not in name)
    cover_name = next(name for name in module.functions
                      if "__cover__specialized" in name
                      and "__planned_region_" not in name and "__ssa_sequence_" not in name)
    cover = module.functions[cover_name]
    # The returned slice owns a physical sequence, rather than being treated
    # as the scalar row of the list written inside the loop.
    slice_id, = cover.metadata["source_output_value_ids"]
    assert module.sequence_tables[cover_name].by_id(slice_id) is not None
    function = module.functions[put_name]
    instructions = [instruction for block in function.blocks.values()
                    for instruction in block.instrs]
    definitions = {instruction.res.id: instruction for instruction in instructions
                   if instruction.res is not None}
    append = next(instruction for instruction in instructions
                  if instruction.attributes.get("ssa_sequence_operation") == "append")
    converted = append.args[-1]
    assert converted.dtype == "uint32"
    cast = definitions[converted.id]
    assert cast.op == "Cast"
    length = definitions[cast.args[0].id]
    assert length.op == "Load"
    assert length.attributes.get("binding") == "ssa_sequence_length_value"
    assert length.res.dtype == "int64"
    report = concordance_report(module)
    assert "0 finding(s)" in report, report
