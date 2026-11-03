"""A linked accepted exception body keeps its original process proof owner."""

from pathlib import Path

from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler.identity_concordance import (
    CorrelationTable, begin_identity_book, end_identity_book, identity_book,
)
from src.transmogrifier.ssa import IRModule


def test_accepted_exception_and_process_origins_read_retained_book():
    source = (
        "def root(x: int) -> int:\n"
        "    if x < 0:\n"
        "        raise ValueError('negative')\n"
        "    return x\n"
    )
    owner_module, _outputs, _exports = lower_ast_source_to_ssa(
        source, "root", name="accepted_exception_origin",
        extraction_contract=(Path(__file__).resolve().parents[1]
                             / "extraction_contracts/program_extraction.yaml"),
        progress=lambda _message: None, progress_bars=False,
    )
    owner = identity_book(owner_module)
    retained = {
        name: function
        for name, function in owner_module.functions.items()
        if any(
            instruction.attributes.get("exception_operation_cell") is not None
            for block in function.blocks.values()
            for instruction in block.instrs
        )
    }
    assert retained
    for function in retained.values():
        function.metadata["accepted_ssa_component"] = {
            "schema": "turing.accepted-ssa-component.v1",
            "artifact_sha256": "accepted-exception-test-digest",
            "identity_book": owner,
        }
    destination, token = begin_identity_book()
    try:
        composed = IRModule(retained, metadata={"identity_book": destination})
        assert not any(
            finding.kind in {"ssa-exception-provenance", "ssa-process-provenance"}
            for finding in (
                *CorrelationTable._operator_safety_findings(composed),
                *CorrelationTable._thread_operation_findings(composed),
            )
        )
        exception = next(
            instruction
            for function in retained.values()
            for block in function.blocks.values()
            for instruction in block.instrs
            if instruction.attributes.get("exception_operation_cell") is not None
        )
        source = exception.attributes.pop("source_process_cell")
        try:
            assert any(
                finding.kind == "ssa-process-provenance"
                for finding in CorrelationTable._thread_operation_findings(composed)
            )
        finally:
            exception.attributes["source_process_cell"] = source
    finally:
        end_identity_book(token)
