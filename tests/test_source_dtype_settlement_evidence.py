"""Source ProgramABI class settlement reads the published value identity."""

from pathlib import Path

from src.compiler.extraction_contract import ExtractionContract
from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
from src.compiler import source_evidence_resolver
from src.compiler.source_evidence_resolver import ClassEvidence


def test_authored_optional_class_settlement_reads_exact_source_class(monkeypatch):
    seen = []
    original = source_evidence_resolver.require_class

    def recording_query(scope, value_id, *, book):
        answer = original(scope, value_id, book=book)
        if isinstance(answer, ClassEvidence):
            seen.append((answer, book))
        return answer

    monkeypatch.setattr(source_evidence_resolver, "require_class", recording_query)
    contract = ExtractionContract(
        Path(__file__).resolve().parents[1]
        / "extraction_contracts" / "program_extraction.yaml"
    )
    module, _outputs, exports = lower_ast_source_to_ssa(
        "class Registry:\n"
        "    pass\n"
        "def accepts(registry: Registry | None) -> bool:\n"
        "    return registry is None\n",
        "accepts",
        extraction_contract=contract,
        progress=lambda _message: None,
        progress_bars=False,
    )
    none_type = tuple(
        (answer, book) for answer, book in seen
        if answer.class_identity == "builtins.NoneType"
    )
    assert none_type
    assert all(answer.subject_cell == answer.class_cell
               and answer.proof_cells
               and book.latest_ref(answer.class_cell.page,
                                   answer.class_cell.row) == answer.class_cell
               for answer, book in none_type)
    function = module.functions[exports[0]]
    assert function.metadata["native_value_type_abi"][0].alternatives[0].endswith(
        ".Registry"
    )
