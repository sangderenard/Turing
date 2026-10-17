"""Real source producers must leave no migration worklist behind."""
from src.compiler.identity_concordance import CorrelationTable
from tools.audit_identity_concordance import _lower_mapping


def test_mapping_loop_producers_have_complete_sources():
    module = _lower_mapping()
    table = CorrelationTable.build(module)
    assert not table.findings(module)
    groups, identities = table._unsourced_worklist(module)
    assert not groups and not identities


def test_an_imported_function_is_measured_against_the_book_that_minted_its_ids():
    """An accepted SSA body keeps its own book: the composing unit's book
    holds no mint rows for the ids that body minted, and the detector reads
    them where the Novel posts were made."""
    from types import SimpleNamespace

    from src.compiler.identity_concordance import (
        IdentityBook, Mode, NEW, Novel, Registry, RowField, RowFieldKind,
    )
    from src.compiler.id_space import MINTED, has_flag

    registry = Registry()
    page = registry.declare_page("imported_value", (
        RowField("scope", RowFieldKind.SCOPE),
        RowField("id", RowFieldKind.VALUE_ID),
    ), object)
    stage = registry.declare_stage("probe")
    root = registry.declare_transform("probe_root", 0)
    owner = IdentityBook(registry=registry)
    composing = IdentityBook(registry=registry)
    cell = owner.post(
        page, ("f", NEW), "v", stage=stage, provenance=Novel(root, ()),
        mode=Mode.CONCORD,
    )
    minted = int(cell.row[1])
    assert has_flag(minted, MINTED)

    table = CorrelationTable()
    table.row("imported", minted)
    imported = SimpleNamespace(metadata={"accepted_ssa_component": {
        "schema": "turing.accepted-ssa-component.v1",
        "artifact_sha256": "0" * 64, "identity_book": owner,
    }})
    own = SimpleNamespace(metadata={})

    def measured(function):
        module = SimpleNamespace(
            metadata={"identity_book": composing},
            functions={"imported": function},
        )
        return table._unsourced_worklist(module)[1]

    assert measured(imported) == []
    # The composing book alone never minted it: without the owner it is a
    # mint outside any Novel post, as before.
    assert measured(own) == [("imported", minted)]


_RECORD_PAGES = {
    "record_field_access", "record_field_access_path",
    "record_field_layout_concordance", "record_scalar_shape_concordance",
    "field_slot_storage_concordance", "record_forwarding_edge",
}


def _lower_and_check(source, entry_pages):
    from pathlib import Path

    from src.common.tensors.accelerator_backends.c_backend_llvm_ssa import (
        c_backend_repository_ssa_reference,
    )
    from src.compiler.fortran_c_shell import lower_ast_source_to_ssa
    from src.compiler.identity_concordance import identity_book

    module, _outputs, _exports = lower_ast_source_to_ssa(
        source, name="record_page_writers",
        extraction_contract=(
            Path(__file__).resolve().parents[1]
            / "extraction_contracts" / "program_extraction.yaml"
        ),
        tensor_ssa_reference=c_backend_repository_ssa_reference(),
        progress=lambda _message: None,
    )
    book = identity_book(module)
    for name in entry_pages:
        page = book.pages.get(name)
        assert page is not None and list(page.rows()), name
    table = CorrelationTable.build(module)
    return [
        finding for finding in table.findings(module)
        if finding.kind == "unsourced-fact" and finding.function in _RECORD_PAGES
    ]


def test_record_forwarding_pages_are_written_through_the_book():
    """access / path / layout / scalar-shape / forwarding-edge rows are all
    posted DERIVED; none is a raw primitive write."""
    assert _lower_and_check(
        """
from dataclasses import dataclass

@dataclass
class Box:
    a: float = 0.0
    b: float = 0.0

    def get(self) -> float:
        return self.a + self.b

    def twice(self) -> float:
        return self.get() * 2.0
""",
        ("record_field_access", "record_field_access_path",
         "record_field_layout_concordance", "record_scalar_shape_concordance",
         "record_forwarding_edge"),
    ) == []


def test_nested_record_slot_storage_is_written_through_the_book():
    assert _lower_and_check(
        """
from dataclasses import dataclass

@dataclass
class Inner:
    x: float = 0.0

@dataclass
class Outer:
    inner: Inner
    scale: float = 1.0

    def bump(self, v: float) -> float:
        self.inner = Inner(v)
        return self.inner.x * self.scale
""",
        ("field_slot_storage_concordance", "record_scalar_shape_concordance"),
    ) == []


def test_record_pages_have_no_raw_page_writers():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "compiler"
    shell = (root / "fortran_c_shell.py").read_text(encoding="utf-8")
    precompile = (root / "precompile_to_ssa.py").read_text(encoding="utf-8")
    assert not re.search(r"record_field_access_path\.concord\(", shell)
    assert not re.search(r"record_field_access\[[^\]]+\]\s*=", shell)
    assert not re.search(r"record_forwarding_edges\[[^\]]+\]\s*=", shell)
    assert not re.search(r"record_field_concordance\.(set|concord)\(", shell)
    assert not re.search(r"_scalar_shape_page", shell)
    assert not re.search(r"slot_storage_page", precompile)


def test_frame_and_planning_pages_have_no_raw_page_writers():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "compiler"
    shell = (root / "fortran_c_shell.py").read_text(encoding="utf-8")
    precompile = (root / "precompile_to_ssa.py").read_text(encoding="utf-8")
    for page in (
        "numeral_leaf_materialization_concordance",
        "program_abi_keyed_row_record",
    ):
        assert not re.search(
            rf'"{page}"\s*\)\s*\.(concord|revise|set)\(', shell,
        ), page
    # The planning page is bound through the book, never ``bind_alias``.
    assert not re.search(r"value_concordance\.bind_alias\(", precompile)
    # An argument binding with no named cell still stands on its call.
    assert "_frame_call_cells(caller_graph, planned_call.callsite_id)" in shell
