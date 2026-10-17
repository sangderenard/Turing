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


def test_resident_vocabulary_ids_are_minted_through_the_book():
    """``_string_token_glyph_materializations`` mints each glyph vocabulary
    table's sequence id as a NOVEL ``string_glyph_vocabulary`` row under
    ``string_glyph_vocabulary_mint``, from the cell the table was built from
    (the Enum's source glyph table; for string tokens one ``cell_set`` of
    the ``string_table_entry`` cells and the recipe cells it serves).  A
    direct ``GLOBAL_MONOTONIC_IDS.mint()`` left the id with no mint edge,
    and the SSA composition audit refused the function that declared the
    table as a dependency: ``unsourced-identity`` (``Registry.page``)."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "compiler"
    source = (root / "resident_sequence_materialization.py").read_text(
        encoding="utf-8",
    )
    assert not re.search(
        r"import GLOBAL_MONOTONIC_IDS|GLOBAL_MONOTONIC_IDS\.mint", source,
    )
    assert re.search(
        r"STRING_GLYPH_VOCABULARY, \(source_scope, NEW\)", source,
    )
    assert re.search(
        r"provenance=Novel\(STRING_GLYPH_VOCABULARY_MINT, \(operand,\)\)",
        source,
    )


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


def test_reducer_and_region_pages_have_no_raw_page_writers():
    """The reducer's concordance rows (precision boundaries, packs and
    operators, numeric components and intrinsics, operator dispatch and
    specialization, receiver constructors, type normalization, static
    parameter identity, the return-root lexical reads) and the planner's
    precision regions are posted through the book, DERIVED from the node,
    span and symbol cells each fact was read from.  No ``IdentityPage.set``
    / ``concord`` / ``revise`` call survives in the reducer, and a page
    named by its string there is only ever read."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src"
    reducer = (
        root / "common" / "tensors" / "topological_reducer.py"
    ).read_text(encoding="utf-8")
    raw = [
        reducer.count("\n", 0, match.start()) + 1
        for match in re.finditer(r"\w*page\.(set|concord|revise)\(", reducer)
    ]
    assert raw == [], raw
    written_by_name = [
        reducer.count("\n", 0, match.start()) + 1
        for match in re.finditer(
            r'\.page\(\s*"[^"]+"\s*\)\s*\.(set|concord|revise)\(', reducer,
        )
    ]
    assert written_by_name == [], written_by_name
    planner = (
        root / "compiler" / "glsl_deployment_strategy.py"
    ).read_text(encoding="utf-8")
    assert not re.search(r"region_page\.(set|concord|revise)\(", planner)
    assert 'page("source_precision_region_concordance")' not in planner


def test_precision_pipeline_and_sympy_ingestion_have_no_raw_page_writers():
    """The precision pipeline's channel-shape, declared-formal-width and
    single-input-Phi rows, and every node row the SymPy law ingestion posts,
    go through the book DERIVED from the value, operand, boundary and
    expression cells each fact was read from.  No ``IdentityPage.set`` /
    ``concord`` / ``revise`` call survives in ``ir_identities`` or
    ``symbolic_process_graph``, and neither names a page by its string to
    write it."""
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "compiler"
    for module in ("ir_identities.py", "symbolic_process_graph.py"):
        source = (root / module).read_text(encoding="utf-8")
        raw = [
            source.count("\n", 0, match.start()) + 1
            for match in re.finditer(
                r"\w*page\.(set|concord|revise)\(", source,
            )
        ]
        assert raw == [], (module, raw)
        written_by_name = [
            source.count("\n", 0, match.start()) + 1
            for match in re.finditer(
                r'\.page\(\s*"[^"]+"\s*\)\s*\.(set|concord|revise)\(', source,
            )
        ]
        assert written_by_name == [], (module, written_by_name)
    pipeline = (root / "ir_identities.py").read_text(encoding="utf-8")
    for page in (
        "precision_channel_shape_concordance",
        "precision_declared_formal_width_concordance",
        "single_input_phi_descriptor_concordance",
    ):
        assert f'page("{page}")' not in pipeline, page


def test_precision_and_compile_entry_pages_are_declared_and_classified():
    """The pages this session's writers post are ``declare_page`` objects
    with a spill lifecycle: the precision pipeline's three, the compile
    entry's numeric type / method / parameter pages and the destructuring
    projection page."""
    from src.compiler import concordance_declarations as declared
    from src.compiler.identity_concordance import REGISTRY
    from src.compiler.page_lifecycle import PAGE_LIFECYCLE

    for page in (
        declared.PRECISION_CHANNEL_SHAPE,
        declared.PRECISION_DECLARED_FORMAL_WIDTH,
        declared.SINGLE_INPUT_PHI_DESCRIPTOR,
        declared.SOURCE_NUMERIC_TYPE_DEPENDENCY,
        declared.SOURCE_NUMERIC_METHOD_DEPENDENCY,
        declared.SOURCE_NUMERIC_PARAMETER_ABI,
        declared.SOURCE_NUMERIC_RECORD_ABI,
        declared.SOURCE_NUMERIC_PARAMETER_RECORD_VIEW,
        declared.NUMERAL_LEAF_WIDTH,
        declared.RECORD_FIELD_DEMAND,
        declared.ASSIGNMENT_PROJECTION_SOURCE,
    ):
        assert REGISTRY.pages[page.name] is page
        assert page.name in PAGE_LIFECYCLE, page
    for stage in (
        declared.PRECISION_CARRY, declared.PRECISION_EXPANSION,
        declared.SINGLE_INPUT_PHI_SETTLEMENT,
    ):
        assert REGISTRY.stages[stage.name] is stage


def test_reducer_pages_are_declared_and_classified():
    """Each page the reducer posts is a ``declare_page`` object with the
    ``(numeric scope, value id)`` row shape its readers index by, and has a
    spill lifecycle like its ``source_value_class`` sibling."""
    from src.compiler import concordance_declarations as declared
    from src.compiler.identity_concordance import REGISTRY, RowFieldKind
    from src.compiler.page_lifecycle import PAGE_LIFECYCLE

    numeric_pages = (
        declared.SOURCE_PRECISION_BOUNDARY, declared.SOURCE_PRECISION_PACK,
        declared.SOURCE_PRECISION_OPERATOR, declared.SOURCE_PRECISION_REGION,
        declared.SOURCE_NUMERIC_COMPONENT, declared.SOURCE_NUMERIC_INTRINSIC,
        declared.SOURCE_NUMERIC_OPERATOR_SPECIALIZATION,
        declared.SOURCE_RECEIVER_CONSTRUCTOR, declared.SOURCE_OPERATOR_DISPATCH,
        declared.SOURCE_TYPE_NORMALIZATION,
    )
    for page in numeric_pages:
        assert REGISTRY.pages[page.name] is page
        assert tuple(field.kind for field in page.row_fields) == (
            RowFieldKind.SCOPE, RowFieldKind.VALUE_ID,
        ), page
        assert page.fact_type is tuple, page
        assert page.name in PAGE_LIFECYCLE, page
    parameter = declared.SOURCE_PARAMETER_IDENTITY
    assert REGISTRY.pages[parameter.name] is parameter
    assert tuple(field.kind for field in parameter.row_fields) == (
        RowFieldKind.SCOPE, RowFieldKind.NAME,
    )
    assert parameter.name in PAGE_LIFECYCLE
