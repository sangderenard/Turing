"""Projected source classes keep their authored declaration identities."""

import ast
import copy
import inspect

import pytest

from src.common.tensors.compression.bitstream import PackedBitstream
from src.compiler.concordance_declarations import (
    CLASS_DECLARATION, CLASS_FIELD_DECLARATION, SOURCE_RECORD_VIEW_SCHEMA,
)
from src.compiler.fortran_c_shell import _authored_dataclass_record_views
from src.compiler.identity_concordance import begin_identity_book, end_identity_book
from src.transmogrifier.graph.graph_express2 import (
    _class_schema_from_ast, _stamp_reference_class_identity,
)


def test_projected_packet_schema_keeps_authored_field_cells():
    book, token = begin_identity_book()
    try:
        definition = ast.parse(inspect.getsource(PackedBitstream)).body[0]
        _stamp_reference_class_identity(PackedBitstream, definition)
        original = _class_schema_from_ast(definition)
        row = (PackedBitstream.__module__, PackedBitstream.__qualname__)
        assert original["class_identity"] == row[1]
        fields = {
            field: book.latest_ref(CLASS_FIELD_DECLARATION, (*row, field))
            for field in ("octets", "valid_bits", "symbol_offsets", "symbol_lengths")
        }
        assert all(fields.values())

        # Run54's pursued ClassDef contains the original field statements,
        # without the docstring/methods, and carries this same live identity.
        projected = copy.deepcopy(definition)
        projected.body = [item for item in projected.body
                          if isinstance(item, ast.AnnAssign)]
        projected._python_source_identity = row
        schemas = {}
        _authored_dataclass_record_views(
            ast.Module(body=[projected], type_ignores=[]),
            projection_schemas=schemas, carried_declarations=True,
        )
        assert schemas[PackedBitstream.__name__]["source_record_view_schema"] == row
        schema_ref = book.latest_ref(SOURCE_RECORD_VIEW_SCHEMA, row)
        assert schema_ref is not None
        schema_causes = {source for source, _ in book.edges_into(schema_ref)}
        for field, authored in fields.items():
            assert book.latest_ref(CLASS_FIELD_DECLARATION, (*row, field)) == authored
            assert authored in schema_causes
            assert authored not in {source for source, _ in book.edges_into(authored)}
        assert book.latest_ref(
            CLASS_DECLARATION, (row[0], ".".join(row)),
        ) is None
    finally:
        end_identity_book(token)


@pytest.mark.parametrize("module,qualname", (
    ("first_module", "Outer.Payload"),
    ("second_module", "Outer.Payload"),
    ("first_module", "Other.Payload"),
))
def test_pursued_class_keeps_lexical_qualification(module, qualname):
    book, token = begin_identity_book()
    try:
        definition = ast.parse("class Payload:\n    count: int\n").body[0]
        definition._turing_source_module = module
        definition._turing_source_qualname = qualname
        _class_schema_from_ast(definition)
        row = (module, qualname)
        declaration = book.latest_ref(CLASS_DECLARATION, row)
        field = book.latest_ref(CLASS_FIELD_DECLARATION, (*row, "count"))
        assert declaration is not None and field is not None
        definition._python_source_identity = row
        pursued = _class_schema_from_ast(definition, carried_declarations=True)
        # Map IR needs its global object label; the declaration has the same
        # source owner and field cells under either source representation.
        assert pursued["class_identity"] == f"{module}.{qualname}"
        assert pursued["class_node_id"][:2] == row
        assert book.latest_ref(CLASS_DECLARATION, row) == declaration
        assert book.latest_ref(CLASS_FIELD_DECLARATION, (*row, "count")) == field
        assert book.latest_ref(CLASS_DECLARATION,
                               (module, pursued["class_identity"])) is None
    finally:
        end_identity_book(token)
