"""Source discovery of a dependency class's imports and field declarations.

Program: the AbstractTensor octet codec compiled as a driver.  Its
``CanonicalHuffmanTable`` methods reach a module whose guarded imports name
a backend that cannot load here, and its optional ``symbols`` field is
declared under the class's qualified identity.
"""

from __future__ import annotations

import ast
import sys

from src.compiler.concordance_declarations import (
    CLASS_FIELD_DECLARATION, FieldFact, INGESTION,
)
from src.compiler.frame_identity_book import _record_field_declaration_cell
from src.compiler.identity_concordance import (
    Mode, Novel, begin_identity_book, declare_transform,
)
from src.transmogrifier.graph.graph_express2 import _import_ast_bindings


def test_a_module_that_fails_to_import_binds_nothing(tmp_path, monkeypatch):
    # ``torch_backend`` raised AttributeError (``torch.Tensor`` on
    # ``torch = None``); discovery caught only ImportError and the whole
    # lowering refused.
    (tmp_path / "broken_backend_probe.py").write_text(
        "missing = None\nvalue = missing.Tensor\n", encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    tree = ast.parse(
        "try:\n"
        "    import broken_backend_probe\n"
        "    from broken_backend_probe import value\n"
        "except Exception:\n"
        "    pass\n"
        "import ast as syntax\n"
    )
    bindings = _import_ast_bindings(tree, {})
    assert "broken_backend_probe" not in bindings and "value" not in bindings
    assert bindings["syntax"] is ast
    assert "broken_backend_probe" not in sys.modules


ROOT = declare_transform("dependency_class_discovery_probe", 0)


def _declare(book, module, owner, field):
    book.post(
        CLASS_FIELD_DECLARATION, (module, owner, field),
        FieldFact("class", "AbstractTensor | None", ()),
        stage=INGESTION, provenance=Novel(ROOT, ()), mode=Mode.CONCORD,
    )


def test_a_bare_record_identity_finds_its_qualified_declaration():
    book, _token = begin_identity_book()
    module = "src.common.tensors.compression.huffman"
    _declare(book, module, f"{module}.CanonicalHuffmanTable", "symbols")
    cell = _record_field_declaration_cell(
        {"identity": "CanonicalHuffmanTable"}, "symbols",
    )
    assert cell is not None and cell.row[1].endswith(".CanonicalHuffmanTable")


def test_a_bare_identity_two_classes_share_names_nothing():
    book, _token = begin_identity_book()
    for module in ("one.place", "other.place"):
        _declare(book, module, f"{module}.Table", "symbols")
    assert _record_field_declaration_cell({"identity": "Table"}, "symbols") is None
