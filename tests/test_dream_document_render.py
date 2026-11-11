"""render_dream_document is the exact inverse of parse_dream_document, and refuses what would not read back unchanged."""
from types import MappingProxyType

import pytest

from src.compiler.dream_document import (
    DreamBlock,
    DreamDocument,
    DreamDocumentError,
    DreamParallelDeployment,
    load_dream_document,
    parse_dream_document,
    render_dream_document,
)

ROOT_EXAMPLE = "examples/reversible_chip_simulator.dream"


def block(identity, payload, *, language="python", encoding="utf-8", kind="segment", stage=None, inputs=(), outputs=(),
          **decorations):
    return DreamBlock(identity=identity, kind=kind, language=language, payload=payload, encoding=encoding, stage=stage,
                      inputs=tuple(inputs), outputs=tuple(outputs), decorations=MappingProxyType(dict(decorations)))


def same(a: DreamDocument, b: DreamDocument) -> bool:
    def facts(doc):
        return ([(x.identity, x.kind, x.language, x.payload, x.encoding, x.stage, x.inputs, x.outputs, dict(x.decorations),
                  x.content_hash) for x in doc.blocks],
                [(g.identity, g.members, g.join) for g in doc.parallel])
    return facts(a) == facts(b)


def test_a_document_reads_back_exactly():
    document = DreamDocument(
        (block("setup", "def f(x):\n    return x\n", outputs=("f",), file="a/b.py", note="first part"),
         block("step", "export const g = 1;\n", language="javascript", inputs=("f",), outputs=("g", "h")),
         block("colors", "void main() {}\n", language="glsl", kind="shader", stage="compute", abi="turing.shader-component.v1"),
         block("empty", "")),
        (DreamParallelDeployment("frame", ("step", "colors"), "frame-barrier"),))
    read = parse_dream_document(render_dream_document(document))
    assert [b.payload for b in read.blocks] == [b.payload for b in document.blocks]
    assert read.block("setup").decorations == {"file": "a/b.py", "note": "first part"}
    assert read.block("step").inputs == ("f",) and read.block("step").outputs == ("g", "h")
    assert read.block("colors").stage == "compute" and read.parallel[0].members == ("step", "colors")
    assert all(len(b.content_hash) == 64 for b in read.blocks)
    assert render_dream_document(read) == render_dream_document(document), "rendering is stable"


@pytest.mark.parametrize("payload", ["ends with a return\r", "\r\nstarts with a line break", "\n\nblank lines\n\n",
                                     "crlf inside\r\nand at the end\r\n", "\r", "\n", "no line break at all"])
def test_line_breaks_at_the_edges_of_a_payload_survive(payload):
    read = parse_dream_document(render_dream_document(DreamDocument((block("p", payload),))))
    assert read.block("p").payload == payload


def test_any_bytes_are_carried_under_latin_1():
    raw = bytes(range(256)) * 3 + b"\x00asm\x01\x00\x00\x00\r"
    read = parse_dream_document(render_dream_document(
        DreamDocument((block("binary", raw.decode("latin-1"), language="wasm", encoding="latin-1"),))))
    assert read.block("binary").payload.encode("latin-1") == raw


def test_the_sha256_sentinel_is_written_and_checked():
    rendered = render_dream_document(DreamDocument((block("p", "x = 1\n"),)))
    assert b"sha256 = " in rendered
    with pytest.raises(DreamDocumentError, match="sha256"):
        parse_dream_document(rendered.replace(b"x = 1", b"x = 2"))


@pytest.mark.parametrize("bad", [
    block("p", "text /*@turing.end*/ more"),
    block("p", "x", note="two\nlines"),
    block("p", "x", note="café"),
    block("p", "x", note=" padded "),
    block("p", "x", note="ends the header @end*/ early"),
    block("p", "x", language="python", sha256="0" * 64),
    block("p", "x", inputs=("a,b",)),
    block("p", "x", kind="shader", language="glsl", stage=None),
    block("p", "é", encoding="ascii"),
    block("", "x"),
    block("p", "x", language=""),
])
def test_what_would_not_read_back_is_refused(bad):
    with pytest.raises(DreamDocumentError):
        render_dream_document(DreamDocument((bad,)))


def test_the_example_document_survives_a_round_trip():
    original = load_dream_document(ROOT_EXAMPLE)
    assert same(parse_dream_document(render_dream_document(original)), original)
