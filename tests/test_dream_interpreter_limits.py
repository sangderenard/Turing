"""Pin the dream interpreter's measured limits and the 2026-11-11 repairs.

Each test names one fact from docs/DREAM_INTERPRETER_LIMITS.md so that a
change in the interpreter's reach is a test change, not a surprise.
"""

import pytest
from pycparser import c_parser

from src.compiler.cpp_shell_desugar import CppShellUnsupported, desugar_cpp_shell
from src.compiler.dream_document import DreamDocumentError, parse_dream_document
from src.compiler.source_control_ingestion import UnadmittedSourceControlTopology


def _segment(language, source, identity="p"):
    return parse_dream_document(
        f"/*@turing.segment.v1\nid={identity}\nlanguage={language}\n@end*/\n"
        f"{source}\n/*@turing.end*/"
    )


def _shader(source, stage="fragment"):
    return parse_dream_document(
        f"/*@turing.shader.v1\nid=p\nlanguage=glsl\nstage={stage}\n@end*/\n"
        f"{source}\n/*@turing.end*/"
    )


PLAIN = "def g(a, b):\n    if a > b:\n        return a - b\n    return b - a\nresult = g(3, 1)\n"
WITH = "def f():\n    with open('x') as h:\n        return h.read()\n"


def test_python_route_refuses_exceptions_and_resource_scopes_by_name():
    for source, kind in (
        ("def f(x):\n    if x < 0:\n        raise ValueError('neg')\n    return x\n", "Raise"),
        ("def f(x):\n    try:\n        return 1 / x\n    except ZeroDivisionError:\n        return 0\n", "Try"),
    ):
        with pytest.raises(UnadmittedSourceControlTopology, match=kind):
            _segment("python", source).compile_sections()
    with pytest.raises(Exception, match="native resource scope"):
        _segment("python", WITH).compile_sections()


def test_a_refused_block_does_not_poison_the_next_block_in_the_same_process():
    # Measured 2026-11-11: a `with` refusal left rows in the shared detached
    # identity book and every later Python block failed with a
    # native_exception_control_gap disagreement.  Each block now has its own book.
    with pytest.raises(Exception, match="native resource scope"):
        _segment("python", WITH, "a").compile_sections()
    (first,) = _segment("python", PLAIN, "b").compile_sections()
    (second,) = _segment("python", PLAIN, "c").compile_sections()
    assert first.graph["functions"][0]["state"] == "resolved"
    assert second.graph["functions"][0]["state"] == "resolved"


def test_python_route_admits_the_ordinary_constructs():
    sources = {
        "generator": "def g(n):\n    for i in range(n):\n        yield i\n",
        "comprehension": "def f(xs):\n    return [x * 2 for x in xs if x > 0]\n",
        "closure": "def mk(k):\n    def add(x):\n        return x + k\n    return add\n",
        "class": "class C:\n    def __init__(self, v):\n        self.v = v\n    def dbl(self):\n        return self.v * 2\n",
        "match": "def f(x):\n    match x:\n        case 1:\n            return 'one'\n        case _:\n            return 'other'\n",
    }
    for name, source in sources.items():
        (compilation,) = _segment("python", source).compile_sections()
        assert compilation.route == "ast", name
        assert all(f["state"] == "resolved" for f in compilation.graph["functions"]), name


def test_cpp_shell_drops_includes_refuses_other_directives_and_rewrites_constructors():
    parsed = desugar_cpp_shell("#include <string>\nint f() { return 1; }\n")
    assert "#include" not in parsed
    c_parser.CParser().parse(parsed)

    with pytest.raises(CppShellUnsupported, match="preprocessor directives"):
        desugar_cpp_shell("#define TWO 2\nint f() { return TWO; }\n")

    parsed = desugar_cpp_shell(
        "class C {\n    int v;\n    C(int value) { v = value; }\n"
        "    int get() { return v; }\n};\nint use() { C c(3); return c.get(); }\n"
    )
    assert "C c = C__new(3);" in parsed and "C__get(&c)" in parsed
    c_parser.CParser().parse(parsed)

    # An unknown receiver is left as written, never rewritten to None__method.
    parsed = desugar_cpp_shell("struct Q { int v; };\nint use(struct Q* q) { return q.get(); }\n")
    assert "q.get()" in parsed and "None__" not in parsed


def test_cpp_route_stops_at_the_operand_position_gate_not_at_the_parser():
    # The shipped example now parses; the route then stops in the reducer's
    # operand-position concordance gate.  This pins where the C route ends
    # today so a change there is noticed (see DREAM_INTERPRETER_LIMITS.md).
    document = _segment("cpp", "int add(int a, int b) { return a + b; }\n")
    with pytest.raises(ValueError, match="operand-position scope"):
        document.compile_sections()


def test_glsl_route_lowers_straight_line_scalar_main_without_shortfall():
    (compilation,) = _shader(
        "uniform float a;\nuniform float b;\nout float o;\n"
        "void main() {\n    float c = a * b + 1.0;\n    float d = mix(a, b, 0.5);\n"
        "    o = clamp(c + d, 0.0, 1.0);\n}\n"
    ).compile_sections()
    assert compilation.shortfalls == ()
    assert sum(node["type"] == "SSAInstruction" for node in compilation.graph["nodes"]) > 0

    (compilation,) = _shader(
        "uniform float a;\nout float o;\nvoid main() {\n"
        "    float t = 0.0;\n    for (int i = 0; i < 4; i++) { t += a; }\n    o = t;\n}\n"
    ).compile_sections()
    assert any("GLSL_CONTROL" in item for item in compilation.shortfalls)

    (compilation,) = _shader(
        "float main(float a) {\n    return a;\n}\n"
    ).compile_sections()
    assert any("no void entry function" in item for item in compilation.shortfalls)


def test_javascript_route_sees_exported_and_destructured_declarations():
    (compilation,) = _segment(
        "javascript",
        "import {x} from './m.js';\nexport function f() { return g(); }\n"
        "export default class K {}\nfunction g() { return 1; }\n"
        "const {a, b: c} = {a: 1, b: 2};\nlet [d, ...rest] = [a];\n",
    ).compile_sections()
    names = {node["declaration_name"] for node in compilation.graph["nodes"]}
    assert names == {"f", "K", "g", "a", "c", "d", "rest"}
    assert "x" not in names  # an import binds a name declared elsewhere
    assert any(edge["from"] != edge["to"] for edge in compilation.graph["edges"])


def test_javascript_route_fails_closed_on_a_syntax_error():
    with pytest.raises(DreamDocumentError, match="failed to parse"):
        _segment("javascript", "function ( {\n").compile_sections()
