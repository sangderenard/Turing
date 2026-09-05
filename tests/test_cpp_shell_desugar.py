"""Verify the narrow C++-like shell desugars to real, parseable C.

See CPP_LIKE_SHELL_FOR_C_INTENT.md for scope and why this exists. Every
test here must confirm the desugared output actually parses via the real,
existing pycparser path -- "produces text" is not enough, matching the
verification discipline used for the JavaScript slice
(oop_language_translations.py / test_oop_language_translations.py).
"""

import re

from pycparser import c_parser

from src.compiler.cpp_shell_desugar import CppShellUnsupported, desugar_cpp_shell


def _parse(c_source: str):
    return c_parser.CParser().parse(c_source)


def test_a_class_with_a_field_and_a_method_desugars_to_parseable_c():
    source = """
    class Counter {
        int value;

        Counter(int start) {
            value = start;
        }

        int bump(int amount) {
            value = value + amount;
            return value;
        }
    };

    int run() {
        Counter c = Counter__new(10);
        int total = c.bump(5);
        return total;
    }
    """
    desugared = desugar_cpp_shell(source)
    ast = _parse(desugared)
    function_names = {
        decl.decl.name for decl in ast.ext
        if type(decl).__name__ == "FuncDef"
    }
    assert function_names == {"Counter__new", "Counter__bump", "run"}
    assert "typedef struct Counter Counter;" in desugared


def test_method_call_site_is_rewritten_to_a_free_function_call():
    source = """
    class Counter {
        int value;
        int bump(int amount) {
            value = value + amount;
            return value;
        }
    };

    int run() {
        Counter c;
        return c.bump(1);
    }
    """
    desugared = desugar_cpp_shell(source)
    assert "Counter__bump(&c, 1)" in desugared
    assert "c.bump(1)" not in desugared
    _parse(desugared)


def test_bare_call_to_an_inherited_method_resolves_through_the_base_class():
    # Found via a real end-to-end torture test through the actual
    # dream_document pipeline, not written in advance: a method calling an
    # *inherited* method with no explicit receiver (implicit ``this``) was
    # silently left as a call to an undefined function, syntactically valid
    # to pycparser but never actually callable.
    source = """
    class Animal {
        int mass;
        int weigh() {
            return mass;
        }
    };

    class Dog : public Animal {
        int score() {
            int w = weigh();
            return w + 1;
        }
    };
    """
    desugared = desugar_cpp_shell(source)
    assert "Animal__weigh(&(self->base))" in desugared
    assert re.search(r"(?<!\w)weigh\(\)", desugared) is None
    ast = _parse(desugared)
    function_names = {
        decl.decl.name for decl in ast.ext
        if type(decl).__name__ == "FuncDef"
    }
    assert {"Animal__weigh", "Dog__score"}.issubset(function_names)


def test_single_inheritance_embeds_the_base_struct():
    source = """
    class Base {
        int id;
    };

    class Derived : public Base {
        int extra;
    };
    """
    desugared = desugar_cpp_shell(source)
    assert "struct Base base;" in desugared
    ast = _parse(desugared)
    struct_names = {
        decl.type.name for decl in ast.ext
        if type(decl).__name__ == "Decl" and type(decl.type).__name__ == "Struct"
    }
    assert {"Base", "Derived"}.issubset(struct_names)


def test_templates_are_rejected_not_silently_misdesugared():
    source = "template <typename T> class Box { T value; };"
    try:
        desugar_cpp_shell(source)
        assert False, "expected CppShellUnsupported"
    except CppShellUnsupported as error:
        assert "template" in str(error)


def test_virtual_functions_are_rejected():
    source = "class Shape { virtual int area() { return 0; } };"
    try:
        desugar_cpp_shell(source)
        assert False, "expected CppShellUnsupported"
    except CppShellUnsupported as error:
        assert "virtual" in str(error)


def test_multiple_inheritance_is_rejected():
    source = "class Both : public A, public B { int x; };"
    try:
        desugar_cpp_shell(source)
        assert False, "expected CppShellUnsupported"
    except CppShellUnsupported as error:
        assert "multiple inheritance" in str(error)
