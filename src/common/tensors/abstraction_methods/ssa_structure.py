"""SSA structural mirrors reachable from ``AbstractTensor``.

The SSA module already holds a program's own shape as tables --
``SSAClassDefinition``, ``SSAClassField``, ``SSAClassMethod``,
``SSAClassTable``, and the function table. That data travels *alongside* a
program, so a definition could be transported but never stated from this side
of the boundary.

These mirrors close that. They are deliberately redundant for anyone writing
ordinary tensor code: you define a class with ``class`` and a function with
``def``, and nothing here is a better way to do that. They exist so a complete
program -- structure included -- can cross a suite boundary in the shared
vocabulary rather than as an out-of-band attachment, which is what makes a
definition round-trip bit-exactly. An authored constant inside a method body,
a custom trig epsilon being the obvious case, survives only if the method it
lives in survives as a definition rather than as a name.

Nothing is re-declared here. The constructors return the real repository SSA
objects, so these mirrors cannot drift from the structures they mirror. The
matching operator names in the shared catalog are ``class_define``,
``field_define``, ``method_define``, ``function_define``, ``getattr`` and
``setattr`` (turing ``Handler`` members ``ClassDefine``, ``FieldDefine``,
``MethodDefine``, ``FunctionDefine``, ``GetAttr``, ``SetAttr``).

The import is deferred to call time on purpose: ``transmogrifier``'s package
import pulls in the graph/simulator stack, and this module is reached from
``abstraction``, which must stay importable without paying for that.
"""

from __future__ import annotations

from typing import Any, Iterable


def _ssa():
    from ....transmogrifier import ssa as _module

    return _module


def _handler():
    from ....transmogrifier.ssa_registry import Handler

    return Handler


def define_field(name: str, slot: int):
    """One instance field's addressable slot in a class layout.

    Mirrors ``SSAClassField``; the catalog operator is ``field_define``.
    """

    return _ssa().SSAClassField(name=str(name), slot=int(slot))


def define_method(
    name: str, function_reference: int, function_name: str | None = None
):
    """Bind a method name on a class to the function implementing it.

    Mirrors ``SSAClassMethod``; the catalog operator is ``method_define``.
    The binding is what carries the class/function relationship across a
    translation, which a bare function name cannot.
    """

    return _ssa().SSAClassMethod(
        name=str(name),
        function_reference=int(function_reference),
        function_name=None if function_name is None else str(function_name),
    )


def define_class(
    identity: str,
    fields: Iterable[Any] = (),
    methods: Iterable[Any] = (),
):
    """State a class: its identity, field layout and method bindings.

    Mirrors ``SSAClassDefinition``; the catalog operator is ``class_define``.
    """

    return _ssa().SSAClassDefinition(
        identity=str(identity),
        fields=tuple(fields),
        methods=tuple(methods),
    )


def class_table(classes: Iterable[Any] = ()):
    """Every class definition carried in one module.

    Mirrors ``SSAClassTable``. Holding definitions rather than reference
    lookups is what lets a backend emit a class's methods as real,
    individually linkable functions.
    """

    return _ssa().SSAClassTable(classes=tuple(classes))


def accessor(kind: str = "get"):
    """The ``Handler`` member naming one half of the field accessor pair.

    ``get`` resolves a named field against the record/class descriptor;
    ``set`` writes it. The pair exists as named-field operations rather than
    slot loads and stores precisely so the field's *name* -- and therefore its
    meaning -- survives translation.
    """

    handler = _handler()
    selected = str(kind).casefold()
    if selected in {"get", "getattr", "read"}:
        return handler.GetAttr
    if selected in {"set", "setattr", "write"}:
        return handler.SetAttr
    raise ValueError(
        f"unknown accessor {kind!r}; expected 'get' or 'set'"
    )


class _SSAStructureNamespace:
    """Nested namespace so these never crowd the tensor method surface.

    Reached as ``AbstractTensor.ssa``. Keeping them one attribute deeper is
    the point: they are program-structure vocabulary, not tensor operations,
    and an author browsing tensor methods should not meet them.
    """

    __slots__ = ()

    define_class = staticmethod(define_class)
    define_field = staticmethod(define_field)
    define_method = staticmethod(define_method)
    class_table = staticmethod(class_table)
    accessor = staticmethod(accessor)

    @staticmethod
    def define_function(name: str, reference: int | None = None):
        """State a function or closure body as a definition.

        The catalog operator is ``function_define``. A function that crosses a
        boundary only as a call target arrives without its body; stated as a
        definition it arrives whole, including the constants inside it.
        """

        return {
            "kind": "function_define",
            "name": str(name),
            "function_reference": None if reference is None else int(reference),
        }

    @staticmethod
    def handler(name: str):
        """Resolve a structural operator name to its ``Handler`` member."""

        return getattr(_handler(), str(name))

    def __repr__(self) -> str:  # pragma: no cover - diagnostic only
        return "<AbstractTensor.ssa structural definition mirrors>"


SSA_STRUCTURE = _SSAStructureNamespace()


__all__ = [
    "SSA_STRUCTURE",
    "accessor",
    "class_table",
    "define_class",
    "define_field",
    "define_method",
]
