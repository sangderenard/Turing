"""The operator that recovers argument POSITION from an edge role.

A ProcessGraph node's ``parents`` is a set of ``(node, role)`` pairs. It is
not an ordered argument list, and it must not be read as one: the set's
iteration order is an artefact of construction, while the role string is
the authoritative statement of what that operand *is* -- ``arg:0``,
``lhs``, ``value``, ``kw:name``.

Position is therefore recoverable, but only through the role. Code that
zips ``parents`` against a parameter list, or slices it, is making a
positional assumption the graph never promised, and this tree has paid for
that assumption repeatedly.

Rather than ask every consumer to stop thinking positionally -- which is
how people naturally read a call -- this supplies the translation, so
ordinary positional code stays ordinary and stays correct. Both spellings
the ProcessGraph emits (``arg:0`` and ``arg0``) normalise here, in one
place, so a consumer never has to know there were two.
"""
from __future__ import annotations

from typing import Any, Iterable

#: Roles that name an operand by meaning rather than by position. A
#: consumer wanting "the arguments" must not silently swallow these: a
#: receiver or a callee reference is not argument zero.
NON_POSITIONAL_ROLES = frozenset({
    "callee", "func", "definition", "self", "cls", "receiver",
})


def positional_argument_index(role: Any) -> int | None:
    """The 0-based argument position a role names, or None if it names none.

    Normalises the two spellings the ProcessGraph uses for the same thing.
    Returning None is meaningful and must not be treated as 0: it says the
    edge is not positional at all.
    """
    text = str(role)
    if text.startswith("arg:") and text[4:].isdigit():
        return int(text[4:])
    if text.startswith("arg") and text[3:].isdigit():
        return int(text[3:])
    return None


def keyword_argument_name(role: Any) -> str | None:
    """The keyword a role names, or None. ``kw:alpha`` -> ``alpha``."""
    text = str(role)
    if text.startswith("kw:") and len(text) > 3:
        return text[3:]
    return None


def ordered_arguments(
    parents: Iterable[tuple[Any, Any]],
) -> tuple[Any, ...]:
    """The positional operands of a node, in argument order.

    This is the translation that lets positional code be written
    positionally: hand it a node's ``parents`` and receive the arguments in
    the order the source wrote them, with non-positional edges (the callee
    reference, a receiver, keywords) left out rather than silently
    occupying a position.

    Ordering is by the declared index, so a graph that yields its parents
    in any order produces the same argument list.
    """
    indexed: list[tuple[int, Any]] = []
    for parent, role in parents or ():
        position = positional_argument_index(role)
        if position is not None:
            indexed.append((position, parent))
    indexed.sort(key=lambda item: item[0])
    return tuple(parent for _position, parent in indexed)


def keyword_arguments(
    parents: Iterable[tuple[Any, Any]],
) -> dict[str, Any]:
    """The keyword operands of a node, by name."""
    found: dict[str, Any] = {}
    for parent, role in parents or ():
        name = keyword_argument_name(role)
        if name is not None:
            found[name] = parent
    return found


def is_positional(role: Any) -> bool:
    """Whether this role names an argument position."""
    return positional_argument_index(role) is not None


__all__ = [
    "NON_POSITIONAL_ROLES",
    "positional_argument_index",
    "keyword_argument_name",
    "ordered_arguments",
    "keyword_arguments",
    "is_positional",
]
